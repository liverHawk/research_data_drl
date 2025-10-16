from dataclasses import dataclass
# ========================
# Qネットワーク最適化用パラメータの型定義
# ========================
@dataclass
class OptimizeModelParams:
    BATCH_SIZE: int = 128
    GAMMA: float = 0.999
    memory: object = None
    policy_net: object = None
    target_net: object = None
    optimizer: object = None
    scaler: object = None
    include_category: bool = True  # カテゴリ変数を含むかどうか

# ========================
# 行動選択用パラメータの型定義
# ========================
@dataclass
class SelectActionParams:
    EPS_END: float = 0.05
    EPS_START: float = 0.9
    EPS_DECAY: int = 200
    policy_net: object = None
    n_actions: int = None
    steps_done: int = 0


import csv
import os
import sys
import yaml
import random
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
import mlflow
import cProfile
import pstats
import threading

from itertools import count
from glob import glob
from torch.amp import GradScaler
from tqdm import tqdm
from queue import Queue
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

import flow_package as fp
from flow_package.multi_df_env import MultiDfEnv, EnvConfig

from utils import setup_logging, rolling_normalize
from network import DeepFlowNetwork
from network_v2 import DeepFlowNetworkV2
from deep_learn import ReplayMemory, Transaction

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"


def setup_mlflow(all_params):
    if all_params["mlflow"]["use_azure"]:
        path = os.path.join(os.path.dirname(__file__), "..", "config.json")
        print(path)
        ml_client = MLClient.from_config(
            credential=DefaultAzureCredential(),
            config_path=path
        )
        mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    elif all_params["mlflow"]["use_dagshub"]:
        import dagshub
        dagshub.init(repo_owner='liverHawk', repo_name='research_data_drl', mlflow=True)
    mlflow_tracking_uri = all_params["mlflow"]["tracking_uri"]
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(
        f"{all_params['mlflow']['experiment_name']}_train_drl"
    )


# ========================
# 不変変数設定
# ========================
F_LOSS = nn.MSELoss()

# ========================
# デバイス設定
# ========================
if torch.cuda.is_available():
    device = torch.device("cuda:1")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ========================
# ディレクトリ作成
# ========================
def make_dirs():
    os.makedirs("train/plots", exist_ok=True)
    os.makedirs("train", exist_ok=True)
    os.makedirs("models", exist_ok=True)


# ========================
# パラメータ・入力パスのロード
# ========================
def load_params():
    if len(sys.argv) != 2:
        print("Usage: python src/train_drl.py <input_file_directory>")
        sys.exit(1)

    input_path = sys.argv[1]
    all_params = yaml.safe_load(open("params.yaml"))

    setup_mlflow(all_params)
    # params = all_params["train_drl"]
    return all_params, input_path


# ========================
# データのロード・正規化
# ========================
def load_csv(input):
    files = glob(os.path.join(input, "*.csv.gz"))
    df = pd.concat([
        pd.read_csv(f) for f in files
    ])

    df = df.reset_index(drop=True)

    # 特徴量の正規化
    df = rolling_normalize(df)

    return df


# ========================
# バッチ状態のアンパック
# ========================
def _unpack_state_batch(state_batch, include_category=False):
    """
    state_batch: list of states (each state は include_category=True の場合 (port, protocol, features) のタプル)
    -> include_category=True の場合は (port_batch, protocol_batch, features_batch) のタプルを返す
    -> include_category=False の場合は (batch_size, n_states) の 2D Tensor を返す
    """
    # state_batch が空の可能性は呼び出し元で弾く想定
    if include_category:
        # 各要素を個別にバッチ化してタプルで返す
        port_batch = torch.stack([s[0].view(-1) for s in state_batch], dim=0).to(device).long()
        protocol_batch = torch.stack([s[1].view(-1) for s in state_batch], dim=0).to(device).long()
        features_batch = torch.stack([s[2].view(-1) for s in state_batch], dim=0).to(device).float()
        return (port_batch, protocol_batch, features_batch)
    else:
        # 非カテゴリ版も各サンプルを stack して (B, n_states) に
        per_samples = [s.view(-1) for s in state_batch]
        return torch.stack(per_samples, dim=0).to(device).float()


def select_action(state_tensor: torch.Tensor, params: SelectActionParams):
    """
    ε-greedy法による行動選択
    """
    sample = random.random()
    eps_threshold = params.EPS_END + (params.EPS_START - params.EPS_END) * np.exp(-1. * params.steps_done / params.EPS_DECAY)
    # state_tensorはリスト形式（[port, protocol, other]）
    if sample > eps_threshold:
        with torch.no_grad():
            q_vals = params.policy_net(state_tensor)
            # ネットワークが (n_actions,) を返す場合にバッチ次元 (1, n_actions) を追加
            if q_vals.dim() == 1:
                q_vals = q_vals.unsqueeze(0)
            return q_vals.max(1).indices.view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(params.n_actions)]],
            dtype=torch.long,
            device=device
        )


def optimize_model(params: OptimizeModelParams):
    """
    経験リプレイからバッチをサンプリングし、Qネットワークを最適化
    詳細なメトリクスを返す
    """
    transitions = params.memory.sample(params.BATCH_SIZE)
    batch = Transaction(*zip(*transitions))
    # バッチを (B, n_states) のテンソルに変換（float32, device）
    state_batch = _unpack_state_batch(batch.state, params.include_category)

    action_batch = torch.cat(batch.action).to(device).long()
    reward_batch = torch.cat([r.view(-1) for r in batch.reward]).to(device).float()

    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
    non_final_next_states = [s for s in batch.next_state if s is not None]

    # next_state_batch が空の場合は None にして処理をスキップする
    if len(non_final_next_states) > 0:
        next_state_batch = _unpack_state_batch(non_final_next_states, params.include_category)
    else:
        next_state_batch = None

    # 損失計算
    state_action_values = params.policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(params.BATCH_SIZE, device=device)
    if next_state_batch is not None:
        with torch.no_grad():
            next_q = params.target_net(next_state_batch)
            next_state_values[non_final_mask] = next_q.max(1).values.float()
    expected_state_action_values = reward_batch + params.GAMMA * next_state_values
    loss = F_LOSS(state_action_values, expected_state_action_values.unsqueeze(1))

    # 最適化
    params.optimizer.zero_grad()
    if params.scaler is not None:
        params.scaler.scale(loss).backward()
        utils.clip_grad_value_(params.policy_net.parameters(), 1000)
        params.scaler.step(params.optimizer)
        params.scaler.update()
    else:
        loss.backward()
        utils.clip_grad_value_(params.policy_net.parameters(), 1000)
        params.optimizer.step()
    
    return loss.item()


def to_tensor(state, include_category=True):
    global device
    if include_category:
        return fp.to_tensor(state, device=device)
    else:
        return torch.tensor(state, device=device, dtype=torch.float32)


# ========================
# バックグラウンドメトリクスロガー
# ========================
import signal

class MetricsLogger:
    """バックグラウンドスレッドでメトリクスをMLflowに送信"""
    
    def __init__(self, logger, total_episodes):
        self.queue = Queue()
        self.metrics_history = []
        self.running = False
        self.logger = logger
        self.thread = threading.Thread(target=self._worker, daemon=False)
        # self.progress_bar = tqdm(total=total_episodes, desc="Logging Metrics", position=0)
        # self.thread.start()
    
    def _worker(self):
        """バックグラウンドでメトリクスを処理"""
        while self.running:
            try:
                item = self.queue.get(timeout=0.5)
                if item is None:  # 終了シグナル
                    break
                
                episode, metrics = item
                start_time = time.time()

                # メトリクスを履歴に保存
                metrics_with_episode = {"episode": episode, **metrics}
                self.metrics_history.append(metrics_with_episode)
                
                # MLflowにログを送信（非ブロッキング）
                try:
                    elapsed_time = time.time() - start_time
                    with open("train/metrics.csv", "a") as f:
                        writer = csv.writer(f)
                        if "loss" in metrics and metrics["loss"] is not None:
                            writer.writerow([episode, metrics["reward"], metrics["accuracy"], metrics["steps"], metrics["loss"], elapsed_time])
                        else:
                            writer.writerow([episode, metrics["reward"], metrics["accuracy"], metrics["steps"], "", elapsed_time])

                except Exception as e:
                    self.logger.warning(f"MLflow logging failed: {e}")
                
                # キューのタスク完了を通知
                self.queue.task_done()
                
                # 処理済みエピソード数を更新し、進捗を表示
                # self.processed_count += 1
                # self.progress_bar.update(1)
                
            except Exception as e:
                if self.queue.qsize() == 0:
                    time.sleep(1000)
                    continue
                continue
        
        return
    
    def log(self, episode, metrics):
        """メトリクスをキューに追加（非ブロッキング）"""
        if not self.running:
            self.running = True
            self.thread.start()
        self.queue.put((episode, metrics))
        if not self.running:
            self.running = True
            self.thread.start()
            return
    
    def shutdown(self):
        """ロガーを終了してメトリクスを保存"""
        self.logger.info("Waiting for all metrics to be logged...")
        self.logger.info(f"Current queue size: {self.queue.qsize()}")
        self.running = False
        self.queue.put(None)
        
        # スレッドの終了を待つ
        self.thread.join(timeout=10)  # 最大10秒待つ
        if self.thread.is_alive():
            self.logger.warning("Metrics logger thread did not finish in time")
        else:
            self.logger.info("Metrics logger thread finished successfully")
        self.thread
        
        # プログレスバーを閉じる
        # self.progress_bar.close()

        # 全メトリクスをCSVに保存
        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            os.makedirs("train", exist_ok=True)
            csv_path = "train/metrics.csv"
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Metrics saved to {csv_path} ({len(self.metrics_history)} episodes)")
            
            # MLflowにアーティファクトとして保存
            try:
                mlflow.log_artifact(csv_path)
                self.logger.info("Metrics CSV uploaded to MLflow")
            except Exception as e:
                self.logger.warning(f"Failed to log metrics CSV to MLflow: {e}")
        if mlflow.active_run() is not None:
            mlflow.end_run()
        
        return self.metrics_history

    def _handle_exit(self, signum, frame):
        """強制終了時にスレッドを安全に終了"""
        self.logger.info("Received termination signal. Shutting down...")
        self.shutdown()


def write_result(cm_memory, episode, n_output):
    # 混同行列の計算
    cm = np.zeros((n_output, n_output), dtype=int)
    # print(cm_memory)
    for log in cm_memory:
        action = int(log[0])
        answer = int(log[1])
        cm[action][answer] += 1
    with open("log/train_result.log", "a") as f:
        f.write(f"Episode {episode}:\n")
        f.write(f"{cm}\n")
    total = cm.sum()
    accuracy = float(cm.diagonal().sum() / total) if total > 0 else 0.0
    return cm, accuracy


def train(df, params):
    drl_options = params.get("drl_options", {})
    params = params.get("train_drl", {})

    logger = setup_logging("log/train_drl.log")
    logger.info("Starting training...")
    
    # バックグラウンドメトリクスロガーを初期化
    metrics_logger = MetricsLogger(logger, params["n_episodes"])
    
    label_count = len(df["Label"].unique())
    reward_matrix = np.ones((label_count, label_count)) * -1.0
    np.fill_diagonal(reward_matrix, 1.0)

    columns = df.columns.tolist()
    columns.remove("Label")

    input = EnvConfig(
        data=df,
        label_column="Label",
        render_mode=None,
        window_size=drl_options.get("window_size", 10),
        max_steps=drl_options.get("max_steps", 100),
        normalize_method="rolling",
        rolling_window=drl_options.get("rolling_window", 10),
    )
    mlflow.log_params({
        "window_size": input.window_size,
        "max_steps": input.max_steps,
        "normalize_method": input.normalize_method,
        "rolling_window": input.rolling_window,
    })
    env = MultiDfEnv(input)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    include_category = params.get("include_category", True)

    logger.info(f"State space: {n_states}, Action space: {n_actions}")

    if include_category:
        logger.info("Using DeepFlowNetwork")
        logger.info(f"State space: {n_states}, Action space: {n_actions}")
        policy_net = DeepFlowNetwork(n_states, n_actions).to(device)
        target_net = DeepFlowNetwork(n_states, n_actions).to(device)
    else:
        logger.info("Using DeepFlowNetworkV2")
        logger.info(f"State space: {n_states}, Action space: {n_actions}")
        policy_net = DeepFlowNetworkV2(n_states, n_actions).to(device)
        target_net = DeepFlowNetworkV2(n_states, n_actions).to(device)

    target_net.load_state_dict(policy_net.state_dict())

    cm_memory = []

    optimizer = optim.Adam(policy_net.parameters(), lr=params["lr"])
    # Learning rate scheduler（オプション）
    use_scheduler = params.get("use_lr_scheduler", False)
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.get("lr_step_size", 100), gamma=params.get("lr_gamma", 0.95))
    else:
        scheduler = None
    
    memory = ReplayMemory(params["memory_size"])

    # select_params, optimize_params = get_args(params, )
    select_params = SelectActionParams(
        EPS_END=params.get("eps_end", 0.05),
        EPS_START=params.get("eps_start", 0.9),
        EPS_DECAY=params.get("eps_decay", 200),
        policy_net=policy_net,
        n_actions=n_actions,
        steps_done=0
    )
    optimize_params = OptimizeModelParams(
        BATCH_SIZE=params.get("batch_size", 128),
        GAMMA=params.get("gamma", 0.999),
        memory=memory,
        policy_net=policy_net,
        target_net=target_net,
        optimizer=optimizer,
        scaler=GradScaler() if torch.cuda.is_available() else None,
        include_category=include_category
    )
    
    # Target network更新頻度
    target_update_freq = params.get("target_update_freq", 10)


    for i_episode in tqdm(range(params["n_episodes"])):
        random.seed(i_episode)

        while True:
            initial_state, info = env.reset()
            if info["sample_data_length"] <= 10_000:
                break
        # logger.info(info["sample_data_length"])

        state = to_tensor(initial_state, include_category)

        episode_reward = 0.0
        episode_steps = 0
        episode_losses = []

        p_bar = tqdm(total=int(info["sample_data_length"]), desc=f"Episode {i_episode}", leave=False)

        for t in count():
            action = select_action(state, select_params)
            select_params.steps_done += 1
            
            raw_next_state, reward, terminated, _, info = env.step(action.item())
            p_bar.update(1)

            # Ensure reward is a scalar by summing if it's an array/list
            if isinstance(reward, (list, tuple, np.ndarray)):
                reward = np.sum(reward)
            episode_reward += float(reward)
            # store a 1-d scalar tensor for reward (shape: [1]) to keep consistent shapes
            preserve_reward = torch.tensor([float(reward)], dtype=torch.float32, device=device)

            cm_index = info["confusion_matrix_index"] # tuple (action, answer)
            # logger.info(cm_index)
            cm_memory.append(list(cm_index)) # list of [action, answer]

            next_state = to_tensor(raw_next_state, include_category) if not terminated else None
            memory.push(state, action, next_state, preserve_reward)
            state = next_state

            if len(memory) > params.get("batch_size", 128):
                loss = optimize_model(optimize_params)
                episode_losses.append(loss)
            
            episode_steps += 1

            if terminated:
                break
        p_bar.close()

        # エピソード終了後の最小限の処理
        cm, accuracy = write_result(cm_memory, i_episode, n_actions)
        
        # epsilon値を計算
        epsilon = select_params.EPS_END + (select_params.EPS_START - select_params.EPS_END) * \
                  np.exp(-1. * select_params.steps_done / select_params.EPS_DECAY)
        
        # lossの平均を計算
        avg_loss = float(np.mean(episode_losses)) if episode_losses else None
        
        # メトリクスをバックグラウンドスレッドに送信（非ブロッキング）
        metrics_logger.log(i_episode, {
            "reward": episode_reward,
            "accuracy": accuracy,
            "steps": episode_steps,
            "buffer_size": len(memory),
            "epsilon": epsilon,
            "loss": avg_loss
        })
        
        # 定期的にログ出力
        # if i_episode % 10 == 0:
        #     logger.info(f"Episode {i_episode}: Reward={episode_reward:.2f}, Accuracy={accuracy:.4f}, Steps={episode_steps}")
        
        # Target networkの更新
        if i_episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
            # logger.info(f"Episode {i_episode}: Target network updated")
        
        # Learning rate schedulerの更新
        if scheduler is not None:
            scheduler.step()

        cm_memory = []

    # メトリクスロガーを終了してメトリクスを保存
    logger.info("Shutting down metrics logger...")
    metrics_history = metrics_logger.shutdown()
    logger.info(f"Logged {len(metrics_history)} episodes")

    # 学習終了後にモデルを保存
    if include_category:
        path = os.path.join("model", "drl_model_with_category.pth")
    else:
        path = os.path.join("model", "drl_model.pth")
    torch.save(policy_net.state_dict(), path)
    mlflow.log_artifact(path, artifact_path="models")
    logger.info("Training completed.")


def main():
    make_dirs()
    params, input_path = load_params()

    with open("log/train_result.log", "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")

    with mlflow.start_run():
        with cProfile.Profile() as pr:
            try:
                print(f"MLflow run name: {mlflow.active_run().info.run_name}")
                df = load_csv(input_path)

                train(df, params)
                mlflow.set_tag("status", "train_completed")
            except Exception as e:
                mlflow.set_tag("status", "train_failed")
                raise e
    
    with open("train_drl.prof", "w") as f:
        ps = pstats.Stats(pr, stream=f)
        ps.sort_stats("cumulative")
        ps.print_stats()
    with open("train_drl.prof", "w") as f:
        ps = pstats.Stats(pr, stream=f)
        ps.sort_stats("time")
        ps.print_stats()


if __name__ == "__main__":
    main()
