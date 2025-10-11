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


import os
import sys
import yaml
import random
import pandas as pd
import numpy as np
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
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import matplotlib.pyplot as plt
import psutil
from collections import deque
from scipy.stats import entropy

from itertools import count
from glob import glob
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import flow_package as fp
from flow_package.multi_flow_env import MultiFlowEnv, InputType

from utils import setup_logging, rolling_normalize
from network import DeepFlowNetwork, DeepFlowNetworkV2
from deep_learn import ReplayMemory, Transaction


def setup_mlflow(all_params):
    if all_params["mlflow"]["use_azure"]:
        import dagshub
        dagshub.init(repo_owner='liverHawk', repo_name='research_data_drl', mlflow=True)
        path = os.path.join(os.path.dirname(__file__), "..", "config.json")
        print(path)
        ml_client = MLClient.from_config(
            credential=DefaultAzureCredential(),
            config_path=path
        )
        mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    else:
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
# リソース監視機能
# ========================
def get_gpu_memory_usage():
    """GPUメモリ使用率を取得（使用可能な場合）"""
    if torch.cuda.is_available():
        return {
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "gpu_memory_allocated_pct": (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
        }
    elif torch.mps.is_available():
        # MPSの場合はメモリ情報の取得が制限されている
        return {
            "gpu_memory_allocated_mb": torch.mps.current_allocated_memory() / 1024**2 if hasattr(torch.mps, 'current_allocated_memory') else 0,
            "gpu_memory_reserved_mb": 0,
            "gpu_memory_allocated_pct": 0
        }
    return {"gpu_memory_allocated_mb": 0, "gpu_memory_reserved_mb": 0, "gpu_memory_allocated_pct": 0}


def get_system_resources():
    """システムリソース（RAM、CPU）の使用状況を取得"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "ram_usage_mb": memory_info.rss / 1024**2,
        "ram_usage_pct": process.memory_percent(),
        "cpu_usage_pct": process.cpu_percent(interval=0.1)
    }


# ========================
# デバイス設定
# ========================
if torch.cuda.is_available():
    device = torch.device("cuda:3")
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
    params = all_params["train_drl"]
    return params, input_path


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
    -> バッチごとに (batch_size, n_states) の 2D Tensor を返す（float32, device に移動）
    """
    # state_batch が空の可能性は呼び出し元で弾く想定
    if include_category:
        # 各サンプルごとに要素を結合して1次元ベクトルにし、それを stack して (B, n_states) にする
        per_samples = []
        for s in state_batch:
            # s[0], s[1], s[2] がテンソルである前提
            a = s[0].view(-1)
            b = s[1].view(-1)
            c = s[2].view(-1)
            per_samples.append(torch.cat([a, b, c], dim=0))
        return torch.stack(per_samples, dim=0).to(device).float()
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

    # メトリクス計算用の辞書
    metrics = {}

    def _calculate_loss():
        # policy_net は (B, n_states) を受け取り (B, n_actions) を返す想定
        state_action_values = params.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(params.BATCH_SIZE, device=device)
        if next_state_batch is not None:
            with torch.no_grad():
                next_q = params.target_net(next_state_batch)
                next_state_values[non_final_mask] = next_q.max(1).values.float()
        expected_state_action_values = reward_batch + params.GAMMA * next_state_values
        loss = F_LOSS(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # メトリクスの保存
        with torch.no_grad():
            metrics["value_pred"] = state_action_values.mean().item()
            metrics["value_target"] = expected_state_action_values.mean().item()
            metrics["td_error"] = (state_action_values.squeeze() - expected_state_action_values).abs().mean().item()
            metrics["q_value_mean"] = state_action_values.mean().item()
            metrics["q_value_std"] = state_action_values.std().item()
            metrics["reward_mean"] = reward_batch.mean().item()
        
        return loss

    # 損失計算・最適化
    if params.scaler is not None:
        with autocast(device_type=device.type):
            loss = _calculate_loss()
        params.optimizer.zero_grad()
        params.scaler.scale(loss).backward()
        
        # 勾配ノルムの計算（スケール前）
        total_norm = 0.0
        for p in params.policy_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        metrics["grad_norm"] = total_norm ** 0.5
        
        utils.clip_grad_value_(params.policy_net.parameters(), 1000)
        params.scaler.step(params.optimizer)
        params.scaler.update()
    else:
        loss = _calculate_loss()
        params.optimizer.zero_grad()
        loss.backward()
        
        # 勾配ノルムの計算
        total_norm = 0.0
        for p in params.policy_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        metrics["grad_norm"] = total_norm ** 0.5
        
        utils.clip_grad_value_(params.policy_net.parameters(), 1000)
        params.optimizer.step()
    
    metrics["loss"] = loss.item()
    return metrics


def to_tensor(state, include_category=True):
    global device
    if include_category:
        return fp.to_tensor(state, device=device)
    else:
        return torch.tensor(state, device=device, dtype=torch.float32)


def write_result(cm_memory, episode, n_output):
    # 混同行列の計算
    cm = np.zeros((n_output, n_output), dtype=int)
    for action, answer in cm_memory:
        cm[action][answer] += 1
    with open("log/train_result.log", "a") as f:
        f.write(f"Episode {episode}:\n")
        f.write(f"{cm}\n")
    total = cm.sum()
    accuracy = float(cm.diagonal().sum() / total) if total > 0 else 0.0
    return cm, accuracy


def train(df, params):
    logger = setup_logging("log/train_drl.log")

    # ログパラメータを保存
    mlflow.log_params({
        "n_episodes": params.get("n_episodes"),
        "batch_size": params.get("batch_size"),
        "lr": params.get("lr"),
        "memory_size": params.get("memory_size")
    })

    logger.info("Starting training...")
    label_count = len(df["Label"].unique())
    reward_matrix = np.ones((label_count, label_count)) * -1.0
    np.fill_diagonal(reward_matrix, 1.0)

    columns = df.columns.tolist()
    columns.remove("Label")

    input = InputType(
        data=df,
        sample_size=params.get("sample_size", 100000),
        normalize_exclude_columns=["Protocol", "Destination Port"],
        # exclude_columns=["Attempted Category"],
        reward_list=reward_matrix
    )
    env = MultiFlowEnv(input)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    include_category = params.get("include_category", True)

    logger.info(f"State space: {n_states}, Action space: {n_actions}")

    if include_category:
        policy_net = DeepFlowNetwork(n_states, n_actions).to(device)
        target_net = DeepFlowNetwork(n_states, n_actions).to(device)
    else:
        policy_net = DeepFlowNetworkV2(n_states, n_actions).to(device)
        target_net = DeepFlowNetworkV2(n_states, n_actions).to(device)

    target_net.load_state_dict(policy_net.state_dict())

    cm_memory = []

    # メトリクス収集用（拡張版）
    metrics = {
        # Reward関連
        "reward": [],
        "reward_moving_avg": [],
        
        # Loss関連
        "loss": [],
        "td_error": [],
        "grad_norm": [],
        "value_pred": [],
        "value_target": [],
        "q_value_mean": [],
        "q_value_std": [],
        
        # Action関連
        "epsilon": [],
        "action_entropy": [],
        
        # Environment
        "steps": [],
        "accuracy": [],
        
        # Learning
        "learning_rate": [],
        
        # Off-policy
        "buffer_size": [],
        "buffer_usage_rate": [],
        
        # Resources
        "ram_usage_mb": [],
        "gpu_memory_mb": [],
        "cpu_usage_pct": [],
        
        # その他
        "last_cm": None
    }
    
    # Moving average用のdeque
    reward_window = deque(maxlen=100)

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

        initial_state = env.reset()

        state = to_tensor(initial_state, include_category)

        episode_reward = 0.0
        episode_metrics = {
            "losses": [],
            "td_errors": [],
            "grad_norms": [],
            "value_preds": [],
            "value_targets": [],
            "q_value_means": [],
            "q_value_stds": [],
            "actions": [],
            "epsilons": []
        }
        episode_steps = 0

        for t in tqdm(count(), leave=False):
            # ε値の計算と記録
            eps_threshold = select_params.EPS_END + (select_params.EPS_START - select_params.EPS_END) * \
                           np.exp(-1. * select_params.steps_done / select_params.EPS_DECAY)
            episode_metrics["epsilons"].append(eps_threshold)
            
            action = select_action(state, select_params)
            select_params.steps_done += 1
            
            # 行動を記録
            episode_metrics["actions"].append(action.item())
            
            raw_next_state, reward, terminated, _, info = env.step(action.item())

            # Ensure reward is a scalar by summing if it's an array/list
            if isinstance(reward, (list, tuple, np.ndarray)):
                reward = np.sum(reward)
            episode_reward += float(reward)
            # store a 1-d scalar tensor for reward (shape: [1]) to keep consistent shapes
            preserve_reward = torch.tensor([float(reward)], dtype=torch.float32, device=device)

            cm_memory.append([
                info["action"],
                info["answer"],
            ])

            next_state = to_tensor(raw_next_state, include_category) if not terminated else None
            memory.push(state, action, next_state, preserve_reward)
            state = next_state

            if len(memory) > params.get("batch_size", 128):
                opt_metrics = optimize_model(optimize_params)
                episode_metrics["losses"].append(opt_metrics["loss"])
                episode_metrics["td_errors"].append(opt_metrics["td_error"])
                episode_metrics["grad_norms"].append(opt_metrics["grad_norm"])
                episode_metrics["value_preds"].append(opt_metrics["value_pred"])
                episode_metrics["value_targets"].append(opt_metrics["value_target"])
                episode_metrics["q_value_means"].append(opt_metrics["q_value_mean"])
                episode_metrics["q_value_stds"].append(opt_metrics["q_value_std"])
            
            episode_steps += 1

            if terminated:
                break

        # エピソード終了後のメトリクス集計
        avg_loss = float(np.mean(episode_metrics["losses"])) if episode_metrics["losses"] else 0.0
        avg_td_error = float(np.mean(episode_metrics["td_errors"])) if episode_metrics["td_errors"] else 0.0
        avg_grad_norm = float(np.mean(episode_metrics["grad_norms"])) if episode_metrics["grad_norms"] else 0.0
        avg_value_pred = float(np.mean(episode_metrics["value_preds"])) if episode_metrics["value_preds"] else 0.0
        avg_value_target = float(np.mean(episode_metrics["value_targets"])) if episode_metrics["value_targets"] else 0.0
        avg_q_value_mean = float(np.mean(episode_metrics["q_value_means"])) if episode_metrics["q_value_means"] else 0.0
        avg_q_value_std = float(np.mean(episode_metrics["q_value_stds"])) if episode_metrics["q_value_stds"] else 0.0
        avg_epsilon = float(np.mean(episode_metrics["epsilons"])) if episode_metrics["epsilons"] else 0.0
        
        # Action entropyの計算
        if episode_metrics["actions"]:
            action_counts = np.bincount(episode_metrics["actions"], minlength=n_actions)
            action_probs = action_counts / action_counts.sum()
            action_entropy_val = entropy(action_probs + 1e-10)
        else:
            action_entropy_val = 0.0
        
        # Moving average reward
        reward_window.append(episode_reward)
        reward_moving_avg = float(np.mean(reward_window))
        
        # リソース使用状況の取得
        sys_resources = get_system_resources()
        gpu_resources = get_gpu_memory_usage()
        
        # Buffer使用率
        buffer_size = len(memory)
        buffer_usage_rate = buffer_size / params["memory_size"]
        
        # 現在の学習率
        current_lr = optimizer.param_groups[0]['lr']
        
        cm, accuracy = write_result(cm_memory, i_episode, n_actions)

        # メトリクスの保存
        metrics["reward"].append(episode_reward)
        metrics["reward_moving_avg"].append(reward_moving_avg)
        metrics["loss"].append(avg_loss)
        metrics["td_error"].append(avg_td_error)
        metrics["grad_norm"].append(avg_grad_norm)
        metrics["value_pred"].append(avg_value_pred)
        metrics["value_target"].append(avg_value_target)
        metrics["q_value_mean"].append(avg_q_value_mean)
        metrics["q_value_std"].append(avg_q_value_std)
        metrics["epsilon"].append(avg_epsilon)
        metrics["action_entropy"].append(action_entropy_val)
        metrics["steps"].append(episode_steps)
        metrics["accuracy"].append(accuracy)
        metrics["learning_rate"].append(current_lr)
        metrics["buffer_size"].append(buffer_size)
        metrics["buffer_usage_rate"].append(buffer_usage_rate)
        metrics["ram_usage_mb"].append(sys_resources["ram_usage_mb"])
        metrics["gpu_memory_mb"].append(gpu_resources["gpu_memory_allocated_mb"])
        metrics["cpu_usage_pct"].append(sys_resources["cpu_usage_pct"])
        metrics["last_cm"] = cm.copy()

        # エピソードごとにログ用CSVへ追記
        metrics_df = pd.DataFrame({
            "episode": np.arange(len(metrics["reward"])),
            "reward": metrics["reward"],
            "reward_moving_avg": metrics["reward_moving_avg"],
            "loss": metrics["loss"],
            "td_error": metrics["td_error"],
            "grad_norm": metrics["grad_norm"],
            "value_pred": metrics["value_pred"],
            "value_target": metrics["value_target"],
            "q_value_mean": metrics["q_value_mean"],
            "q_value_std": metrics["q_value_std"],
            "epsilon": metrics["epsilon"],
            "action_entropy": metrics["action_entropy"],
            "accuracy": metrics["accuracy"],
            "steps": metrics["steps"],
            "learning_rate": metrics["learning_rate"],
            "buffer_size": metrics["buffer_size"],
            "buffer_usage_rate": metrics["buffer_usage_rate"],
            "ram_usage_mb": metrics["ram_usage_mb"],
            "gpu_memory_mb": metrics["gpu_memory_mb"],
            "cpu_usage_pct": metrics["cpu_usage_pct"]
        })
        metrics_df.to_csv("train/metrics.csv", index=False)

        # mlflow に逐次メトリクスを送信（step=i_episode）
        mlflow.log_metric("reward/total", float(episode_reward), step=i_episode)
        mlflow.log_metric("reward/moving_avg", float(reward_moving_avg), step=i_episode)
        mlflow.log_metric("loss/q_loss", float(avg_loss), step=i_episode)
        mlflow.log_metric("loss/td_error", float(avg_td_error), step=i_episode)
        mlflow.log_metric("loss/grad_norm", float(avg_grad_norm), step=i_episode)
        mlflow.log_metric("loss/value_pred", float(avg_value_pred), step=i_episode)
        mlflow.log_metric("loss/value_target", float(avg_value_target), step=i_episode)
        mlflow.log_metric("loss/q_value_mean", float(avg_q_value_mean), step=i_episode)
        mlflow.log_metric("loss/q_value_std", float(avg_q_value_std), step=i_episode)
        mlflow.log_metric("action/epsilon", float(avg_epsilon), step=i_episode)
        mlflow.log_metric("action/entropy", float(action_entropy_val), step=i_episode)
        mlflow.log_metric("env/accuracy", float(accuracy), step=i_episode)
        mlflow.log_metric("env/episode_length", int(episode_steps), step=i_episode)
        mlflow.log_metric("learning/lr", float(current_lr), step=i_episode)
        mlflow.log_metric("buffer/size", int(buffer_size), step=i_episode)
        mlflow.log_metric("buffer/usage_rate", float(buffer_usage_rate), step=i_episode)
        mlflow.log_metric("resource/ram_mb", float(sys_resources["ram_usage_mb"]), step=i_episode)
        mlflow.log_metric("resource/gpu_memory_mb", float(gpu_resources["gpu_memory_allocated_mb"]), step=i_episode)
        mlflow.log_metric("resource/cpu_pct", float(sys_resources["cpu_usage_pct"]), step=i_episode)
        
        # Target networkの更新
        if i_episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
            logger.info(f"Episode {i_episode}: Target network updated")
        
        # Learning rate schedulerの更新
        if scheduler is not None:
            scheduler.step()

        cm_memory = []

    # 学習終了後に最終アーティファクトを保存
    # モデル
    path = os.path.join("model", "drl_model.pth")
    torch.save(policy_net.state_dict(), path)
    mlflow.log_artifact(path, artifact_path="models")

    # TODO: plot moving average of reward
    

    # 全メトリクスCSV と logs をまとめて保存
    mlflow.log_artifact("train/metrics.csv")
    mlflow.log_artifacts("train/plots")


def main():
    make_dirs()
    params, input_path = load_params()

    with open("log/train_result.log", "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")

    mlflow.start_run()
    df = load_csv(input_path)

    train(df, params)
    mlflow.end_run()


if __name__ == "__main__":
    main()
