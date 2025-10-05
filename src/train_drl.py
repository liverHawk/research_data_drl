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
import matplotlib.pyplot as plt

from itertools import count
from glob import glob
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import flow_package as fp
from flow_package.multi_flow_env import MultiFlowEnv, InputType

from utils import setup_logging, rolling_normalize
from network import DeepFlowNetwork
from deep_learn import ReplayMemory, Transaction

# ========================
# デバイス設定
# ========================
if torch.cuda.is_available():
    device = torch.device("cuda")
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


def load_params():
    if len(sys.argv) != 2:
        print("Usage: python src/train_drl.py <input_file_directory>")
        sys.exit(1)

# ========================
# パラメータ・入力パスのロード
# ========================
    input_path = sys.argv[1]
    all_params = yaml.safe_load(open("params.yaml"))

    mlflow.set_tracking_uri(all_params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(
        f"{all_params['mlflow']['experiment_name']}_train_drl"
    )
    params = all_params["train_drl"]
    return params, input_path

def load_csv(input):
    files = glob(os.path.join(input, "*.csv.gz"))
    df = pd.concat([
        pd.read_csv(f) for f in files
    ])

# ========================
# データのロード・正規化
# ========================
    df = df.reset_index(drop=True)

    # 特徴量の正規化
    df = rolling_normalize(df)

    return df


def _unpack_state_batch(state_batch):
    port = torch.cat([s[0] for s in state_batch])
    protocol = torch.cat([s[1] for s in state_batch])
    features = torch.cat([s[2] for s in state_batch])
    return [port, protocol, features]

# ========================
# バッチ状態のアンパック
# ========================


def select_action(state_tensor: torch.Tensor, params: SelectActionParams):
    """
    ε-greedy法による行動選択
    """
    sample = random.random()
    eps_threshold = params.EPS_END + (params.EPS_START - params.EPS_END) * np.exp(-1. * params.steps_done / params.EPS_DECAY)
    # state_tensorはリスト形式（[port, protocol, other]）
    if sample > eps_threshold:
        with torch.no_grad():
            return params.policy_net(state_tensor).max(1).indices.view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(params.n_actions)]],
            dtype=torch.long,
            device=device
        )


def optimize_model(params: OptimizeModelParams):
    """
    経験リプレイからバッチをサンプリングし、Qネットワークを最適化
    """
    if len(params.memory) < params.BATCH_SIZE:
        return None
    # バッチサンプリング
    transitions = params.memory.sample(params.BATCH_SIZE)
    batch = Transaction(*zip(*transitions))
    # 状態バッチをデバイスへ
    state_batch = _unpack_state_batch(batch.state)
    try:
        state_batch = [s.to(device) for s in state_batch]
    except Exception:
        pass
    # 行動・報酬バッチ
    action_batch = torch.cat(batch.action).to(device).long()
    try:
        reward_batch = torch.cat([r.view(-1) for r in batch.reward]).to(device).float()
    except Exception:
        reward_batch = torch.tensor([float(r) for r in batch.reward], dtype=torch.float32, device=device)
    # 次状態バッチ
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
    non_final_next_states = [s for s in batch.next_state if s is not None]
    if non_final_next_states:
        next_state_batch = _unpack_state_batch(non_final_next_states)
        try:
            next_state_batch = [s.to(device) for s in next_state_batch]
        except Exception:
            pass
    else:
        next_state_batch = None

    # 損失計算・最適化
    if params.scaler is not None:
        with autocast(device_type=device.type):
            state_action_values = params.policy_net(state_batch).gather(1, action_batch)
            next_state_values = torch.zeros(params.BATCH_SIZE, device=device)
            if next_state_batch is not None:
                with torch.no_grad():
                    next_state_values[non_final_mask] = params.target_net(next_state_batch).max(1).values.float()
            expected_state_action_values = reward_batch + params.GAMMA * next_state_values
            loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        params.optimizer.zero_grad()
        params.scaler.scale(loss).backward()
        utils.clip_grad_value_(params.policy_net.parameters(), 1000)
        params.scaler.step(params.optimizer)
        params.scaler.update()
    else:
        state_action_values = params.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(params.BATCH_SIZE, device=device)
        if next_state_batch is not None:
            with torch.no_grad():
                next_state_values[non_final_mask] = params.target_net(next_state_batch).max(1).values
        expected_state_action_values = reward_batch + params.GAMMA * next_state_values
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        params.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(params.policy_net.parameters(), 1000)
        params.optimizer.step()
    return loss.item()


def get_args(params):
    select_params = SelectActionParams(
        EPS_END=params.get("eps_end", 0.05),
        EPS_START=params.get("eps_start", 0.9),
        EPS_DECAY=params.get("eps_decay", 200),
        # policy_net, n_actions, steps_doneは後でセット
    )
    optimize_params = OptimizeModelParams(
        BATCH_SIZE=params.get("batch_size", 128),
        GAMMA=params.get("gamma", 0.999),
        # memory, policy_net, target_net, optimizer, scalerは後でセット
        scaler=GradScaler()
    )
    return select_params, optimize_params


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

    logger.info(f"State space: {n_states}, Action space: {n_actions}")
    policy_net = DeepFlowNetwork(n_states, n_actions).to(device)
    target_net = DeepFlowNetwork(n_states, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    cm_memory = []

    # メトリクス収集用
    metrics = {
        "reward": [],
        "loss": [],
        "accuracy": [],
        "steps": [],
        "last_cm": None
    }

    select_params, optimize_params = get_args(params)
    optimizer = optim.Adam(policy_net.parameters(), lr=params["lr"])
    memory = ReplayMemory(params["memory_size"])

    select_params.policy_net = policy_net
    select_params.n_actions = n_actions

    optimize_params.memory = memory
    optimize_params.policy_net = policy_net
    optimize_params.target_net = target_net
    optimize_params.optimizer = optimizer

    include_category = params.get("include_category", False)

    for i_episode in tqdm(range(params["n_episodes"])):
        random.seed(i_episode)

        initial_state = env.reset()

        if not include_category:
            state = torch.tensor(initial_state, device=device)
        else:
            state = fp.to_tensor(initial_state, device=device)

        episode_reward = 0.0
        episode_losses = []
        episode_steps = 0

        for t in tqdm(count(), leave=False):
            action = select_action(state, select_params)
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

            if not include_category:
                next_state = torch.tensor(raw_next_state, device=device) if not terminated else None
            else:
                next_state = fp.to_tensor(raw_next_state, device=device) if not terminated else None
            memory.push(state, action, next_state, preserve_reward)
            state = next_state

            loss = optimize_model(optimize_params)
            if loss is not None:
                episode_losses.append(loss)
            episode_steps += 1

            if terminated:
                break

        avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        cm, accuracy = write_result(cm_memory, i_episode, n_actions)

        metrics["reward"].append(episode_reward)
        metrics["loss"].append(avg_loss)
        metrics["accuracy"].append(accuracy)
        metrics["steps"].append(episode_steps)
        metrics["last_cm"] = cm.copy()

        # エピソードごとにログ用CSVへ追記
        metrics_df = pd.DataFrame({
            "episode": np.arange(len(metrics["reward"])),
            "reward": metrics["reward"],
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "steps": metrics["steps"]
        })
        metrics_df.to_csv("train/metrics.csv", index=False)

        # mlflow に逐次メトリクスを送信（step=i_episode）
        mlflow.log_metric("reward", float(episode_reward), step=i_episode)
        mlflow.log_metric("loss", float(avg_loss), step=i_episode)
        mlflow.log_metric("accuracy", float(accuracy), step=i_episode)
        mlflow.log_metric("steps", int(episode_steps), step=i_episode)

        cm_memory = []

    # 学習終了後に最終アーティファクトを保存
    # モデル
    path = os.path.join("model", "drl_model.pth")
    torch.save(policy_net.state_dict(), path)
    mlflow.log_artifact(path, artifact_path="models")

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
