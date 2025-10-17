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

from utils import setup_logging, rolling_normalize, torch_device
from network import DeepFlowNetwork
from network_v2 import DeepFlowNetworkV2
from deep_learn import ReplayMemory, Transaction
import gymnasium as gym

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

steps_done = 0


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
device = torch_device()

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


def select_action(state_tensor: torch.Tensor, params: SelectActionParams):
    global steps_done

    sample = random.random()
    buf_exp = -1. * steps_done / params.EPS_DECAY
    eps_threshold = params.EPS_END + (params.EPS_START - params.EPS_END) * np.exp(buf_exp)

    if sample > eps_threshold:
        with torch.no_grad():
            q_vals = params.policy_net(state_tensor)
            # ネットワークが (n_actions,) を返す場合にバッチ次元 (1, n_actions) を追加
            if q_vals.dim() == 1:
                q_vals = q_vals.unsqueeze(0)
            return q_vals.max(1).indices.view(1, 1).to(device)
    else:
        return torch.tensor(
            [[random.randrange(params.n_actions)]],
            device=device,
            dtype=torch.long
        )


def optimize_model(opt_params: OptimizeModelParams):
    transitions = opt_params.memory.sample(opt_params.BATCH_SIZE)
    batch = Transaction(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool
    )
    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]
    )

    if len(non_final_next_states) > 0:
        next_state_batch = non_final_next_states
    else:
        next_state_batch = None

    state_action_values = opt_params.policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(opt_params.BATCH_SIZE, device=device)

    if next_state_batch is not None:
        with torch.no_grad():
            next_q = opt_params.target_net(next_state_batch)
            next_state_values[non_final_mask] = next_q.max(1).values.float()
    expected_state_action_values = (next_state_values * opt_params.GAMMA) + reward_batch
    loss = F_LOSS(state_action_values, expected_state_action_values.unsqueeze(1))

    opt_params.optimizer.zero_grad()
    if opt_params.scaler:
        opt_params.scaler.scale(loss).backward()
        utils.clip_grad_norm_(opt_params.policy_net.parameters(), 1.0)
        opt_params.scaler.step(opt_params.optimizer)
        opt_params.scaler.update()
    else:
        loss.backward()
        utils.clip_grad_norm_(opt_params.policy_net.parameters(), 1.0)
        opt_params.optimizer.step()
    
    return loss.item()


def select_action_vectorized(state_tensor: torch.Tensor, params: SelectActionParams):
    # ε-greedy戦略の閾値計算
    buf_exp = -1. * params.steps_done / params.EPS_DECAY
    eps_threshold = params.EPS_END + (params.EPS_START - params.EPS_END) * np.exp(buf_exp)

    batch_size = state_tensor.size(0)
    
    # 全環境を一気にネットワークに通す
    with torch.no_grad():
        q_values = params.policy_net(state_tensor)  # (batch_size, n_actions)
        # 1次元の場合はバッチ次元を追加
        if q_values.dim() == 1:
            q_values = q_values.unsqueeze(0)
        
        # 貪欲行動（Q値最大）を計算
        greedy_actions = q_values.max(1).indices.view(-1, 1)  # (batch_size, 1)
    
    # ランダム行動を生成
    random_actions = torch.randint(
        0, params.n_actions, 
        (batch_size, 1), 
        device=device, 
        dtype=torch.long
    )
    
    # ε-greedy選択のマスクを作成（各環境で独立に判定）
    exploration_mask = torch.rand(batch_size, device=device) <= eps_threshold
    
    # マスクを使って行動を選択（True=ランダム、False=貪欲）
    actions = torch.where(
        exploration_mask.unsqueeze(1),  # (batch_size, 1)に変形
        random_actions,
        greedy_actions
    )
    
    return actions


def vector_train(drl_options):
    global steps_done

    env_input = drl_options["input_path"]

    envs = gym.vector.SyncVectorEnv([
        lambda: MultiDfEnv(env_input) for _ in range(drl_options.get("num_envs", 4))
    ])
    policy_net = DeepFlowNetworkV2(
        input_shape=envs.observation_space[0].shape[0],
        output_shape=envs.action_space[0].n
    ).to(device)
    target_net = DeepFlowNetworkV2(
        input_shape=envs.observation_space[0].shape[0],
        output_shape=envs.action_space[0].n
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=drl_options.get("lr", 0.001))
    EPS_START = drl_options.get("eps_start", 0.9)
    EPS_END = drl_options.get("eps_end", 0.05)
    EPS_DECAY = drl_options.get("eps_decay", 200)
    n_actions = envs.action_space[0].n

    memory = ReplayMemory(drl_options.get("memory_size", 10000))


    for i_episode in range(100):
        obs, infos = envs.reset()
        obs_tensors = torch.tensor(obs, dtype=torch.float32, device=device)

        for t in count():
            select_action_params = SelectActionParams(
                policy_net=policy_net,
                EPS_START=EPS_START,
                EPS_END=EPS_END,
                EPS_DECAY=EPS_DECAY,
                n_actions=n_actions
            )

            actions = select_action_vectorized(
                obs_tensors,
                select_action_params
            )

            next_obs, rewards, dones, truncs, infos = envs.step(actions.cpu().numpy())
            next_obs_tensors = torch.tensor(
                next_obs, dtype=torch.float32, device=device
            )

            for i in range(envs.num_envs):
                state = obs_tensors[i].unsqueeze(0)
                action = actions[i].unsqueeze(0)
                reward = torch.tensor(
                    [[rewards[i]]], device=device, dtype=torch.float32
                )
                if dones[i] or truncs[i]:
                    next_state = None
                else:
                    next_state = next_obs_tensors[i].unsqueeze(0)

            memory.push(state, action, next_state, reward)
        
            obs_tensors = next_obs_tensors
            # steps_done の更新
            select_action_params.steps_done += 1

            if len(memory) > drl_options.get("batch_size", 128):
                opt_params = OptimizeModelParams(
                    BATCH_SIZE=drl_options.get("batch_size", 128),
                    GAMMA=drl_options.get("gamma", 0.999),
                    memory=memory,
                    policy_net=policy_net,
                    target_net=target_net,
                    optimizer=optimizer,
                    scaler=GradScaler() if torch.cuda.is_available() else None,
                    include_category=drl_options.get("include_category", True)
                )
                loss = optimize_model(opt_params)
            steps_done += 1

