import os
import torch


import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils

from dataclasses import dataclass
from glob import glob
from IPython.display import clear_output
from torch.amp import GradScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from deep_learn import ReplayMemory, Transaction, TransactionBatch
from network_v2 import DeepFlowNetworkV2
from flow_package.multi_df_env_v2 import MultiDfEnvV2, EnvConfig, TestEnvWrapper


@dataclass
class TrainEnvConfig:
    # df_data: pd.DataFrame
    # label_column: str
    reward_list: list
    max_steps: int
    normalize_method: str = 'min-max'
    rolling_window: int = 5
    # test_mode: bool = False

@dataclass
class VectorDRLConfig:
    train_data_path: str
    test_data_path: str
    train_env_config: TrainEnvConfig


def _get_device_name():
    # return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except Exception:
        return "cpu"


def _check_config(config):
    if not os.path.exists(config.train_data_path):
        raise FileNotFoundError(f"Data path {config.train_data_path} does not exist")
    if not os.path.exists(config.test_data_path):
        raise FileNotFoundError(f"Data path {config.test_data_path} does not exist")


def _load_data(data_path):
    files = glob(os.path.join(data_path, "*.csv.gz"))
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs)
    return df


def _make_envs(env_config):
    def _init():
        return MultiDfEnvV2(env_config)
    return _init


def _moving_average(data, window_size):
    """移動平均を計算"""
    if window_size <= 1 or len(data) == 0:
        return np.array(data)
    if len(data) < window_size:
        window_size = max(1, len(data))
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def _enhanced_plot_loss(loss_list, episode=None):
    """改良版のLossプロット関数"""
    clear_output(wait=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 全体のLoss推移
    axes[0].plot(loss_list, alpha=0.7, color='blue', label='Raw')
    if len(loss_list) > 10:
        smooth = _moving_average(loss_list, min(50, len(loss_list)//4))
        axes[0].plot(range(len(smooth)), smooth, color='red', linewidth=2, label='Smooth')
    axes[0].set_title(f'Training Loss (Step {len(loss_list)})')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. 最近のLoss（最後の200ステップ）
    recent = loss_list[-200:] if len(loss_list) > 200 else loss_list
    axes[1].plot(recent, color='green', linewidth=2)
    axes[1].set_title('Recent Loss')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)
    
    # 3. 統計情報
    axes[2].axis('off')
    current_loss = loss_list[-1] if loss_list else 0
    min_loss = min(loss_list) if loss_list else 0
    avg_loss = np.mean(loss_list) if loss_list else 0
    
    stats = f"""
    Current Loss: {current_loss:.4f}
    Min Loss: {min_loss:.4f}
    Avg Loss: {avg_loss:.4f}
    
    Steps: {len(loss_list)}
    """
    axes[2].text(0.1, 0.5, stats, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    plt.show()


def _data_split(df, split_size=10):
    split_dfs = []
    remaining_df = df.copy()

    for i in range(split_size - 1):
        split_df, remaining_df = train_test_split(
            remaining_df,
            train_size=1/(split_size - i),
            random_state=42,
        )
        split_dfs.append(split_df)
    else:
        split_dfs.append(remaining_df)
    return split_dfs


class VectorDRL:
    def __init__(self, config):
        self.device = torch.device(_get_device_name())
        _check_config(config)

        self.train_data = _load_data(config.train_data_path)
        self.test_data = _load_data(config.test_data_path)
        
        self.train_env_config = EnvConfig(
            df_data=self.train_data,
            label_column="Label",
            reward_list=config.train_env_config.reward_list,
            max_steps=config.train_env_config.max_steps,
            normalize_method=config.train_env_config.normalize_method,
            rolling_window=config.train_env_config.rolling_window,
        )
        self.train_envs = gym.vector.SyncVectorEnv([_make_envs(self.train_env_config) for _ in range(7)])
        self.n_states = self.train_envs.observation_space.shape[1]
        self.n_actions = self.train_envs.action_space[0].n

        self.policy_net = DeepFlowNetworkV2(self.n_states, self.n_actions).to(self.device)
        self.target_net = DeepFlowNetworkV2(self.n_states, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayMemory(100000)
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.F_LOSS = nn.SmoothL1Loss()
        self.scaler = GradScaler() if torch.cuda.is_available() else None
    
    def _optimize_model(self):
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transaction(*zip(*transitions))

        state_batch = torch.tensor(batch.state, device=self.device)
        action_batch = torch.cat(batch.action).to(self.device).long().unsqueeze(1)
        reward_batch = torch.cat(batch.reward).to(self.device).float()
        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            device=self.device,
            dtype=torch.bool
        )
        non_final_next_states = [s for s in batch.next_state if s is not None]

        if len(non_final_next_states) > 0:
            next_state_batch = torch.tensor(non_final_next_states, device=self.device)
        else:
            next_state_batch = None

        state_action = self.policy_net(state_batch)
        state_action_values = state_action.gather(1, action_batch)
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)

        if next_state_batch is not None:
            with torch.no_grad():
                next_q = self.target_net(next_state_batch)
                next_state_values[non_final_mask] = next_q.max(1).values.float()
        expected_state_action_values = reward_batch + self.GAMMA * next_state_values

        loss = self.F_LOSS(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            utils.clip_grad_value_(self.policy_net.parameters(), 1000)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            utils.clip_grad_value_(self.policy_net.parameters(), 1000)
            self.optimizer.step()
        
        return loss.item()
    

    def _select_action(self, state_tensor: torch.Tensor):
        random_float_list = np.random.rand(len(state_tensor))

        with torch.no_grad():
            net_action = self.policy_net(state_tensor).argmax(dim=1)
        
        random_action = self.train_envs.action_space.sample()
        random_action_tensor = torch.tensor(random_action, device=self.device)

        mask = torch.tensor(random_float_list > 0.5, device=self.device)

        action_tensor = torch.where(mask, net_action, random_action_tensor)
        return action_tensor.unsqueeze(1)
    
    def _test_prepare(self):
        print("method _test_prepare is called")
        self.test_data = self.test_data.sample(n=1000)
        self.test_data_split_dfs = _data_split(self.test_data, split_size=10)
        vector_envs_input = []
        for df in self.test_data_split_dfs:
            config = EnvConfig(
                df_data=df,
                label_column="Label",
                reward_list=self.train_env_config.reward_list,
                max_steps=len(df),
                normalize_method=self.train_env_config.normalize_method,
                rolling_window=self.train_env_config.rolling_window,
                test_mode=True,
                n_actions=self.n_actions,
            )
            vector_envs_input.append(_make_envs(config))
        test_envs = gym.vector.SyncVectorEnv(vector_envs_input)
        self.test_envs = TestEnvWrapper(test_envs)
        print("method _test_prepare is finished")


    def train(self, n_steps=1000):
        loss_list = []
        obs, infos = self.train_envs.reset()
        obs_tensor = torch.tensor(obs, device=self.device)

        for step in range(n_steps):
            actions = self._select_action(obs_tensor)
            next_obs, rewards, terminated, truncated, infos = self.train_envs.step(actions)

            preserve_rewards = torch.tensor(
                [[float(r)] for r in rewards],
                dtype=torch.float32,
                device=self.device,
            )
            self.memory.push_batch(TransactionBatch(obs, actions, next_obs, preserve_rewards))

            if len(self.memory) > self.BATCH_SIZE:
                loss = self._optimize_model()
                loss_list.append(loss)
                if step % 5 == 0:
                    _enhanced_plot_loss(loss_list)

            obs_tensor = torch.tensor(next_obs, device=self.device)

    def test(self):
        print("method test is called")
        self._test_prepare()
        obs, infos = self.test_envs.reset()
        obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32)

        result_list = []

        print("start testing")
        print(infos["data_length"])

        try:
            while True:
                pred = self.policy_net(obs_tensor)
                actions = pred.argmax(dim=1)
                next_obs, rewards, terminated, truncated, infos = self.test_envs.step(actions)
                obs_tensor = torch.tensor(next_obs, device=self.device, dtype=torch.float32)

                for item in infos["matrix_position"]:
                    result_list.append(item)
                print(f"{len(result_list)} : {infos['steps']}")
                if self.test_envs.finished_envs.all():
                    break
        except Exception as e:
            print()
            raise e
        return result_list
    
    def save_model(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.policy_net.state_dict(), os.path.join(dir_path, "policy_net.pth"))
        torch.save(self.target_net.state_dict(), os.path.join(dir_path, "target_net.pth"))


def main():
    train_env_config = TrainEnvConfig(
        reward_list=[0.5, 1.5, -1.0, -0.5, -1.0],
        max_steps=10_000,
        normalize_method="min-max",
        rolling_window=10,
    )
    vector_drl_config = VectorDRLConfig(
        train_data_path=os.path.join(os.path.dirname(__file__), "../data/train/binary"),
        test_data_path=os.path.join(os.path.dirname(__file__), "../data/test/binary"),
        train_env_config=train_env_config,
    )

    vector_drl = VectorDRL(vector_drl_config)
    print("Training...")
    vector_drl.train(n_steps=20)
    print("Testing...")
    result_list = vector_drl.test()
    with open("result.csv", "w") as f:
        f.write("action,actual\n")
        for result in result_list:
            f.write(f"{result[0]},{result[1]}\n")
    vector_drl.save_model("model")


if __name__ == "__main__":
    main()
