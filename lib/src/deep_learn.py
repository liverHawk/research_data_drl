from collections import namedtuple, deque
import random
import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class TransactionBatch:
    states: list
    actions: list
    next_states: list
    rewards: list


Transaction = namedtuple('Transaction', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transaction(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def push_batch(self, batch: TransactionBatch):
        # actionsを個別に分解して格納
        for state, action, next_state, reward in zip(batch.states, batch.actions, batch.next_states, batch.rewards):
            # actionが2次元tensorの場合、1次元に変換してから格納
            if isinstance(action, torch.Tensor) and action.dim() == 2:
                action = action.squeeze(1)  # (batch_size, 1) -> (batch_size,)
            self.push(state, action, next_state, reward)


def moving_average(data, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')