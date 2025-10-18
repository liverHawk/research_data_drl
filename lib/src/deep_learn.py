from collections import namedtuple, deque
import random
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
        for z in zip(batch.states, batch.actions, batch.next_states, batch.rewards):
            self.push(*z)


def moving_average(data, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')