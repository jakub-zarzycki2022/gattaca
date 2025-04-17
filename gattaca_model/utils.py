import numpy as np
import gym
import torch
import random
from argparse import ArgumentParser
import os
import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')
from scipy.ndimage.filters import gaussian_filter1d


def arguments():
    parser = ArgumentParser()
    parser.add_argument('--env')

    return parser.parse_args()


def save(agent, rewards, args):
    path = './runs/{}/'.format(args.env)
    os.makedirs(path, exist_ok=True)

    torch.save(agent.q.state_dict(), os.path.join(path, 'model_state_dict'))

    plt.cla()
    plt.plot(rewards, c='r', alpha=0.3)
    plt.plot(gaussian_filter1d(rewards, sigma=5), c='r', label='Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative reward')
    plt.title('Branching DDQN: {}'.format(args.env))
    plt.savefig(os.path.join(path, 'reward.png'))

    pd.DataFrame(rewards, columns=['Reward']).to_csv(os.path.join(path, 'rewards.csv'), index=False)


class AgentConfig:

    def __init__(self,
                 epsilon_start=1.,
                 epsilon_final=0.2,
                 epsilon_decay=20000,
                 reward_discount_rate=.95,
                 learning_rate=0.0001,
                 bins=5,
                 target_net_update_freq=7_000,
                 memory_size=10**5,
                 batch_size=256,
                 learning_starts=278,
                 time_steps=10_000_000):
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.epsilon_by_frame = lambda i: \
            self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1. * i / self.epsilon_decay)

        self.gamma = reward_discount_rate
        self.reward_discount_rate = reward_discount_rate
        self.lr = learning_rate
        self.learning_rate = learning_rate

        self.target_net_update_freq = target_net_update_freq
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.learning_starts = max(learning_starts, batch_size)
        self.time_steps = time_steps
        print(self.time_steps)
        self.bins = bins

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print(f"Training on {self.device}")


class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):

        batch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for b in batch:
            states.append(b[0])
            actions.append(b[1])
            rewards.append(b[2])
            next_states.append(b[3])
            dones.append(b[4])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
