import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from collections import deque
from core import ReplayBuffer,print_hyperparamters
import json

class DQN(nn.Module):
    def __init__(self,obs_dim, act_dim,h_dim,
                 eps_start,eps_end,eps_decay
                 ):
        super(DQN, self).__init__()

        self.t = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.act_dim = act_dim

        self.layers = nn.Sequential(
            nn.Linear(obs_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state):
        self.t+=1
        eps_start,eps_end,eps_decay = self.eps_start,self.eps_end,self.eps_decay
        epsilon = eps_end + (eps_start - eps_end) * math.exp(-1. * self.t / eps_decay)
        if random.random() > epsilon:
            state   = torch.FloatTensor(state).unsqueeze(0)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.act_dim)
        return action

def train(env_name="CartPole-v0",
          render=0,
          epochs=50,
          steps_per_epoch=1000,
          batch_size=64,
          capacity=10000,
          h_dim=64,
          steps_per_update=20,
          lr=1e-2,
          gamma=0.99,
          eps_start=1.0,
          eps_end=0.01,
          eps_decay=500
          ):

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = DQN(obs_dim, act_dim,h_dim,eps_start,eps_end,eps_decay)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    replay_buffer = ReplayBuffer(1000)

    def dqn_update(batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        q_values      = model(state)
        next_q_values = model(next_state)
        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + gamma * next_q_value * (1 - done)
        loss = (q_value - expected_q_value.data).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    losses = []
    all_rewards = []
    episode_reward = 0

    state = env.reset()
    for steps in range(epochs*steps_per_epoch):
        action = model.act(state)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > batch_size:
            loss = dqn_update(batch_size)
            losses.append(loss.item())
        if steps % steps_per_epoch == 0 and steps>0:
            i_epoch = steps // steps_per_epoch
            print('Epoch:',i_epoch,'Reward:', np.mean(all_rewards[-10:]),'Loss',round(np.mean(losses[-10:]),2))

if __name__ == '__main__':
    # read in parameters from json file
    with open("./algos/dqn/config.json", "r") as read_file:
        params = json.load(read_file)
    print_hyperparamters(params)
    train(**params)
