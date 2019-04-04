import math
import random
import gym
import numpy as np
import torch as tr
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import json
# utility functions
from core import soft_update,hard_update,print_hyperparamters
# OUNoise for Gaussian policy and NormalizedActions for env
from core import OUNoise,NormalizedActions
# replay buffer and networks
from core import ReplayBuffer,PolicyNetwork,ValueNetwork


def train(env_name="Pendulum-v0",
          render=0,
          epochs=50,
          steps_per_epoch=1000,
          max_steps=500,
          batch_size=64,
          capacity=10000,
          h_dim=256,
          steps_per_update=20,
          pi_lr=1e-3,
          q_lr=1e-4,
          gamma=0.99,
          tau=1e-2
          ):


    def ddpg_update():

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state      = tr.FloatTensor(state)
        next_state = tr.FloatTensor(next_state)
        action     = tr.FloatTensor(action)
        reward     = tr.FloatTensor(reward).unsqueeze(1)
        done       = tr.FloatTensor(np.float32(done)).unsqueeze(1)

        pi_loss = value_net(state, policy_net(state))
        pi_loss = -pi_loss.mean()

        next_action    = target_policy_net(next_state)
        target_value   = target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * gamma * target_value

        value = value_net(state, action)
        q_loss = value_criterion(value, expected_value.detach())

        policy_optimizer.zero_grad()
        pi_loss.backward()
        policy_optimizer.step()

        value_optimizer.zero_grad()
        q_loss.backward()
        value_optimizer.step()

        soft_update(value_net,target_value_net,tau)
        soft_update(policy_net,target_policy_net,tau)

        return pi_loss.item(),q_loss.item()

    env = NormalizedActions(gym.make(env_name))
    ou_noise = OUNoise(env.action_space)

    obs_dim  = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    value_net  = ValueNetwork(obs_dim, act_dim, h_dim)
    policy_net = PolicyNetwork(obs_dim, act_dim, h_dim)

    target_value_net  = ValueNetwork(obs_dim, act_dim, h_dim)
    target_policy_net = PolicyNetwork(obs_dim, act_dim, h_dim)

    hard_update(value_net,target_value_net)
    hard_update(policy_net,target_policy_net)

    value_optimizer  = optim.Adam(value_net.parameters(),  lr=q_lr)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=pi_lr)

    value_criterion = nn.MSELoss()

    replay_buffer = ReplayBuffer(capacity)

    steps   = 0
    rewards     = []
    pi_loss_history = []
    q_loss_history = []
    while steps < epochs*steps_per_epoch:
        state = env.reset()
        ou_noise.reset()
        episode_reward = 0


        for step in range(max_steps):
            if render:
                env.render()
            action = policy_net.get_action(state)
            action = ou_noise.get_action(action, step)
            next_state, reward, done, _ = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) > batch_size:
                pi_loss,q_loss = ddpg_update()
                pi_loss_history.append(pi_loss)
                q_loss_history.append(q_loss)

            state = next_state
            episode_reward += reward
            steps += 1

            if steps % max(1,steps_per_epoch) == 0:
                i_epoch = steps // steps_per_epoch
                mean_rewards = int(np.mean(rewards[-10:]))
                mean_pi_loss = round(np.mean(pi_loss_history[-10:]),3)
                mean_q_loss = round(np.mean(q_loss_history[-10:]),3)
                print('Epoch',i_epoch, 'Rewards' ,mean_rewards,'Pi Loss',mean_pi_loss,'Q Loss',mean_q_loss)
            if done:
                break

        rewards.append(episode_reward)

if __name__ == '__main__':
    # read in parameters from json file
    with open("./algos/ddpg/config.json", "r") as read_file:
        params = json.load(read_file)
    print_hyperparamters(params)
    train(**params)
