import numpy as np
import torch as tr
import torch.nn as nn
import torch.tensor as tensor
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import json
from core import *

"""
Implementation of the REINFORCE on-policy algorithm.
Parameters are defined in config.json

REINFORCE Algorithm:

1. initialize the buffer, logger, actor policy pi, and optimizer
2. for epoch in epochs:
3.      for t in episode:
4.           observe state (obs)
5.           given obs, sample action (act) from policy (pi)
6.           perform action (act) and observe next state (obs2), reward (rew), and done (done)
7.           store obs,rew,act,obs2,done in episode buffer
8.           if done:
9.                store trajectory (the episode buffer) in epoch buffer
10.               reset episode buffer and reset environment
11.          if epoch_done:
12.               get trajectories from epoch buffer
13.               calculate log_likelihood(all_obs,all_acts)
14.               compute policy loss loss_pi = - mean(log_likelihood * R(tau))
15.               perform on step of optimization on policy network pi

Classes:

* Actor - network that estimates the mean action mu.
        - methods:
                  - forward: passes observations through MLP and outputs logits
                  - gaussian_policy: the policy for selecting a continuous action given the mean value mu
                  - caregorical_policy: the policy for selecting a discrete action from N possible actions
                  - policy: implements gaussian or categorical policy depending on the action space
                  - log_prob: the log_likelihood of actions (acts) given observations (obs)
        - Note: works with both continuous and discrete action spaces

Functions:

* train - runs the main training loop
"""

class Actor(nn.Module):
    def __init__(self,obs_dim,act_dim,h_dim):
        super(Actor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.layer1 = nn.Linear(obs_dim, h_dim)
        self.layer2 = nn.Linear(h_dim, h_dim)
        self.layer3 = nn.Linear(h_dim, act_dim)  # Prob of Left

    def forward(self, x):
        x = tr.tanh(self.layer1(x))
        x = tr.tanh(self.layer2(x))
        x = self.layer3(x)
        return x

    def policy(self,x):
        return self.categorical_policy(x)

    def categorical_policy(self,x):
        # get logits
        logits = self.forward(x)
        # convert logits to probabilities
        probs = tr.exp(logits)/(1+tr.exp(logits))
        # sample from multinomial distribution
        dist = Categorical(probs)
        act = dist.sample().unsqueeze(0)
        return act

    def log_prob(self,obs,acts):
        one_hot_acts = (acts.reshape(-1,1).float() == tr.arange(self.act_dim).float()).float()
        logp = tr.sum(one_hot_acts*F.log_softmax(self.forward(obs),dim=1),1)
        return logp

# main train loop
def train(env_name='CartPole-v0',
          pi_lr=1e-2, gamma=.99, epochs=100,
          steps_per_epoch=500,h_dim=256):

    env_wrapper = EnvWrapper(env_name)
    env,obs_dim,act_dim = env_wrapper.setup()
    assert isinstance(env.action_space,gym.spaces.Discrete),\
    'Must provide env with discrete action space'


    pi = Actor(obs_dim,act_dim,h_dim)

    optimizer_pi = tr.optim.Adam(pi.parameters(),lr=pi_lr)

    buffer = Buffer()

    def train_one_iteration(epoch):
        tot_eps = 0
        # one epoch loop
        buffer.reset()
        tot_r = []
        obs, rew, done, ep_rews = env_wrapper.reset()
        while True:
            # one episode loop
            act = pi.policy(tensor(obs).float())[0].detach().numpy()
            obs2, rew, done, _ = env.step(act)
            buffer.store_episode(obs,act,rew)
            obs=obs2

            if done:
                # add episode to batch

                # get episode from buffer
                ep_obs, ep_acts, ep_rews, ep_len = buffer.get_episode()
                ep_rtgs = list(discount_cumsum(ep_rews, gamma))

                buffer.store_batch(ep_obs, ep_acts,ep_rtgs)
                tot_r.append(sum(ep_rews))
                tot_eps+=1

                # reset episode

                buffer.reset_episode()

                obs, rew, done, ep_rews = env_wrapper.reset()
                if len(buffer) > steps_per_epoch:
                    break

        b_o, b_a, b_rtg = buffer.get_batch()
        optimizer_pi.zero_grad()
        # get log-likelihoods of state-action pairs
        logp = pi.log_prob(b_o,b_a)
        # normalize rewards
        b_rtg -= tr.mean(b_rtg)
        b_rtg/= tr.std(b_rtg)
        # choose loss to maximize likelihood*discounted rewards to go
        loss_pi = -tr.mean(logp*b_rtg)
        loss_pi.backward()
        optimizer_pi.step()
        print('Epoch',epoch,'Reward',int(np.mean(tot_r)),'Pi Loss',round(loss_pi.item(),3))
        tot_r = []


    for epoch in range(epochs):
        train_one_iteration(epoch)

if __name__ == '__main__':
    # read in parameters from json file
    with open("./algos/reinforce/config.json", "r") as read_file:
        params = json.load(read_file)
    print_hyperparamters(params,is_discrete=True)
    train(**params)
