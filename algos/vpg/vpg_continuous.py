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
Implementation of the VPG on-policy algorithm.
Parameters are defined in config.json

Vanilla Policy Gradient (VPG) Algorithm:

1. initialize the buffer and logger
2. initialize the actor policy (pi), critic policy (vf), actor optimizer, and critic optimizer
3. for epoch in epochs:
4.      for t in episode:
5.           observe state (obs)
6.           given obs, sample action (act) from policy (pi)
7.           estimate the value of the of the observation (v_t)
8.           perform action (act) and observe next state (obs2), reward (rew), and done (done)
9.           store obs,rew,act,obs2,done,v_t in episode buffer
10.          if done:
11.               compute advantage estimates (advs) using Generalized Advantage Estimation (GAE)
12.               store trajectory (the episode buffer) and advantages (advs) in epoch buffer
13.               reset episode buffer and reset environment
14.          if epoch_done:
15.               get trajectories from epoch buffer
16.               calculate log_likelihood(all_obs,all_acts)
17.               compute actor loss loss_pi = - mean(log_likelihood * R(tau))
18.               compute critic loss loss_v = mean((rewards - value estimates)**2)
19.               perform on step of optimization on both actor and critic networks

Generalized Advantage Estimation (GAE) algorithm:

1. Given episode rewards (ep_rtgs), episode value estimates (ep_vals), and parameters gamma & lambda
2. Compute delta_t = r_t + gamma * v_{t+1} - v_t for all t
3. Compute advantage estimate adv_t = sum_j^T (gamma lambda)^j delta_{t+1} for all t
-  Note: although the trajectory has length T, the last element in the sum has a delta_{T+1} term
         meaning we will need to append an extra term to the trajectory temporarily to compute the estimate

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
        return self.gaussian_policy(x)

    def categorical_policy(self,x):
        # get logits
        logits = self.forward(x)
        # convert logits to probabilities
        probs = tr.exp(logits)/(1+tr.exp(logits))
        # sample from multinomial distribution
        dist = Categorical(probs)
        act = dist.sample().unsqueeze(0)
        return act

    def act(self,obs):
        return self.policy(tensor(obs).float())[0].detach().numpy()



    def log_prob(self,obs,acts):

        pi,mu,std = self.gaussian_policy(obs)
        logp = log_likelihood(acts,mu,std)
        return logp


class Critic(nn.Module):
    def __init__(self,obs_dim,h_dim):
        super(Critic, self).__init__()

        self.layer1 = nn.Linear(obs_dim, h_dim)
        self.layer2 = nn.Linear(h_dim, h_dim)
        self.layer3 = nn.Linear(h_dim, 1)  # Prob of Left

    def forward(self, x):
        x = tr.tanh(self.layer1(x))
        x = tr.tanh(self.layer2(x))
        x = tr.relu(self.layer3(x))
        return x

    def value_estimate(self,obs):
        return self(tensor(obs).float())[0].detach().numpy()

# main train loop
def train(env_name='CartPole-v0',
          pi_lr=1e-2,vf_lr=1e-3, gamma=0.99,
          lam=.95, n_iters=100, steps_per_epoch=5000,h_dim=64):

    env_wrapper = EnvWrapper(env_name)
    env,obs_dim,act_dim = env_wrapper.setup()
    assert isinstance(env.action_space,gym.spaces.Box),\
    'Must provide env with continuous action space'

    pi = Actor(obs_dim,act_dim,h_dim)
    vf = Critic(obs_dim,h_dim)

    optimizer_pi = tr.optim.Adam(pi.parameters(),lr=pi_lr)
    optimizer_v = tr.optim.Adam(vf.parameters(),lr=vf_lr)

    buffer = Buffer()
    logger = Logger()

    def train_one_iteration(epoch):
        # one epoch loop
        buffer.reset()
        logger.reset()

        obs, rew, done, ep_rews, ep_vals = env_wrapper.reset()
        while True:
            # one episode loop

            act = pi.act(obs)
            v_t = vf.value_estimate(obs)

            obs2, rew, done, _ = env.step(act)

            buffer.store_episode(obs,act,rew,v_t)

            obs=obs2

            if done:
                # add episode to batch

                # get episode from buffer
                ep_obs, ep_acts, ep_rews, ep_vals, ep_len = buffer.get_episode() #len(ep_rews)

                # run GAE to get advs
                # outputs estimate for adv
                # if agent died, last_value=reward
                alive = ep_len == env._max_episode_steps
                last_val = vf(tensor(obs).float()).detach().numpy()[0] if alive else rew

                # start advantage compute
                # add last value to compute TD \gamma * V_{t+1} - V_{t}
                ep_rews.append(last_val)
                ep_vals.append(last_val)

                # compute deltas for GAE
                deltas=np.array(ep_rews[:-1]) + gamma * np.array(ep_vals[1:]) - np.array(ep_vals[:-1])

                # go back to how it was
                ep_rews = ep_rews[:-1]
                ep_vals = ep_vals[:-1]

                ep_advs = list(discount_cumsum(deltas, gamma*lam))
                ep_rtgs = list(discount_cumsum(ep_rews, gamma))

                buffer.store_batch(ep_obs, ep_acts,ep_advs,ep_rtgs)
                logger.store(sum(ep_rews),len(ep_rews))

                # reset episode

                buffer.reset_episode()

                obs, rew, done, ep_rews, ep_vals = env_wrapper.reset()
                if len(buffer) > steps_per_epoch:
                    break

        b_o, b_a, b_adv, b_rtg = buffer.get_batch()

        optimizer_pi.zero_grad()
        # get log-likelihoods of state-action pairs
        logp = pi.log_prob(b_o,b_a)
        # normalize advantages
        b_adv -= tr.mean(b_adv)
        b_adv/=tr.std(b_adv)
        # choose loss to maximize likelihood*advantage
        loss_pi = -tr.mean(logp*b_adv)
        loss_pi.backward()
        optimizer_pi.step()

        optimizer_v.zero_grad()

        loss_v = tr.mean((b_rtg-vf(b_o).flatten())**2)
        loss_v.backward()
        optimizer_v.step()
        logger.print_epoch(epoch,loss_pi.detach().numpy(),np.sqrt(loss_v.detach().numpy()))

    for epoch in range(n_iters):
        train_one_iteration(epoch)

if __name__ == '__main__':
    # read in parameters from json file
    with open("./algos/vpg/config.json", "r") as read_file:
        params = json.load(read_file)
    print_hyperparamters(params,is_discrete=False)
    train(**params)
