import numpy as np
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
import json
import gym
from core import *
"""
PPO algorithm is similar to a VPG but with the following modifications:
1. We use a surrogate loss pi(theta)/pi(theta_k)*adv instead of the likelihood loss
2. We clip the surrogate loss to limit large policy updates

Helper functions:

From core, we're using the following helper functions:
    1. discount cumsum (x,gamma) - computes cumulative sum = sum_t gamma^t x_t
    2. normalize (x) - normalizes x tensor to mean 0 and std 1
    3. using as_tensor (x) - casts scalar, list, or array as a tensor
    4. using as_tensors (*args) - applies as_tensor to multiple args
    5. buffer - generic buffer class for storing trajectories

Gradients: the only variables that hold gradients should be
    1. log probs for the current policy (not the old one)
    2. the current values estimate during update (not the old ones)

Advantage estimation: this code can estimate advantages in two ways
    1. Generalized Advantage Estimation (preferred)
    2. Simple estimation using R-V as the advantage

"""

def train(env_name = "CartPole-v1",
        render = False,
        log_interval = 10,
        h_dim = 64,
        eps_per_update = 10,
        pi_lr = 0.0007,
        v_lr = 0.0007,
        gamma = 0.99,
        lam=0.95,
        n_updates = 10,
        eps_clip = 0.2,
        random_seed = 0,
        total_episodes=1000,
        use_gae=False
        ):

    env = gym.make(env_name)
    assert isinstance(env.action_space,gym.spaces.Discrete),\
    'Must provide env with discrete action space'
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    max_steps = env._max_episode_steps

    if random_seed:
        tr.manual_seed(random_seed)
        env.seed(random_seed)

    class Actor(nn.Module):

        def __init__(self,obs_dim,act_dim,h_dim):
            super(Actor,self).__init__()
            self.probs = nn.Sequential(
                    nn.Linear(obs_dim, h_dim),
                    nn.Tanh(),
                    nn.Linear(h_dim, h_dim),
                    nn.Tanh(),
                    nn.Linear(h_dim, act_dim),
                    nn.Softmax()
                    )
        def forward(self,s):
            return self.probs(s)

        def log_prob(self,s,a):
            probs = self.forward(s.float())
            dist = Categorical(probs)
            logp = dist.log_prob(a.float())
            return logp

        def get_action_ops(self,s):
            s = as_tensor(s).float()
            a_prob = self.forward(s)
            dist = Categorical(a_prob)
            a = dist.sample()
            logp = dist.log_prob(a).item()
            v = value_estimate(s).item()
            a = a.item()
            return a,logp,v

    value_estimate = nn.Sequential(
            nn.Linear(obs_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, 1)
            )
    policy = Actor(obs_dim,act_dim,h_dim)
    policy_old = Actor(obs_dim,act_dim,h_dim)
    optimizer_pi = tr.optim.Adam(policy.parameters(),lr=pi_lr)
    optimizer_v = tr.optim.Adam(value_estimate.parameters(),lr=v_lr)

    tot_r = []
    # parameters we'll be storing in memory
    buffer_params = ['s','a','a','logp','v','adv']
    memory = Buffer()

    def GAE(ep_rews,ep_vals,s):
        """
        Generalized Advantage Estimate
        """
        n = len(ep_rews)
        alive = n == max_steps # if agent died, last_value=reward
        last_val = value_estimate(as_tensor(s).float()).detach().numpy()[0] if alive else ep_rews[-1]
        # add last values to compute TD \gamma * V_{t+1} - V_{t} - need final T+1 value
        v_t_plus_1 = np.array(ep_vals[1:] + [last_val])
        v_t = np.array(ep_vals)
        # compute deltas for GAE
        deltas=np.array(ep_rews) + gamma * v_t_plus_1 - v_t
        advs = discount_cumsum(deltas, gamma*lam)
        return advs

    def update():
        """
        Optimization step using PPO update rule
        """
        d = memory.retrieve('s','a','r','logp','adv')
        s,a,r,logp_old,adv = d['s'],d['a'],d['r'],d['logp'],d['adv']
        r = discount_cumsum(r,gamma)
        s,a,logp_old,r = as_tensors(s,a,logp_old,r)
        r = normalize(r)
        losses = {'pi':[],'v':[]}
        for _ in range(n_updates):
            logp = policy.log_prob(s,a)
            # logp has parameters theta
            # logp_old has not gradient parameters, it's a normalization factor
            ratio = tr.exp(logp - logp_old)
            vals = value_estimate(s.float()).flatten()
            # either uses GAE or (R-V) as the advantage
            adv = as_tensor(adv) if use_gae else r - vals.detach()
            adv = normalize(adv)

            # PPO surrogate loss
            surr1 = ratio * adv
            surr2 = tr.clamp(ratio, 1-eps_clip, 1+eps_clip)
            loss = -tr.min(surr1, surr2).mean()
            #loss = -tr.mean(logp*adv)
            optimizer_pi.zero_grad()
            loss.backward()
            optimizer_pi.step()

            loss_v =  tr.mean(0.5*(vals-r)**2)
            optimizer_v.zero_grad()
            loss_v.backward()
            optimizer_v.step()

            losses['pi'].append(loss.item())
            losses['v'].append(loss_v.item())

        # copy new weights to the old policy
        policy_old.load_state_dict(policy.state_dict())
        return np.mean(losses['pi']),np.mean(losses['v'])

    step = 0
    for i_episode in range(1, total_episodes):
        step+=1
        s = env.reset()
        ep_rews, ep_vals = [],[]
        for t in range(max_steps):
            # run old policy
            a,logp,v = policy_old.get_action_ops(s)
            s_, r, d, _ = env.step(a)
            # save s,r,a,logp,v into buffer
            memory.store(s=s,a=a,r=r,logp=logp,v=v)
            s = s_
            ep_rews.append(r)
            ep_vals.append(v)
            if render:
                env.render()
            if d:
                adv = GAE(ep_rews,ep_vals,s)
                memory.store(batch=True,adv=adv)
                tot_r.append(sum(ep_rews))
                break

        # updates at the end of every episode
        loss_pi,loss_v= update()
        memory.reset()
        if i_episode % log_interval == 0:
            print('Episode',i_episode,'Reward',int(np.mean(tot_r)),'Pi Loss',round(loss_pi,3),'V Loss',round(loss_v,3))
            tot_r = []

if __name__ == '__main__':
    # read in parameters from json file
    with open("./algos/ppo/config.json", "r") as read_file:
        params = json.load(read_file)
    print_hyperparamters(params,is_discrete=True)
    train(**params)
