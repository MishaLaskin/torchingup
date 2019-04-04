import numpy as np
import torch as tr
import gym
from gym.spaces import Box,Discrete
EPS = 1e-8

"""
Utilities for running the VPG

Classes:

* Buffer - collects episode trajectories for training
* Logger - collects relevant info for printing
* EnvWrapper - convenience class for collecting info from a gym env

Functions:

* log_likelihood - the log likelihood of an action conditioned on an observation
* discount_cumsum - the discounted rewards-to-go
* numpify - converts torch tensors to numpy objects
* tensor - converts numpy object to float tensor
"""
device = tr.device("cuda:0" if tr.cuda.is_available() else "cpu")

class EnvWrapper:

    def __init__(self,env_name):
        self.env = gym.make(env_name)

    # env helpers
    def setup(self):
        is_discrete = True
        obs_dim = self.env.observation_space.shape[0]
        act_space = self.env.action_space
        if isinstance(act_space,Box):
            is_discrete = False

        act_dim = act_space.shape[0] if not is_discrete else act_space.n
        max_n = self.env._max_episode_steps
        return self.env,obs_dim,act_dim,is_discrete,max_n

    def reset(self):
        obs, rew, done, ep_rews, ep_vals = self.env.reset(), 0, False, [], []
        return obs, rew, done, ep_rews, ep_vals

def print_hyperparamters(params,is_discrete=False):
    m = 40
    title = 'PPO with Categorical Policy' if is_discrete else 'PPO with Gaussian Policy'
    print('='*m)
    n = m - len('| '+title)-1
    print('| '+title + n*' '+'|')
    print('='*m)
    n = m - len('| Hyperparameters:')-1
    print('| Hyperparameters:'+n*' '+'|')
    print('|'+(m-2)*' '+'|')
    for k,v in params.items():
        p = '| '+str(k)+' -- '+ str(v)
        n = m - len(p)-1
        print(p+n*' '+'|')
    print('='*m)


def tensor(x,grad=False):

    if isinstance(x,tr.Tensor):
        return x.clone().detach().requires_grad_(grad)
    else:
        return tr.tensor(x).float()

class Buffer:

    def __init__(self):
        self.reset()

    def store(self,batch=False,**kwargs):
        if batch:
            # stores a list of scalar values in buffer e.g. actions = [0,1,...,0]
            for k,v in kwargs.items():
                if k not in self.buffer:
                    self.buffer[k] = list(v)
                else:
                    self.buffer[k]+=list(v)
        else:
            # stores an individual scalar value in buffer
            for k,v in kwargs.items():
                if k not in self.buffer:
                    self.buffer[k] = [v]
                else:
                    self.buffer[k].append(v)

    def retrieve(self,*args):
        return {arg:self.buffer[arg] for arg in args}

    def reset(self):
        self.buffer = {}

    def __str__(self):
        for k,v in self.buffer.items():
            print(k,'---','...',v[0],'...','total',len(v))
        return ""

def log_likelihood(a,mu,std):
    if len(a.shape)<2: # ensure args always have shape (batch,dim)
        a = a.unsqueeze(0)
        mu = mu.reshape(a.shape)
        std = std.reshape(a.shape)
    summand = (a-mu)**2/(std+EPS)**2 + 2*tr.log(std) + tr.log(2*tensor(np.pi))
    return -.5*tr.sum(summand,1)


def discount_cumsum(rews, gamma):
    y = gamma**np.arange(len(rews))
    gamma_mat=[np.roll(y, i, axis=0) for i in range(len(y))]
    rews_mat = np.repeat([rews], [len(rews)], axis=0)
    rews_mat = np.triu(rews_mat)*gamma_mat
    return list(np.sum(rews_mat,axis=1))

def normalize(x):
    return (x-x.mean())/(x.std()+1e-6)

def as_tensor(x):
    return tr.tensor(x).to(device)

def as_tensors(*args):
    return [as_tensor(x) for x in args]
