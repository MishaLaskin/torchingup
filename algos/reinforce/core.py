import numpy as np
import torch as tr
import torch.tensor as tensor
import gym
from gym.spaces import Box
EPS = 1e-8

"""
Utilities for running REINFORCE

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

class Buffer:

    def __init__(self):
        self.reset()

    def reset_episode(self):
        self.ep_o, self.ep_a,self.ep_r = [],[],[]
        self.ep_l = 0

    def reset_epoch(self):
        self.obs_buf, self.acts_buf,self.rtgs_buf = [],[],[]
        self.logp_prev = None
        self.total_eps = 0


    def store_batch(self,ep_obs,ep_acts,ep_rtgs):
        # when episode is over, appends episode vals to batch
        self.obs_buf += ep_obs
        self.acts_buf += ep_acts
        self.rtgs_buf += ep_rtgs
        self.total_eps+=1
    def get_batch(self):

        b_a, b_o = np.array(self.acts_buf).reshape(-1), np.array(self.obs_buf)
        # important: for continuous action space reshape acts to [batch_size,1]
        b_a = b_a.reshape(-1,1)
        b_rtg = np.array(self.rtgs_buf)
        b_o, b_a, b_rtg = tensor(b_o).float(),tensor(b_a).float(),tensor(b_rtg).float()

        return [b_o,b_a,b_rtg]

    def __len__(self):
        return len(self.obs_buf)

    def store_episode(self,o,a,r):
        self.ep_o.append(o)
        self.ep_a.append(a)
        self.ep_r.append(r)
        self.ep_l+=1

    def get_episode(self):
        return self.ep_o,self.ep_a,self.ep_r,self.ep_l

    def reset(self):
        # for evaluation at end of epoch
        self.reset_epoch()
        # for storing an episode
        self.reset_episode()

class Logger:
    """
    Logs relevant values and prints them
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.train_r, self.ep_len = [],[]

    def store(self, train_r=None, ep_len=None, train=True):
        if train:
            self.train_r.append(train_r)
            self.ep_len.append(ep_len)
        else:
            pass

    def get_vals(self):

        vals = np.round([np.mean(self.train_r),np.mean(self.ep_len)],2)
        return vals

    def print_epoch(self, epoch, loss):
        train_r, ep_len = self.get_vals()
        print('epoch {0}  pi_loss {1:.3f} episode length {2}  returns {3}'.format(epoch,loss, ep_len,train_r ))
        self.reset()

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
        return self.env,obs_dim,act_dim

    def reset(self):
        obs, rew, done, ep_rews = self.env.reset(), 0, False, []
        return obs, rew, done, ep_rews

def log_likelihood(a,mu,std):
    summand = (a-mu)**2/(std+EPS)**2 + 2*tr.log(std) + tr.log(2*tensor(np.pi))
    return -.5*tr.sum(summand,1)

def discount_cumsum(rews, gamma):
    y = gamma**np.arange(len(rews))
    gamma_mat=[np.roll(y, i, axis=0) for i in range(len(y))]
    rews_mat = np.repeat([rews], [len(rews)], axis=0)
    rews_mat = np.triu(rews_mat)*gamma_mat
    return np.sum(rews_mat,axis=1)

def numpify(x):
    return x.detach().numpy()

def tensor(x):
    return tr.tensor(x).float()

def print_hyperparamters(params,is_discrete=False):
    m = 40
    title = 'REINFORCE with Categorical Policy' if is_discrete else 'REINFORCE with Gaussian Policy'
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
