import random
import gym
import numpy as np
import torch as tr
import torch.nn as nn

def soft_update(model,model_target,tau):
    for target_param, param in zip(model_target.parameters(), model.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

def hard_update(model,model_target):
    soft_update(model,model_target,1.0)

def print_hyperparamters(params):
    m = 40
    title = 'DDPG with Gaussian Policy'
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


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action

    def reverse_action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return actions


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.model = nn.Sequential(
        nn.Linear(obs_dim + act_dim,hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,1)
        )

    def forward(self, state, action):
        x = tr.cat([state, action], 1)
        return self.model(x)


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()

        self.model = nn.Sequential(
        nn.Linear(obs_dim,hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,act_dim),
        nn.Tanh()
        )
    def forward(self, state):
        return self.model(state)

    def get_action(self, state):
        state  = tr.FloatTensor(state).unsqueeze(0)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0, 0]
