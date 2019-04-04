import numpy as np
import torch as tr
import torch.nn as nn
from collections import deque
import random
import gym

def print_hyperparamters(params):
    m = 40
    title = 'DQN with Epsilon Greedy Policy'
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

# Utils
def hard_update(model, target_model):
    for target_param, local_param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(local_param.data)


# Q network
class Qnet(nn.Module):

    def __init__(self,obs_dim,act_dim,h_dim):
        super(Qnet,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, act_dim)
        )
    def forward(self,x):
        return self.model(x)

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):

        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))

        state      = tr.FloatTensor(np.float32(state))
        next_state = tr.FloatTensor(np.float32(next_state))
        action     = tr.LongTensor(action)
        reward     = tr.FloatTensor(reward)
        done       = tr.FloatTensor(done)


        return state.squeeze(), action, reward, next_state.squeeze(), done

    def __len__(self):
        return len(self.buffer)
