import numpy as np
import copy
import random
import torch
import math
from tqdm import tqdm


class QHD_Model(object):
    def __init__(self,
                dimension=10000,
                n_actions=2,
                n_obs=4,
                epsilon=1.0,
                epsilon_decay=0.995,
                minimum_epsilon=0.01,
                reward_decay=0.9,
                lr = 0.005,
                train_sample_size = 5,
                device = 'cpu',
                q_levels = 1001 ###(15+5)*50 + 1
                ):
        self.D = dimension
        self.n_actions = n_actions
        self.n_obs = n_obs
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.minimum_epsilon = minimum_epsilon
        self.reward_decay = reward_decay
        self.lr = lr
        self.ts = train_sample_size
        self.device = device
        
        self.logs = [] # temp log for current episode

        self.classes = q_levels
        self.model = torch.zeros(self.n_actions, self.classes, self.D).to(device)
        self.basis = torch.randn(self.D, self.n_obs).to(device)
        self.base = torch.empty(self.D).uniform_(0.0, 2*math.pi).to(device)
        self.delay_model = copy.deepcopy(self.model)
        
        self.model_update_counter = 0
    
    def idx2value(self, index):
        value = -5 + 0.02*index.item()
        return value
    
    def value2idx(self, value):
        index = int( (np.clip(value,-5,15) + 5) / 0.02)
        return index

    def encode(self, x):
        if len(x.shape) == 1:
            x = x[None]
        n = x.size(0)
        bsize = math.ceil(0.01*n)
        h = torch.empty(n, self.D, device=x.device, dtype=x.dtype)
        temp = torch.empty(bsize, self.D, device=x.device, dtype=x.dtype)
        for i in range(0, n, bsize):
            torch.matmul(x[i:i+bsize], self.basis.T, out=temp)
            torch.add(temp, self.base, out=h[i:i+bsize])
            h[i:i+bsize].cos_().mul_(temp.sin_())
        return h
    
    def cos_cdist(self, x1, x2, eps=1e-8):
        eps = torch.tensor(eps, device=x1.device)
        norms1 = x1.norm(dim=1).unsqueeze_(1).max(eps)
        norms2 = x2.norm(dim=1).unsqueeze_(0).max(eps)
        cdist = x1 @ x2.T
        cdist.div_(norms1).div_(norms2)
        return cdist

    def store_transition(self, s, a, r, n_s, done):
        self.logs.append((s,a,r,n_s,done))
        if len(self.logs) > 102400:
            self.logs.pop(0)

    def act(self, obs):
        if (random.random() <= self.epsilon):
            action = random.randint(0, self.n_actions-1)
        else:
            obs = self.encode(torch.FloatTensor(obs).to(self.device))
            q_value = []
            for a in range(self.n_actions):
                q_idx = self.cos_cdist(obs, self.model[a]).argmax(1)
                q_value.append(self.idx2value(q_idx))
            action = np.argmax(q_value)
        return action

    def feedback(self):

        if len(self.logs) < self.ts:
            logs = self.logs
        else:
            logs = random.sample(self.logs, self.ts)
        #for i in range(5):    
        for log in logs:
            (obs, action, reward, next_obs, done) = log
            obs = torch.FloatTensor(obs).to(self.device)
            next_obs = torch.FloatTensor(next_obs).to(self.device)
            obs_ = self.encode(torch.FloatTensor(obs).to(self.device))
            next_obs_ = self.encode(torch.FloatTensor(next_obs).to(self.device))
            scores = self.cos_cdist(obs_, self.model[action])
            y_pred = scores.argmax(1) ## index of predicted q-value
            q_list = []
            for a in range(self.n_actions):
                q_list.append(self.idx2value(self.cos_cdist(next_obs_, self.model[a]).argmax(1)))
            y_true = torch.LongTensor([self.value2idx(reward + (1-float(done))*self.reward_decay*max(q_list))]) ## index of true q-value
            
            #print(self.idx2value(y_pred), reward + (1-float(done))*self.reward_decay*max(q_list), action)
            
            ############ Update the Classification Model #############
            if not torch.equal(y_pred, y_true):
                self.model[action][y_pred] -= self.lr * (1.0 - torch.max(scores)) * obs_
                self.model[action][y_true] += self.lr * (1.0 - torch.max(scores)) * obs_
                