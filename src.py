import numpy as np
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def set_seed(seed=2024):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def wCov(X, W=None):
    if W is None:
        return torch.cov(X.T)
    else:
        W_normalized = W / W.sum()
        Center = X - torch.sum(X * W_normalized, 0)
        return (Center.T @ (W_normalized * Center)) / (len(X) - 1) * len(X)
    
def wSTD(X, W=None):
    if W is None:
        return torch.std(X, 0)
    else:
        W_normalized = W / W.sum()
        squared_diff = (X - torch.sum(X * W_normalized, 0)) ** 2
        weighted_variance_vectorized = torch.sum(squared_diff * W_normalized, axis=0)
        return torch.sqrt(weighted_variance_vectorized)

def wMean(X, W=None):
    if W is None:
        return torch.mean(X, 0)
    else:
        W_normalized = W / W.sum()
        return torch.sum(X * W_normalized, 0)
    
def wMCov(X, W=None):
    if W is None:
        return torch.mean(X, 0), torch.cov(X.T)
    else:
        W_normalized = W / W.sum()
        X_mean = torch.sum(X * W_normalized, 0)
        Center = X - X_mean
        return X_mean, (Center.T @ (W_normalized * Center)) / (len(X) - 1) * len(X)
    
class WNetwork(nn.Module):
    def __init__(self, num, times=2):
        super(WNetwork, self).__init__()
        self.W = nn.Parameter(torch.randn(num, 1))
        self.times = times

    def forward(self, X):
        sigmoid_weights = torch.sigmoid(self.W)
        W = sigmoid_weights * (self.times - 1/self.times) + 1/self.times
        mean, cov = wMCov(X, W)
        return W, mean, cov
    
    def getW(self):
        sigmoid_weights = torch.sigmoid(self.W)
        W = sigmoid_weights * (self.times - 1/self.times) + 1/self.times
        return W

class OneNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(OneNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)  

    def forward(self, x):
        tx1 = F.elu(self.fc1(x))
        tx2 = F.elu(self.fc2(x))
        return tx1 + tx2 + tx1*tx2

    
class MyNetwork(nn.Module):
    def __init__(self, t_dim, x_dim, hidden_dim=32):
        super(MyNetwork, self).__init__()
        self.fc1 = OneNet(t_dim + x_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, t, x):
        tx = torch.cat((t, x), dim=1)
        tx1 = F.elu(self.fc1(tx))
        output = self.fc2(tx1)
        return output
    
class RepNetwork(nn.Module):
    def __init__(self, t_dim, x_dim, r_dim=1, hidden_dim=32):
        super(RepNetwork, self).__init__()
        self.fc1 = OneNet(t_dim + x_dim, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, r_dim)

    def forward(self, t, x):
        tx = torch.cat((t, x), dim=1)
        tx1 = F.elu(self.fc1(tx))
        tx2 = self.fc2(tx1)
        output = torch.sigmoid(tx2)
        return output

class torchDataset(Dataset):
    def __init__(self, data):
        self.X = torch.tensor(data.X, dtype=torch.float32)
        self.U = torch.tensor(data.U, dtype=torch.float32)
        self.T = torch.tensor(data.T, dtype=torch.float32)
        self.W = torch.tensor(data.W, dtype=torch.float32)
        self.Y = torch.tensor(data.Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'U': self.U[idx],
            'T': self.T[idx],
            'W': self.W[idx],
            'Y': self.Y[idx]
        }
    