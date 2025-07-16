import torch
import torch.nn as nn
import torch.nn.functional as F


# Resample part
def get_resample_jump(t_T, jump_length=10, jump_n_sample=10):
    jumps = {}
    for j in range(0, t_T - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1
    t = t_T
    ts = []
    while t >= 1 :
        t = t-1
        ts.append(t)
        if jumps.get(t, 0)>0:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t+1
                ts.append(t)
    ts.append(-1)

    return ts


# GAIN part
def binary_sampler(p, shape, device):
    '''Sample binary random variables.
    
    Args:
        - p: probability of 1
        - rows: the number of rows
        - cols: the number of columns
        
    Returns:
        - binary_random_matrix: generated binary random matrix.
    '''
    unif_random_matrix = torch.rand(shape, device=device)
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix


def uniform_sampler(low, high, shape, device):
    '''Sample uniform random variables using PyTorch.
    
    Args:
        - low: low limit
        - high: high limit
        - rows: the number of rows
        - cols: the number of columns
        
    Returns:
        - uniform_random_matrix: generated uniform random matrix.
    '''
    # 生成0到1之间的随机数
    uniform_random_matrix = torch.rand(shape, device=device)
    # 缩放和位移以匹配low到high的范围
    uniform_random_matrix = uniform_random_matrix * (high - low) + low
    
    return uniform_random_matrix     


class SpectrogramUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels ):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = nn.ConvTranspose2d(in_channels, out_channels,  [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x