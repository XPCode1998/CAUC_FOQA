import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from apps.ml_models.LGTDM.utils.sample import binary_sampler, uniform_sampler


## GAIN architecture
# Generator
class Generator(nn.Module):
    def __init__(self, dim, h_dim):
        super(Generator, self).__init__()
        self.gru1 = nn.GRU(2*dim, 2*h_dim, 2, batch_first=True)
        self.gru2 = nn.GRU(2*h_dim, h_dim, 2, batch_first=True)
        self.gru3 = nn.GRU(h_dim, dim, 2, batch_first=True)

        self.fc1 = nn.Linear(dim * 2, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, dim)

    def forward(self, x, m):
        # Concatenate Mask and Data
        inputs = torch.cat((x, m), dim=-1)
        output, hn = self.gru1(inputs)
        output = F.relu(output)
        output, hn = self.gru2(output)
        output = F.relu(output)
        output, hn = self.gru3(output)
        # output = F.sigmoid(output)
        G_prob = output
        return G_prob

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, dim, h_dim):
        super(Discriminator, self).__init__()
        self.gru1 = nn.GRU(2*dim, 2*h_dim, 2, batch_first=True)
        self.gru2 = nn.GRU(2*h_dim, h_dim, 2, batch_first=True)
        self.gru3 = nn.GRU(h_dim, dim, 2, batch_first=True)

        self.fc1 = nn.Linear(dim * 2, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, dim)

    def forward(self, x, h):
        # Concatenate Data and Hint
        inputs = torch.cat((x, h), dim=-1)
        output, hn = self.gru1(inputs)
        output = F.relu(output)
        output, hn = self.gru2(output)
        output = F.relu(output)
        output, hn = self.gru3(output)
        output = F.sigmoid(output)
        D_prob = output
        return D_prob


class Model(nn.Module):
    def __init__(self, args, seq_dim, num_label, device) :
        super(Model, self).__init__()
        self.device = device
        self.args = args

        self.seq_dim = seq_dim
        self.h_dim = 2 * int(self.seq_dim)
       

        self.hint_rate = args.gain_hint_rate
        self.alpha = args.gain_alpha

        self.train_missing_ratio_fixed = args.train_missing_ratio_fixed,
        self.missing_ratio = args.missing_ratio


        self.generator = Generator(self.seq_dim, self.h_dim)
        self.discriminator = Discriminator(self.seq_dim, self.h_dim)

        self.g_optimizer = optim.Adam(self.generator.parameters())
        self.d_optimizer = optim.Adam(self.discriminator.parameters())


    # 随机掩码
    def get_mask_rm(self, obs_mask, train_missing_ratio_fixed=False, missing_ratio = None):
        random_mask = torch.rand_like(obs_mask) * obs_mask
        random_mask = random_mask.reshape(len(random_mask), -1)
        for i in range(len(obs_mask)):
            if not train_missing_ratio_fixed:
                missing_ratio = np.random.rand()
            num_observed = obs_mask[i].sum().item()
            num_masked = round(num_observed * missing_ratio)
            random_mask[i][random_mask[i].topk(num_masked).indices] = -1
        random_mask = (random_mask > 0).reshape(obs_mask.shape).float()

        return random_mask


    def forward(self, mode, input_data, label, loss_fn=None):

        # 数据形状变换
        data, obs_mask, gt_mask = [x for x in input_data]  # (B, K, L)

        if mode == 'train':
            rm_mask = self.get_mask_rm(obs_mask, self.train_missing_ratio_fixed, self.missing_ratio)
            X = data * rm_mask
            M = rm_mask
        else:
            X = data * gt_mask
            M = gt_mask
       

        # 随机噪声 ：生成0 ～ 0.01 区间内的均匀分布
        Z = uniform_sampler(0, 0.01, data.shape, self.device)
        # 生成提示矩阵 ： 
        H_temp = binary_sampler(self.hint_rate, data.shape, self.device)
        # 提示掩码
        H = M * H_temp

        X = M * X + (1 - M) * Z


        if mode == 'train':
            # Train Discriminator
            G_sample = self.generator(X, M)
            X_Hat = X * M + G_sample * (1-M)
            D_prob = self.discriminator(X_Hat, H)
            D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) +(1 - M) * torch.log(1 - D_prob + 1e-8))            
            self.d_optimizer.zero_grad()
            D_loss.backward()
            self.d_optimizer.step()

            # Train Generator
            G_sample = self.generator(X, M)
            # 填充后的数据
            X_Hat = X * M + G_sample * (1 - M)
            D_prob = self.discriminator(X_Hat, H)
            G_loss_temp = -torch.mean((1 - M) * torch.log(D_prob + 1e-8))            
            MSE_loss = torch.mean((M * X - M * G_sample)**2) / torch.mean(M)
            G_loss = G_loss_temp + self.alpha * MSE_loss
            self.g_optimizer.zero_grad()
            G_loss.backward()
            self.g_optimizer.step()

            return MSE_loss

        elif mode == 'val':
            # Train Generator
            G_sample = self.generator(X, M)
            # 填充后的数据
            X_Hat = X * M + G_sample * (1 - M)
            D_prob = self.discriminator(X_Hat, H)
            G_loss_temp = -torch.mean((1 - M) * torch.log(D_prob + 1e-8))
            MSE_loss = torch.mean(torch.pow((M * X - M * G_sample), 2)) / torch.mean(M)
            G_loss = G_loss_temp + self.alpha * MSE_loss
            return MSE_loss
        
        elif mode == 'test':
            X_hat = self.generator(X, M)
            imputed_data = M * X + (1 - M) * X_hat
            return imputed_data





    