import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from sklearn.cluster import KMeans
import os
from apps.ml_models.LGTDM.modules.Diffusion_Net import TDM_Net
from apps.ml_models.LGTDM.utils.tools import cosine_beta_schedule, linear_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule, extract, default
from apps.ml_models.LGTDM.utils.sample import get_resample_jump
from apps.ml_models.LGTDM.modules.Former_Family import TransformerEncoder

# 标签分类模块
class LCM(nn.Module):
     def __init__(self, seq_len, seq_dim, d_model, n_heads, e_layers, d_ff, dropout, activation, factor, distil, embed, freq, num_label): 
        super().__init__()
        self.temporal_former = TransformerEncoder(seq_dim, e_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq)
        self.fc1 = nn.Linear(seq_dim*seq_len, 64)
        self.fc2 = nn.Linear(64, num_label)
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.000001)
        self.loss_function = nn.CrossEntropyLoss()

     def forward(self, x, x_mark=None):
        x = self.temporal_former(x, x_mark)
        x = x.view(-1, x.size(1)*x.size(2))
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.log_softmax(x)         
        return x

     def train_d(self, x, target, flag=1):
        if flag == 1:
            outputs = self.forward(x.detach())
            loss= self.loss_function(outputs, target)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
        else:
            outputs = self.forward(x)
            loss= self.loss_function(outputs, target)
        return loss

# GAN-判别器
class Discriminator(nn.Module):
    def __init__(self, seq_dim, seq_len, hidden_dim=64):
        super().__init__()
        self.seq_dim = seq_dim
        self.seq_len = seq_len
        # 使用1D卷积处理时序数据
        self.conv_net = nn.Sequential(
            nn.Conv1d(seq_dim, hidden_dim, kernel_size=5, stride=2, padding=2),  # (B, hidden, L/2)
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=5, stride=2, padding=2),  # (B, 2h, L/4)
            nn.InstanceNorm1d(hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim*2, hidden_dim*4, kernel_size=5, stride=2, padding=2),  # (B, 4h, L/8)
            nn.InstanceNorm1d(hidden_dim*4),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1)  # (B, 4h, 1)
        )
        
        # 最终分类层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.000001)
        self.loss_function = nn.BCELoss()

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def train_d(self, x, target):
        outputs = self.forward(x)
        if target == 1:
            labels= torch.ones_like(outputs)
        elif target == 0:
            labels= torch.zeros_like(outputs)
        loss = self.loss_function(outputs, labels) 
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item()


class Model(nn.Module):
    def __init__(self, args, seq_dim, num_label, device) :
        super(Model, self).__init__()
        self.device = device
        self.args = args
        self.missing_ratio = args.missing_ratio
        
        # data part
        self.seq_len = args.seq_len
        self.seq_dim = seq_dim
        self.num_label = num_label
        
        # diffusion model part
        self.diff_steps = args.diff_steps
        self.diff_layers = args.diff_layers   
        self.res_channels = args.res_channels
        self.skip_channels = args.skip_channels

        # conv part
        self.dilation = args.dilation

        # former part
        self.n_heads = args.n_heads
        self.e_layers = args.e_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.activation = args.activation
        self.dropout = args.dropout
        self.factor = args.factor
        self.distil = args.distil

        # tau part
        self.tau_kernel_size = args.tau_kernel_size
        self.tau_dilation = args.tau_dilation

        # diffusion noise part
        self.beta_schedule = args.beta_schedule
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end

        # diffusion step emb part
        self.step_emb_dim = args.step_emb_dim
        self.c_step = args.c_step

        # diffusion conditional emb part
        self.unconditional = args.unconditional
        self.c_cond = args.c_cond
        self.pos_emb_dim = args.pos_emb_dim
        self.fea_emb_dim = args.fea_emb_dim
        
        # label emb part
        self.is_label_emb = args.is_label_emb
        self.label_emb_dim = args.label_emb_dim

        # label classifier part
        self.lcm_hidden_dim = args.lcm_hidden_dim

        # train part
        self.only_generate_missing = args.only_generate_missing
        self.train_missing_ratio_fixed = args.train_missing_ratio_fixed

        # test part
        self.n_samples = args.n_samples
        self.select_k = args.select_k

        embed = None 
        freq = None

        self.gan_loss_ratio = args.gan_loss_ratio
        self.classifier_loss_ratio = args.classifier_loss_ratio
        
        # 扩散模型
        self.model = TDM_Net(self.seq_dim, self.num_label, self.res_channels, self.skip_channels, self.diff_steps, self.diff_layers, self.dilation, self.step_emb_dim, 
                             self.c_step, self.unconditional, self.pos_emb_dim, self.fea_emb_dim, self.label_emb_dim, self.c_cond, self.seq_len, embed, freq, self.n_heads, 
                             self.e_layers, self.d_model, self.d_ff, self.activation, self.dropout, self.factor, self.distil, self.tau_kernel_size, self.tau_dilation, self.is_label_emb)
        
        # diffusion noise set
        if self.beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif self.beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif self.beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        elif self.beta_schedule == 'quad':
            beta_schedule_fn = quadratic_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {self.beta_schedule}')
        
        betas = beta_schedule_fn(self.diff_steps, beta_start=self.beta_start, beta_end=self.beta_end)
        # sampling related parameters
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas', torch.sqrt(1. / alphas))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # GAN-判别器
        self.discriminator = Discriminator(
            seq_dim=seq_dim,
            seq_len=args.seq_len,
            hidden_dim=args.d_hidden_dim  # 需要添加到args参数
        )

        # 标签分类模块
        self.LCM = LCM(self.seq_len, self.seq_dim, self.d_model, self.n_heads, self.e_layers, self.d_ff, self.dropout, self.activation, self.factor, self.distil, embed, freq, self.num_label)

        
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


    def q_sample(self, x_start, t, noise=None):
        r'''
        forward process : from x_0 to x_t
        Args:
            x_start : raw data with missing value
            t : current diffusion step
            noise : random sample from gaussian distribution 
        Returns:
            x_t : x value at t step
        '''
        if noise is None:
            noise = default(noise, lambda: torch.randn_like(x_start))
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t


    def p_losses_t(self, data, label, obs_mask, t, loss_fn, noise=None, mask=None):
        r'''
        cal loss for predicting noise and true noise at t step
        Args:
            data : raw data (obs masked)
            labele : raw data's datestamp
            obs_mask : the position mask of raw data
            t : current diffusion step
            loss : loss function
            noise : random sample from gaussian distribution 
            mask : random mask based on the obs_mask (in train) or ground truth mask (in val)
            writer : TensorBoard SummaryWriter
            flaf : the flag for TensorBoard to draw moel 
        Returns:
            x_t : loss value
        '''

        # 在train阶段，在obs_mask基础上，随机生成mask
        if mask is None:
            mask= self.get_mask_rm(obs_mask, self.train_missing_ratio_fixed, self.missing_ratio)
        
        # 随机噪声
        noise = default(noise, lambda: torch.randn_like(data))
        
        # 前向扩散，计算出t时间步的噪声数据
        x_noisy = self.q_sample(data, t, noise)
        
        # 含有缺失值的初始值x_0
        x_start = data * mask
        
        # 设置模型输入
        if self.only_generate_missing:
            diff_input_obs = x_start.unsqueeze(1)  # (B, K, L, 1)
            diff_input_noise = (x_noisy * (1-mask)).unsqueeze(1)  # (B, K, L, 1)
            diff_input = torch.cat([diff_input_obs, diff_input_noise], 1)  # (B, K, L, 2)
        else:
            diff_input_obs = x_start.unsqueeze(1)  # (B, K, L, 1)
            diff_input_noise = x_noisy.unsqueeze(1)  # (B, K, L, 1)
            diff_input = torch.cat([diff_input_obs, diff_input_noise], 1)  # (B, K, L, 2)
        input_data = (diff_input, label, mask, t)
        
        # 模型预测的t时刻的噪声
        predicted_noise = self.model(diff_input, label, mask, t)
        
        # 损失值计算点位
        loss_mask = obs_mask - mask
        loss_mask = loss_mask.bool()
        
        # # 计算损失
        # if self.only_generate_missing:
        #     loss = loss_fn(noise[loss_mask], predicted_noise[loss_mask])
        # else:
        #     loss = loss_fn(noise, predicted_noise)

        # 计算预测的原始数据
        # 真实的原始数据

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        x_pred = (x_noisy - sqrt_one_minus_alphas_cumprod_t * predicted_noise)/sqrt_alphas_cumprod_t
        

        return noise, predicted_noise, obs_mask, mask, x_pred


    # def p_losses(self, data, label, obs_mask, t, loss_fn, noise=None, mask=None, mode = 'train'):

    #     if mode == 'train':
    #     # 训练判别器
    #         noise, predicted_noise, obs_mask, mask = self.p_losses_t(data, label, obs_mask, t, loss_fn, noise, mask)
    #         predicted_noise = predicted_noise*(obs_mask-mask)+ noise*mask # （B, K, L)
    #         d_loss1 = self.discriminator.train_d(noise, 1)
    #         d_loss2 = self.discriminator.train_d(predicted_noise.detach(), 0)
        
    #     # 训练生成器
    #     noise, predicted_noise, obs_mask, mask = self.p_losses_t(data, label, obs_mask, t, loss_fn, noise, mask)
    #     predicted_noise = predicted_noise*(obs_mask-mask)+ noise*mask # （B, K, L)
    #     d_output = self.discriminator(predicted_noise)
    #     labels= torch.ones_like(d_output)
    #     d_loss = self.discriminator.loss_function(d_output, labels)
        

    #     # 损失值计算点位
    #     loss_mask = obs_mask - mask
    #     loss_mask = loss_mask.bool()

    #     # 计算损失
    #     if self.only_generate_missing:
    #         f_loss = loss_fn(noise[loss_mask], predicted_noise[loss_mask])
    #     else:
    #         f_loss = loss_fn(noise, predicted_noise)
        
    #     loss = (1-self.adv_loss_ratio)*f_loss + self.adv_loss_ratio*d_loss

    #     if mode == 'train':
    #         return d_loss1+d_loss2, d_loss.item(), loss
    #     else:
    #         return loss


    def p_losses(self, data, label, obs_mask, t, loss_fn, noise=None, mask=None, mode = 'train', epoch=100):

        if mode == 'train' and (epoch<20 or epoch% 2 == 0) and epoch<40:
        # 训练判别器
            noise, predicted_noise, obs_mask, mask, x_pred = self.p_losses_t(data, label, obs_mask, t, loss_fn, noise, mask)
            predicted_noise = predicted_noise*(obs_mask-mask)+ noise*mask # （B, K, L)
            d_loss1 = self.discriminator.train_d(data, 1)
            d_loss2 = self.discriminator.train_d(x_pred.detach(), 0)
        
        # 训练生成器
        noise, predicted_noise, obs_mask, mask, x_pred = self.p_losses_t(data, label, obs_mask, t, loss_fn, noise, mask)
        predicted_noise = predicted_noise*(obs_mask-mask)+ noise*mask # （B, K, L)
        d_output = self.discriminator(x_pred)
        labels= torch.ones_like(d_output)
        d_loss = self.discriminator.loss_function(d_output, labels)

        if mode == 'train' and 20<epoch<=40:
            self.LCM.train_d(x_pred.permute(0, 2, 1), label, flag=1)
            c_loss = 0
        elif mode == 'train' and 40<epoch<50:
            c_loss = self.LCM.train_d(x_pred.permute(0, 2, 1), label,flag=0)
        else:
            c_loss = 0
        
        
        # 损失值计算点位
        loss_mask = obs_mask - mask
        loss_mask = loss_mask.bool()

        # 计算损失
        if self.only_generate_missing:
            f_loss = loss_fn(noise[loss_mask], predicted_noise[loss_mask])
        else:
            f_loss = loss_fn(noise, predicted_noise)
        
        if c_loss == 0:
            loss = (1-self.gan_loss_ratio)*f_loss + self.gan_loss_ratio*d_loss
        else:
            loss = (1-self.gan_loss_ratio- self.classifier_loss_ratio)*f_loss + self.gan_loss_ratio*d_loss + self.classifier_loss_ratio*c_loss

        
        if mode == 'train':
            if mode == 'train' and (epoch<20 or epoch% 2 == 0) and epoch<40:
                return c_loss, d_loss1+d_loss2, d_loss.item(), loss
            else:
                return c_loss, 0, d_loss.item(), loss
        else:
            return loss


    @torch.no_grad()
    def p_sample(self, x_t, x_start, label, mask, t):
        r'''
        Reverse process : from x_t to x_t-1
        Args:
            x_t : x value at t step
            x_start : raw data with missing value (gt masked)
            labele : raw data's datestamp
            mask : the position mask of raw data
            t : current diffusion step 
        Returns:
            x_tminus1 : x value at t-1 step
        '''
        
        # 批量化时间步
        t_index = t
        t = torch.full((x_t.shape[0],), t_index, device=self.device, dtype=torch.long)  # (B, )
        
        # 初始化当前时间步的beta值
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)
        
        # 设置模型输入
        if self.only_generate_missing:
            diff_input_obs = x_start.unsqueeze(1)  # (B, K, L, 1)
            diff_input_noise = (x_t * (1-mask)).unsqueeze(1)  # (B, K, L, 1)
            diff_input = torch.cat([diff_input_obs, diff_input_noise], 1)  # (B, K, L, 2)
        else:
            diff_input_obs = x_start.unsqueeze(1)  # (B, K, L, 1)
            diff_input_noise = x_t.unsqueeze(1)  # (B, K, L, 1)
            diff_input = torch.cat([diff_input_obs, diff_input_noise], 1)  # (B, K, L, 2)
        input_data = (diff_input, label, mask, t)
        
        # 根据模型预测的噪声，计算均值
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * self.model(diff_input, label, mask, t) / sqrt_one_minus_alphas_cumprod_t)
        
        # 方差值
        if t_index == 0:
            x_unknown = model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            x_unknown = model_mean + torch.sqrt(posterior_variance_t) * noise
        
        x_tminus1 = x_unknown

        return x_tminus1
    

    # 聚类选择
    @torch.no_grad()
    def nearest_mean(self, data):
        batch_size, sample_num, K, L = data.shape
        nearest_means = torch.zeros((batch_size, K, L), device=data.device)

        for i in range(batch_size):
            batch_data = data[i].view(-1, K * L).cpu().numpy()  # 将PyTorch张量转换为NumPy数组
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(batch_data)
            labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_

            counts = np.bincount(labels)
            
            if len(counts) == 2:
                p1 = counts[0]/len(labels)
                p2 = counts[1]/len(labels)
                
                new_p1 = math.log(p2)/(math.log(p1)+math.log(p2))
                new_p2 = math.log(p1)/(math.log(p1)+math.log(p2))
                cluster_cent = new_p1 * cluster_centers[0] + new_p2 * cluster_centers[1]
            else:
                cluster_cent = cluster_centers[0]

            nearest_means[i] = torch.tensor(cluster_cent.reshape(K, L), device=data.device)

        return nearest_means


    @torch.no_grad()
    def p_sample_loop(self, x_start, label, mask, noise=None):
        r'''
        Reverse process : from x_t to x_0
        Args:
            x_start : raw data with missing value (gt masked)
            labele : raw data's datestamp
            mask : the position mask of raw data
            noise : random sample from gaussian distribution
        Returns:
            x_samples : imputation result for x_start
        '''
        
        B, K, L = x_start.shape

        # 多次填充(n_samples次), 最后求平均, 保证填充结果的稳定性
        x_samples = torch.zeros((B, self.n_samples, K, L), device=self.device)
        
        data_time_list = []
       
        # n_sample次填充求平均
        for i in range(self.n_samples):
            # 初始化随机噪声 x_t
            x = default(noise, lambda: torch.randn_like(x_start))
            # 逆向扩散去噪声
            for t in reversed(range(0, self.diff_steps)):
                x = self.p_sample(x, x_start, label, mask, t)
                if i == 1:
                    data_time_list.append(x.permute(0,2,1).detach())
            x_samples[:, i] = x.detach()
        
        # 聚类选择
        if self.select_k:
            x_samples = x_samples * ((1-mask).unsqueeze(1))
            re_x_samples = self.nearest_mean(x_samples)
            return re_x_samples, x_samples, data_time_list
        else:
            return x_samples.median(dim=1).values, x_samples, data_time_list

    # 调用填充模型是
    def forward(self, mode, input_data, label, loss_fn=None, epochs=None):
        # 数据形状变换
        data, obs_mask, gt_mask = [x.permute(0, 2, 1) for x in input_data]  # (B, K, L)
        
        
        # 训练阶段
        if mode == 'train':
            t = torch.randint(0, self.diff_steps, (data.shape[0],), device=self.device).long()
            c_loss, d_loss, g_loss, loss = self.p_losses(data, label, obs_mask, t, loss_fn, mode = 'train', epoch=epochs)
            return c_loss, d_loss, g_loss, loss
        
        # 验证阶段
        elif mode == 'val':
            val_loss = 0
            for i in range(self.diff_steps):
                t = (torch.ones(data.shape[0], device=self.device)*i).long()
                loss = self.p_losses(data, label, obs_mask, t, loss_fn, mask=gt_mask, mode = 'val')
                val_loss += loss.detach()
            return val_loss / self.diff_steps
        
        # 测试阶段
        elif mode == 'test':
            x_start =  data * gt_mask
            imputation, samples, data_time_list = self.p_sample_loop(x_start, label, gt_mask)
            imputation = imputation.permute(0, 2, 1) # B, L, K
            samples = samples.permute(0, 1, 3, 2)
            return imputation, samples, data_time_list



   