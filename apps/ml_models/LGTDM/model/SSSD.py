import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from apps.ml_models.LGTDM.modules.Diffusion_Net import SSSD_Net
from apps.ml_models.LGTDM.utils.tools import cosine_beta_schedule, linear_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule, extract, default


class Model(nn.Module):
    def __init__(self, args, seq_dim, num_label, device) :
        super(Model, self).__init__()
        self.device = device
        self.args = args
           
        # data part
        self.seq_dim = seq_dim
        self.seq_len = args.seq_len
        self.embed = None
        self.freq = None
        self.missing_ratio = args.missing_ratio
        
        # diffusion model part
        self.diff_steps = args.diff_steps
        self.diff_layers = args.diff_layers   
        self.res_channels = args.res_channels
        self.skip_channels = args.skip_channels

        # diffusion noise part
        self.beta_schedule = args.beta_schedule
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end

        # diffusion step emb part
        self.diffusion_step_embed_dim_in = args.step_emb_dim
        self.diffusion_step_embed_dim_mid = args.c_step
        self.diffusion_step_embed_dim_out = args.c_step

        # s4 part
        self.s4_lmax = args.s4_lmax
        self.s4_d_state = args.s4_d_state
        self.s4_dropout = args.s4_dropout
        self.s4_bidirectional = args.s4_bidirectional
        self.s4_layernorm = args.s4_layernorm
      
        # train part
        self.only_generate_missing = args.only_generate_missing
        self.train_missing_ratio_fixed = args.train_missing_ratio_fixed

        # test part
        self.n_samples = args.n_samples
        
        # 扩散模型
        self.model = SSSD_Net(self.seq_dim, self.res_channels, self.skip_channels, self.seq_dim, self.diff_layers, self.diffusion_step_embed_dim_in, self.diffusion_step_embed_dim_mid,
                              self.diffusion_step_embed_dim_out, self.s4_lmax, self.s4_d_state, self.s4_dropout, self.s4_bidirectional, self.s4_layernorm)
        
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


    # 随机掩码
    def get_mask_rm(self, observed_mask, train_missing_ratio_fixed=False, missing_ratio = None):
        random_mask = torch.rand_like(observed_mask) * observed_mask
        random_mask = random_mask.reshape(len(random_mask), -1)
        for i in range(len(observed_mask)):
            if not train_missing_ratio_fixed:
                missing_ratio = np.random.rand()
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * missing_ratio)
            random_mask[i][random_mask[i].topk(num_masked).indices] = -1
        random_mask = (random_mask > 0).reshape(observed_mask.shape).float()

        return random_mask


    # forward diffusion : from x_0 to x_t
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = default(noise, lambda: torch.randn_like(x_start))
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


    # cal noise
    def p_losses(self, data, obs_mask, t, loss_fn, noise=None, mask=None):
        
        # 在train阶段，在obs_mask基础上，随机生成mask
        if mask is None:
            mask= self.get_mask_rm(obs_mask, self.train_missing_ratio_fixed, self.missing_ratio)
        
        # 随机噪声
        noise = default(noise, lambda: torch.randn_like(data))

        if self.only_generate_missing:
            noise = data * mask  + noise * (1 - mask)
        
        # 前向扩散，计算出t时间步的噪声数据
        x_noisy = self.q_sample(data, t, noise)
        
        # 含有缺失值的初始值x_0
        x_start = data * mask
        
        # 设置模型输入
        diff_input = x_noisy
        
        input_data = (diff_input, x_start, mask, t)

        # 模型预测的t时刻的噪声
        predicted_noise = self.model(diff_input, x_start, mask, t)
        
        
        # 损失值计算点位
        loss_mask = obs_mask - mask
        loss_mask = loss_mask.bool()
        
        # 计算损失
        if self.only_generate_missing:
            loss = loss_fn(noise[loss_mask], predicted_noise[loss_mask])
        else:
            loss = loss_fn(noise, predicted_noise)

        return loss


    # reverse process : from x_t to x_t-1
    # x_t : x value at t step
    # x_start : raw data with missing value
    # data_stampe : raw data's datestamp
    # mask : the position mask of raw data
    # t : current diffusion step 
    @torch.no_grad()
    def p_sample(self, x_t, x_start, mask, t,):
        
        # 批量化时间步
        t_index = t
        t = torch.full((x_t.shape[0],), t_index, device=self.device, dtype=torch.long)  # (B, )
        
        # 初始化当前时间步的beta值
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)
        
         # 设置模型输入
        if self.only_generate_missing:
            diff_input = x_start + x_t * (1 - mask)  # (B, K, L)
        else:
            diff_input = x_t
        input_data = (diff_input, x_start, mask, t)
        
        # 根据模型预测的噪声，计算均值
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * self.model(diff_input, x_start, mask, t) / sqrt_one_minus_alphas_cumprod_t)
        
        # 方差值
        if t_index == 0:
            x_unknown = model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            x_unknown = model_mean + torch.sqrt(posterior_variance_t) * noise
        
        return x_unknown


    @torch.no_grad()
    def p_sample_loop(self, x_start, mask, noise=None,):
        # 初始化随机噪声 x_t
        x = default(noise, lambda: torch.randn_like(x_start))
        for t in reversed(range(0, self.diff_steps)):
            x = self.p_sample(x, x_start, mask, t)
               
        return x
    

    # forward process : from x_t-1 to x_t
    @torch.no_grad()
    def undo(self, x_tminus1, t, noise=None):
        if noise is None:
            noise = default(noise, lambda: torch.randn_like(x_tminus1))
        beta = extract(self.betas, t, x_tminus1.shape)
        x_t = torch.sqrt(1 - beta) * x_tminus1 + torch.sqrt(beta) * torch.randn_like(x_tminus1)

        return x_t


    # 调用填充模型是
    def forward(self, mode, input_data, label, loss_fn=None):

        data, obs_mask, gt_mask = [x.permute(0, 2, 1) for x in input_data]  # (B, K, L)
            
        # 训练阶段
        if mode == 'train':
            t = torch.randint(0, self.diff_steps, (data.shape[0],), device=self.device).long()
            loss = self.p_losses(data, obs_mask, t, loss_fn)
            return loss
        
        # 验证阶段
        elif mode == 'val':
            val_loss = 0
            for i in range(self.diff_steps):
                t = (torch.ones(data.shape[0], device=self.device)*i).long()
                loss = self.p_losses(data, obs_mask, t, loss_fn, mask=gt_mask)
                val_loss += loss.detach()
            return val_loss / self.diff_steps
        
        # 测试阶段
        elif mode == 'test':
            if self.args.model == 'I2TDM':
                self.cls = torch.load(self.pretrain_model_path, map_location=self.device)
            x_start =  data * gt_mask
            imputation = self.p_sample_loop(x_start, gt_mask)
            imputation = imputation.permute(0, 2, 1) # B, L, K
            return imputation
