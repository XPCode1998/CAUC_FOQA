import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from apps.ml_models.LGTDM.modules.Diffusion_Net import CSDI_Net
from apps.ml_models.LGTDM.utils.tools import cosine_beta_schedule, linear_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule, extract, default


class Model(nn.Module):
    def __init__(self, args, seq_dim, num_label, device) :
        super(Model, self).__init__()

        self.device = device
        self.args = args
        # data part
        self.seq_dim = seq_dim
        self.seq_len = args.seq_len
        self.missing_ratio = args.missing_ratio
        # diffusion part
        self.diff_steps = args.diff_steps
        self.diff_layers = args.diff_layers   
        self.res_channels = args.res_channels
        self.skip_channels = args.skip_channels
        # diffusion noise part
        self.beta_schedule = args.beta_schedule
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end
        # diffusion step emb part
        self.step_emb_dim = args.step_emb_dim
        self.c_step = args.c_step
        self.pos_emb_dim = args.pos_emb_dim
        self.fea_emb_dim =args.fea_emb_dim
        # diffusion cond emb part
        self.unconditional = args.unconditional 
        self.pos_emb_dim = args.pos_emb_dim
        self.fea_emb_dim = args.fea_emb_dim
        self.c_cond = self.pos_emb_dim + self.fea_emb_dim
        if not self.unconditional:
            self.c_cond = self.c_cond+1
        # former part
        self.n_heads = args.n_heads
        self.e_layers = args.e_layers
        self.d_ff = args.d_ff
        self.activation = args.activation
        self.dropout = args.dropout
        self.factor = args.factor
        self.distil = args.distil
        # train part
        self.train_missing_ratio_fixed = args.train_missing_ratio_fixed
        # test part
        self.n_samples = args.n_samples

        # model define
        self.model = CSDI_Net(self.res_channels, self.skip_channels, self.diff_steps, self.diff_layers, self.step_emb_dim, 
                             self.c_step, self.unconditional, self.c_cond, self.n_heads, self.e_layers, self.d_ff, 
                             self.activation, self.dropout, self.factor, self.distil)
        
        self.embed_layer = nn.Embedding(num_embeddings=self.seq_dim, embedding_dim=self.fea_emb_dim)
        
        # beta define
        beta_schedule = self.beta_schedule
        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        elif beta_schedule == 'quad':
            beta_schedule_fn = quadratic_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        betas = beta_schedule_fn(self.diff_steps, beta_start=self.beta_start, beta_end=self.beta_end)
        # sampling related parameters
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        # alphas^hat
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas', torch.sqrt(alphas))
        register_buffer('sqrt_betas', torch.sqrt(betas))
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
    

    # 时间嵌入
    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    

    # 生成条件控制信息
    def get_side_info(self, mask):
        B, K, L = mask.shape
        observed_step = torch.arange(self.seq_len, device = self.device)  # (L)
        observed_step = observed_step.unsqueeze(0).expand(B, -1) # (B, L)
        time_embed = self.time_embedding(observed_step, self.pos_emb_dim) # (B, L, pos_emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1) # (B, L, K, pos_emb)
        fea_embed = self.embed_layer(torch.arange(self.seq_dim).to(self.device)) # (K, fea_emb)
        fea_embed = fea_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1) # (B, L, K, fea_emb)
        side_info = torch.cat([time_embed,fea_embed],dim=-1) # (B, L, K,  pos_emb+fea_emb)
        side_info = side_info.permute(0, 3, 2, 1) # (B, pos_emb+fea_emb, K, L)
        if self.unconditional == False:
            side_mask = mask.unsqueeze(1)  # (B, 1, K, L)
            side_info = torch.cat([side_info, side_mask], dim=1) # (B, c_cond, K, L)
        return side_info


    # 生成扩散模型输入
    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.unconditional == True:
            diff_input = noisy_data.unsqueeze(1)  # (B, 1, K, L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)  # (B, 1, K, L)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)  # (B, 1, K, L)
            diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B, 2, K, L)
        return diff_input

    def q_sample(self, data, t, noise=None):
        if noise is None:
            noise = default(noise, lambda: torch.randn_like(data))
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, data.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)
        return sqrt_alphas_cumprod_t * data + sqrt_one_minus_alphas_cumprod_t * noise


    def p_losses(self, data, obs_mask, t, loss_fn, noise=None, mask=None):
        if mask is None:
            mask= self.get_mask_rm(obs_mask, self.train_missing_ratio_fixed, self.missing_ratio)
        side_info = self.get_side_info(mask)
        noise = default(noise, lambda: torch.randn_like(data))
        noisy_data = self.q_sample(data, t, noise)
        diff_input = self.set_input_to_diffmodel(noisy_data, data, mask)
        csdi_input = (diff_input, side_info, t)
        predicted_noise = self.model(diff_input, side_info, t)
        loss_mask = obs_mask - mask
        loss_mask = loss_mask.bool()
        loss = loss_fn(noise[loss_mask], predicted_noise[loss_mask])
        return loss

    @torch.no_grad()
    def get_noisy_obs(self, data, t, noise=None):
        batch = data.shape[0]
        t_index = t
        t = torch.full((batch,), t_index, device=self.device, dtype=torch.long)
        if noise is None:
            noise = default(noise, lambda: torch.randn_like(data))
        sqrt_alphas_t = extract(self.sqrt_alphas, t, data.shape)
        sqrt_betas_t = extract(self.sqrt_betas, t, data.shape)
        return sqrt_alphas_t * data + sqrt_betas_t * noise
    
    @torch.no_grad()
    def p_sample(self, x, diff_input, side_info, t,):
        batch = diff_input.shape[0]
        t_index = t
        t = torch.full((batch,), t_index, device=self.device, dtype=torch.long)

        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        csdi_input = (diff_input, side_info, t)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.model(diff_input, side_info, t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        

    @torch.no_grad()
    def p_sample_loop(self, x_start,  mask, noise=None,):
        B, K, L = x_start.shape
        x_samples = torch.zeros(B, self.n_samples, K, L).to(self.device)
        side_info = self.get_side_info(mask)
        for i in range(self.n_samples):
            x = default(noise, lambda: torch.randn_like(x_start))
            for t in reversed(range(0, self.diff_steps)):
                if self.unconditional:
                    noisy_obs = self.get_noisy_obs(x_start, t)
                    diff_input = noisy_obs * mask + x * (1 - mask)  
                    diff_input = diff_input.unsqueeze(1)
                else:
                    cond_obs = (mask * x_start).unsqueeze(1)
                    noisy_target = ((1 - mask) * x).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1) 
                x = self.p_sample(x, diff_input, side_info, t)
            x_samples[:, i] = x.detach()
        return x_samples.median(dim=1).values
        

    def forward(self, mode, input_data, label, loss_fn=None):
        # CSDI 模型输入 (B, K, L)
        data, obs_mask, gt_mask = [x.permute(0, 2, 1) for x in input_data]

        if mode == 'train':
            t = torch.randint(0, self.diff_steps, (data.shape[0],), device=self.device).long()
            loss = self.p_losses(data, obs_mask, t, loss_fn)
            return loss
        elif mode == 'val':
            val_loss = 0
            for i in range(self.diff_steps):
                t = (torch.ones(data.shape[0], device=self.device)*i).long()
                loss = self.p_losses(data, obs_mask, t, loss_fn, mask=gt_mask)
                val_loss += loss.detach()
            return val_loss / self.diff_steps
        elif mode == 'test':
            x_start =  data * gt_mask
            pred = self.p_sample_loop(x_start, gt_mask)
            pred = pred.permute(0, 2, 1)
            return pred
   