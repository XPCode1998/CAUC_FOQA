import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from apps.ml_models.LGTDM.modules.Residual_Block import TDM_Residual_Block_S, TDM_Residual_Block, CSDI_Residual_Block, SSSD_Residual_group, TDM_N_TAU_Residual_Block
from apps.ml_models.LGTDM.layers.Embed import DiffusionEmbedding, Data_Projection, PositionEmbedding, time_embedding, FixedEmbedding, LabelEmbedding
from apps.ml_models.LGTDM.layers.Conv import  Conv1d, ZeroConv1d, Conv

class TDM_Net(nn.Module):
    def __init__(self, seq_dim, num_label, res_channels, skip_channels, diff_steps, diff_layers, dilation, step_emb_dim, c_step, unconditional, pos_emb_dim, fea_emb_dim, label_emb_dim, c_cond, 
                 seq_len, embed, freq, n_heads, e_layers, d_model, d_ff, activation, dropout, factor, distil, tau_kernel_size, tau_dilation, is_label_emb):
        super(TDM_Net, self).__init__()
        # 残差层维度
        self.res_channls = res_channels
        # 输入投影
        # input_channel = 2 if only_generate_missing else 1
        self.input_projection = Conv1d(2, res_channels, 1)
        # 扩散步嵌入
        self.diffusion_embedding = DiffusionEmbedding(num_steps=diff_steps, embedding_dim=step_emb_dim, projection_dim=c_step)
        # 条件信息嵌入
        self.is_label_emb = is_label_emb
        if self.is_label_emb:
            cond_emb_dim = pos_emb_dim + fea_emb_dim + label_emb_dim
        else:
            cond_emb_dim = pos_emb_dim + fea_emb_dim

        self.is_pos_fea = pos_emb_dim + fea_emb_dim
        
        cond_emb_dim = cond_emb_dim if unconditional else cond_emb_dim+1
        self.label_embedding = LabelEmbedding(num_label, embedding_dim=label_emb_dim)
        if self.is_pos_fea:
            self.pos_embedding = PositionEmbedding(max_len=seq_len, d_model=pos_emb_dim) # (L, pos_dim)
            self.fea_embedding = nn.Embedding(num_embeddings=seq_dim, embedding_dim=fea_emb_dim) # (K, fea_dim)
        if not unconditional:
            self.conditioner_embedding = nn.Sequential(
                Conv1d(cond_emb_dim, c_cond, 1),
                nn.ReLU(True),
                Conv1d(c_cond, c_cond, 1),
                nn.ReLU(True),
            )   
        else:
            self.conditioner_embedding = None
        # 残差层
        self.residual_layers = nn.ModuleList([
            TDM_Residual_Block(res_channels, c_step, 2 ** (i % dilation), unconditional, c_cond, embed, 
                                 freq, n_heads, e_layers, d_model, d_ff, activation, dropout, factor, 
                                 distil, tau_kernel_size, tau_dilation)
            for i in range(diff_layers)
        ])
        # 跳跃投影
        self.skip_projection = Conv1d(res_channels, skip_channels, 1)
        # 输出投影
        self.output_projection = Conv1d(skip_channels, 1, 1)  
        nn.init.zeros_(self.output_projection.weight)

    # 条件信息生成
    def get_cond_info(self, mask, label, shape):
        B, _, K, L = shape
        if self.is_pos_fea:
            observed_step = torch.arange(L, device= mask.device)  # (L)
            observed_step = observed_step.unsqueeze(0).expand(B, -1) # (B, L)
            pos_emb = self.pos_embedding(B, L).to(mask.device) # (B, L, pos_emb)
            pos_emb = pos_emb.unsqueeze(2).expand(B, L, K, -1) # (B, L, K, pos_emb)
            fea_emb = self.fea_embedding(torch.arange(K).to(mask.device)) # (K, fea_emb)
            fea_emb = fea_emb.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1) # (B, L, K, fea_emb)
            cond_info = torch.cat([pos_emb, fea_emb],dim=-1) # (B, L, K, pos_emb+fea_emb)
        if self.is_label_emb:
            label_emb = self.label_embedding(label) # (B, label_emb)
            label_emb = label_emb.unsqueeze(1).expand(B, L, -1).unsqueeze(2).expand(B, L, K, -1).to(mask.device)
            if  self.is_pos_fea:
                cond_info = torch.cat([cond_info, label_emb], dim=-1)
            else:
                cond_info = label_emb
        cond_info = cond_info.permute(0, 3, 2, 1)
        mask_emb = mask.unsqueeze(1) # (B, 1, K, L)
        cond_info = torch.cat([cond_info, mask_emb], dim=1) # (B, c_cond, K, L)
        cond_info = cond_info.reshape(B, -1, K*L) # (B, c_cond, K*L)
        return cond_info

    def forward(self, x, label, mask, t):
    
        B, input_dim, K, L = base_shape = x.shape
        # 输入数据投影
        x = x.reshape(B, input_dim, K*L)
        x = self.input_projection(x)  # (B, C, K*L)
        x = F.relu(x)
        x = x.reshape(B, self.res_channls, K, L)  # (B, C, K, L)
        # 扩散步嵌入
        diffusion_step = self.diffusion_embedding(t)  # (B, c_step)
        # 条件信息嵌入
        cond_info = self.get_cond_info(mask, label, base_shape)
        if self.conditioner_embedding:
            conditioner = self.conditioner_embedding(cond_info)  # (B, c_cond, K, L)
        # 残差连接层
        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, label,  diffusion_step, conditioner)
            skip = skip_connection if skip is None else skip_connection + skip
        x = skip / math.sqrt(len(self.residual_layers))  # （B, C, K, L)
        x = x.reshape(B, self.res_channls, K*L)
        # 跳跃投影
        x = self.skip_projection(x)  # （B, skip_C, K, L)
        x = F.relu(x)
        # 输出投影
        x = self.output_projection(x) # （B, 1, K, L)
        x = x.reshape(B, K, L)   # （B, K, L)

        return x


class TDM_Net_N_TAU(nn.Module):
    def __init__(self, seq_dim, res_channels, skip_channels, diff_steps, diff_layers, dilation, step_emb_dim, c_step, unconditional, pos_emb_dim, fea_emb_dim, c_cond, 
                 seq_len, embed, freq, n_heads, e_layers, d_model, d_ff, activation, dropout, factor, distil, tau_kernel_size, tau_dilation, only_generate_missing):
        super(TDM_Net_N_TAU, self).__init__()
        # 残差层维度
        self.res_channls = res_channels
        # 输入投影
        # input_channel = 2 if only_generate_missing else 1
        self.input_projection = Conv1d(2, res_channels, 1)
        # 扩散步嵌入
        self.diffusion_embedding = DiffusionEmbedding(num_steps=diff_steps, embedding_dim=step_emb_dim, projection_dim=c_step)
        # 条件信息嵌入
        cond_emb_dim = pos_emb_dim + fea_emb_dim
        cond_emb_dim = cond_emb_dim if unconditional else cond_emb_dim+1
        self.pos_embedding = PositionEmbedding(max_len=seq_len, d_model=pos_emb_dim) # (L, pos_dim)
        self.fea_embedding = nn.Embedding(num_embeddings=seq_dim, embedding_dim=fea_emb_dim) # (K, fea_dim)
        if not unconditional:
            self.conditioner_embedding = nn.Sequential(
                Conv1d(cond_emb_dim, c_cond, 1),
                nn.ReLU(True),
                Conv1d(c_cond, c_cond, 1),
                nn.ReLU(True),
            )   
        else:
            self.conditioner_embedding = None
        # 残差层
        self.residual_layers = nn.ModuleList([
            TDM_N_TAU_Residual_Block(res_channels, c_step, 2 ** (i % dilation), unconditional, c_cond, embed, 
                                 freq, n_heads, e_layers, d_model, d_ff, activation, dropout, factor, 
                                 distil, tau_kernel_size, tau_dilation)
            for i in range(diff_layers)
        ])
        # 跳跃投影
        self.skip_projection = Conv1d(res_channels, skip_channels, 1)
        # 输出投影
        self.output_projection = Conv1d(skip_channels, 1, 1)  
        nn.init.zeros_(self.output_projection.weight)

    # 条件信息生成
    def get_cond_info(self, mask, shape):
        B, _, K, L = shape
        observed_step = torch.arange(L, device= mask.device)  # (L)
        observed_step = observed_step.unsqueeze(0).expand(B, -1) # (B, L)
        pos_emb = self.pos_embedding(B, L).to(mask.device) # (B, L, pos_emb)
        pos_emb = pos_emb.unsqueeze(2).expand(B, L, K, -1) # (B, L, K, pos_emb)
        fea_emb = self.fea_embedding(torch.arange(K).to(mask.device)) # (K, fea_emb)
        fea_emb = fea_emb.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1) # (B, L, K, fea_emb)
        cond_info = torch.cat([pos_emb, fea_emb],dim=-1) # (B, L, K, pos_emb+fea_emb)
        cond_info = cond_info.permute(0, 3, 2, 1)
        mask_emb = mask.unsqueeze(1) # (B, 1, K, L)
        cond_info = torch.cat([cond_info, mask_emb], dim=1) # (B, c_cond, K, L)
        cond_info = cond_info.reshape(B, -1, K*L) # (B, c_cond, K*L)
        return cond_info

    def forward(self, x, data_stamp, mask, t):
    
        B, input_dim, K, L = base_shape = x.shape
        # 输入数据投影
        x = x.reshape(B, input_dim, K*L)
        x = self.input_projection(x)  # (B, C, K*L)
        x = F.relu(x)
        x = x.reshape(B, self.res_channls, K, L)  # (B, C, K, L)
        # 扩散步嵌入
        diffusion_step = self.diffusion_embedding(t)  # (B, c_step)
        # 条件信息嵌入
        cond_info = self.get_cond_info(mask, base_shape)
        if self.conditioner_embedding:
            conditioner = self.conditioner_embedding(cond_info)  # (B, c_cond, K, L)
        # 残差连接层
        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, data_stamp,  diffusion_step, conditioner)
            skip = skip_connection if skip is None else skip_connection + skip
        x = skip / math.sqrt(len(self.residual_layers))  # （B, C, K, L)
        x = x.reshape(B, self.res_channls, K*L)
        # 跳跃投影
        x = self.skip_projection(x)  # （B, skip_C, K, L)
        x = F.relu(x)
        # 输出投影
        x = self.output_projection(x) # （B, 1, K, L)
        x = x.reshape(B, K, L)   # （B, K, L)

        return x




class TDM_Net_S(nn.Module):
    def __init__(self, seq_dim, res_channels, skip_channels, diff_steps, diff_layers, dilation, step_emb_dim, c_step, unconditional, pos_emb_dim, fea_emb_dim, c_cond, 
                 seq_len, embed, freq, n_heads, e_layers, d_model, d_ff, activation, dropout, factor, distil, tau_kernel_size, tau_dilation, inverted_channel):
        super(TDM_Net_S, self).__init__()

        # 输入投影
        if inverted_channel:
            self.input_projection = Conv1d(seq_len, res_channels, 1)  # L -> C
        else:
            self.input_projection = Conv1d(seq_dim, res_channels, 1)  # K -> C

        # 扩散步嵌入
        self.diffusion_embedding = DiffusionEmbedding(num_steps=diff_steps, embedding_dim=step_emb_dim, projection_dim=c_step)
        
        # 条件信息嵌入
        if not unconditional:
            if inverted_channel:
                self.conditioner_embedding = nn.Sequential(
                    Conv1d(2*seq_len, c_cond, 1),
                    nn.ReLU(True),
                    Conv1d(c_cond, c_cond, 1),
                    nn.ReLU(True),
                ) 
            else:
                self.conditioner_embedding = nn.Sequential(
                    Conv1d(2*seq_dim, c_cond, 1),
                    nn.ReLU(True),
                    Conv1d(c_cond, c_cond, 1),
                    nn.ReLU(True),
                )        
        else:
            self.conditioner_embedding = None
        
        # 残差层
        self.residual_layers = nn.ModuleList([
            TDM_Residual_Block_S(res_channels, c_step, 2 ** (i % dilation), unconditional, c_cond, 
                               seq_len, embed, freq, n_heads, e_layers, d_model, d_ff, activation, dropout, 
                               factor, distil, tau_kernel_size, tau_dilation, inverted_channel)
            for i in range(diff_layers)
        ])
        
        # 跳跃投影
        self.skip_projection = Conv1d(res_channels, skip_channels, 1)
        
        # 输出投影
        if inverted_channel:
            self.output_projection = Conv1d(skip_channels, seq_len, 1)
        else:
            self.output_projection = Conv1d(skip_channels, seq_dim, 1)  
        
        nn.init.zeros_(self.output_projection.weight)


    # 条件信息生成
    def get_cond_info(self, obs_data, mask):
        cond_info = torch.cat([obs_data, mask], dim=1)
        return cond_info


    def forward(self, x, obs_data, data_stamp, mask, t):
        
        # 输入数据投影
        x = self.input_projection(x) # (B, L, K) -> (B, C, K)  or  (B, K, L) -> (B, C, L)
        x = F.relu(x) # (B, C, K)  or  (B, C, L)
        
        # 扩散步嵌入
        diffusion_step = self.diffusion_embedding(t) # (B, c_step)
        
        # 条件信息嵌入
        cond_info = self.get_cond_info(obs_data, mask)  # (B, 2L, K)  or  (B, 2K, L)
        if self.conditioner_embedding:
            conditioner = self.conditioner_embedding(cond_info)  # (B, 2C, K) or (B, 2C, L)


    
        
        # 残差连接层
        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, data_stamp, diffusion_step, conditioner)
            skip = skip_connection if skip is None else skip_connection + skip
        x = skip / math.sqrt(len(self.residual_layers)) # (B, C, K)  or  (B, C, L)
        
        # 跳跃投影
        x = self.skip_projection(x)  # (B, SC, K)  or  (B, SC, L)
        x = F.relu(x)
        
        # 输出投影
        x = self.output_projection(x) # (B, L, K)  or  (B, K, L)
        
        return x



class CSDI_Net(nn.Module):
    
    def __init__(self, res_channels, skip_channels, diff_steps, diff_layers, step_emb_dim, c_step, unconditional, c_cond, n_heads, e_layers, d_ff, activation, dropout, 
                               factor,  distil):
        super(CSDI_Net, self).__init__()

        self.res_channels = res_channels
        self.c_cond = c_cond
        
        # 输入投影
        input_dim = 2 if not unconditional else 1
        self.input_projection = Conv1d(input_dim, res_channels, 1)
            
        # 扩散步嵌入
        self.diffusion_embedding = DiffusionEmbedding(num_steps=diff_steps, embedding_dim=step_emb_dim, projection_dim=c_step)

        # 残差层
        self.residual_layers = nn.ModuleList([
            CSDI_Residual_Block(res_channels, c_step, c_cond, n_heads, e_layers, d_ff, activation, dropout, factor, distil)
            for _ in range(diff_layers)
        ])

        # 跳跃投影
        self.skip_projection = Conv1d(res_channels, skip_channels, 1)

        # 输出投影
        self.output_projection = Conv1d(skip_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)


    def forward(self, x, side_info, t):

        # x, side_info, t = input_data
        B, input_dim, K, L = x.shape
        x = x.reshape(B, input_dim, K*L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x. reshape(B, self.res_channels, K, L)

        diffusion_step = self.diffusion_embedding(t) 
        
        side_info = side_info.reshape(B, self.c_cond, K*L)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, side_info)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / math.sqrt(len(self.residual_layers)) 
        x = x.reshape(B, self.res_channels, K*L)
        
        x = self.skip_projection(x)  
        x = F.relu(x)
        x = self.output_projection(x) 
        x = x.reshape(B, K, L)

        return x
    

# inchannels : seq_dim
# out_channels : seq_dim
# res_channels : res_channels
# skip_channels : skip_channels
# num_res_layers : diff_layers
# diffusion_step_embed_dim_in : step_emb_dim
# diffusion_step_embed_dim_mid : c_step
# diffusion_step_embed_dim_out : c_step
# s4_lmax
# s4_d_state:
# s4_dropout:
# s4_bidirectional
# s4_layernorm
class SSSD_Net(nn.Module):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels, 
                 num_res_layers,
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(SSSD_Net, self).__init__()

        # 输入投影
        self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU())

        # 残差层
        self.residual_layer = SSSD_Residual_group(res_channels=res_channels, 
                                             skip_channels=skip_channels, 
                                             num_res_layers=num_res_layers, 
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             in_channels=in_channels,
                                             s4_lmax=s4_lmax,
                                             s4_d_state=s4_d_state,
                                             s4_dropout=s4_dropout,
                                             s4_bidirectional=s4_bidirectional,
                                             s4_layernorm=s4_layernorm)
        # 输出投影
        self.final_conv = nn.Sequential(Conv(skip_channels, skip_channels, kernel_size=1),
                                        nn.ReLU(),
                                        ZeroConv1d(skip_channels, out_channels))

    def forward(self, noise, conditional, mask, diffusion_steps):
        
        # noise, conditional, mask, diffusion_steps = input_data 
        diffusion_steps = diffusion_steps.unsqueeze(-1)
        # 条件信息
        conditional = conditional * mask
        conditional = torch.cat([conditional, mask.float()], dim=1)

        # 噪声数据
        x = noise
        x = self.init_conv(x)
        x = self.residual_layer((x, conditional, diffusion_steps))
        y = self.final_conv(x)

        return y

