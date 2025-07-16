import torch
import torch.nn as nn
import math
from apps.ml_models.LGTDM.modules.Former_Family import iTransformer, TransformerEncoder, InformerEncoder, FlowformerEncoder, TemporalFormer, iTemporalFormer, TransformerEncoderWithoutEmbedding, XP_TAU
from apps.ml_models.LGTDM.layers.SelfAttention_Family import TemporalAttention1d, TemporalAttention, TAU
from apps.ml_models.LGTDM.layers.Embed import calc_diffusion_step_embedding
from apps.ml_models.LGTDM.layers.Conv import  Conv1d, Conv
from apps.ml_models.LGTDM.layers.S4Layer import S4Layer
from apps.ml_models.LGTDM.utils.tools import swish



class TDM_Residual_Block(nn.Module):
    def __init__(self, res_channels, c_step, dilation, unconditional, c_cond, embed, freq, n_heads, e_layers, d_model, d_ff, 
                 activation, dropout, factor, distil, tau_kernel_size, tau_dilation):
        super(TDM_Residual_Block, self).__init__()

        # 扩散步投影
        self.diffusion_projection = nn.Linear(c_step, res_channels)

        # 时间注意力
        self.temporal_former = TransformerEncoder(res_channels, e_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq)
        # 特征注意力
        self.feature_former =  TransformerEncoder(res_channels, e_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq)
        
        # 时空注意力
        # 原始TAU
        self.tau = TemporalAttention(res_channels, kernel_size=tau_kernel_size, dilation=tau_dilation)
        # 改为1维TAU
        self.tau1d = TemporalAttention1d(res_channels, kernel_size=tau_kernel_size, dilation=tau_dilation)
        # # 重新设计的TAU
        # self.xp_tau = TAU(res_channels, n_heads, bias=True, kernel_size=tau_kernel_size, dilation=tau_dilation)

        self.xp_tau = XP_TAU(res_channels, tau_kernel_size, tau_dilation, e_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq)

        # 膨胀投影
        self.dilated_conv = Conv1d(res_channels, 2 * res_channels, 3, padding=dilation, dilation=dilation)
        
        # 条件投影
        if not unconditional: # conditional model
            self.conditoner_projection = Conv1d(c_cond, 2 * res_channels, 1)
        else: # unconditional model
            self.conditoner_projection = None
        
        # 输出投影
        self.output_projection = Conv1d(res_channels, 2 * res_channels, 1)

    def forward_temporal(self, y, base_shape):
        B, C, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 2, 1, 3).reshape(B*K, C, L)
        y = self.temporal_former(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = y.reshape(B, K, C, L).permute(0, 2, 1, 3).reshape(B, C, K*L) 
        return y
    
    def forward_feature(self, y, base_shape):
        B, C, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 3, 1, 2).reshape(B*L, C, K)
        y = self.feature_former(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = y.reshape(B, L, C, K).permute(0, 2, 3, 1).reshape(B, C, K*L) 
        return y
    
    def forwar_tau_temporal(self, y, base_shape):
        B, C, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 2, 1, 3).reshape(B*K, C, L)
        y = self.tau1d(y)
        y = y.reshape(B, K, C, L).permute(0, 2, 1, 3).reshape(B, C, K*L)
        return y
    
    def forward_tau_feature(self, y, base_shape):
        B, C, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 3, 1, 2).reshape(B*L, C, K)
        y = self.tau1d(y)
        y = y.reshape(B, L, C, K).permute(0, 2, 3, 1).reshape(B, C, K*L) 
        return y
    

    def forward(self, x, label, diffusion_step, conditoner=None):
        B, C, K, L = base_shape = x.shape
        x = x.reshape(B, C, K*L) # (B, C, K*L)
        # 扩散步投影
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1) # (B, C, 1)
        y = x + diffusion_step  # (B, C, K*L)
        y = y.reshape(base_shape)
        y = self.xp_tau(y)
        y = y.reshape(B, C, K*L)

        # y = self.forward_temporal(y, base_shape)
        # y = self.forward_feature(y, base_shape)
        # y = y.reshape(base_shape)
        # y = self.tau(y)
        # y = y.reshape(B, C, K*L)
        # y = self.forward_temporal(y, base_shape)
        # y = self.forward_feature(y, base_shape)
        # y = self.forwar_tau_temporal(y, base_shape)
        # y = self.forward_tau_feature(y, base_shape)
        # y = y.reshape(base_shape)
        # y = self.xp_tau(y)
        # y = y.reshape(B, C, K*L)
        # 膨胀投影
        y = self.dilated_conv(y) # (B, 2C, K*L)
        # 条件信息投影
        if self.conditoner_projection is not None:
            conditoner = self.conditoner_projection(conditoner) # (B, 2C, K*L)
            y = y + conditoner
        # 分割
        gate, filter = torch.chunk(y, 2, dim=1) # (B, C, K*L)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        # 输出投影
        y = self.output_projection(y) # (B, 2C, K*L)
        residual, skip = torch.chunk(y, 2, dim=1) # (B, C, K*L)
        # 形状恢复
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        
        return (x + residual) / math.sqrt(2.0), skip


class TDM_N_TAU_Residual_Block(nn.Module):
    def __init__(self, res_channels, c_step, dilation, unconditional, c_cond, embed, freq, n_heads, e_layers, d_model, d_ff, 
                 activation, dropout, factor, distil, tau_kernel_size, tau_dilation):
        super(TDM_N_TAU_Residual_Block, self).__init__()

        # 扩散步投影
        self.diffusion_projection = nn.Linear(c_step, res_channels)

        # 时间注意力
        self.temporal_former = TransformerEncoder(res_channels, e_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq)
        # 特征注意力
        self.feature_former =  TransformerEncoder(res_channels, e_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq)
        
        # 时空注意力
        # 原始TAU
        self.tau = TemporalAttention(res_channels, kernel_size=tau_kernel_size, dilation=tau_dilation)
        # 改为1维TAU
        self.tau1d = TemporalAttention1d(res_channels, kernel_size=tau_kernel_size, dilation=tau_dilation)
        # # 重新设计的TAU
        # self.xp_tau = TAU(res_channels, n_heads, bias=True, kernel_size=tau_kernel_size, dilation=tau_dilation)

        self.xp_tau = XP_TAU(res_channels, tau_kernel_size, tau_dilation, e_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq)

        # 膨胀投影
        self.dilated_conv = Conv1d(res_channels, 2 * res_channels, 3, padding=dilation, dilation=dilation)
        
        # 条件投影
        if not unconditional: # conditional model
            self.conditoner_projection = Conv1d(c_cond, 2 * res_channels, 1)
        else: # unconditional model
            self.conditoner_projection = None
        
        # 输出投影
        self.output_projection = Conv1d(res_channels, 2 * res_channels, 1)

    def forward_temporal(self, y, base_shape):
        B, C, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 2, 1, 3).reshape(B*K, C, L)
        y = self.temporal_former(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = y.reshape(B, K, C, L).permute(0, 2, 1, 3).reshape(B, C, K*L) 
        return y
    
    def forward_feature(self, y, base_shape):
        B, C, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 3, 1, 2).reshape(B*L, C, K)
        y = self.feature_former(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = y.reshape(B, L, C, K).permute(0, 2, 3, 1).reshape(B, C, K*L) 
        return y
    
    def forwar_tau_temporal(self, y, base_shape):
        B, C, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 2, 1, 3).reshape(B*K, C, L)
        y = self.tau1d(y)
        y = y.reshape(B, K, C, L).permute(0, 2, 1, 3).reshape(B, C, K*L)
        return y
    
    def forward_tau_feature(self, y, base_shape):
        B, C, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 3, 1, 2).reshape(B*L, C, K)
        y = self.tau1d(y)
        y = y.reshape(B, L, C, K).permute(0, 2, 3, 1).reshape(B, C, K*L) 
        return y
    

    def forward(self, x, data_stamp, diffusion_step, conditoner=None):
        B, C, K, L = base_shape = x.shape
        x = x.reshape(B, C, K*L) # (B, C, K*L)
        # 扩散步投影
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1) # (B, C, 1)
        y = x + diffusion_step  # (B, C, K*L)
        # y = y.reshape(base_shape)
        # y = self.xp_tau(y)
        # y = y.reshape(B, C, K*L)
        # 膨胀投影
        y = self.dilated_conv(y) # (B, 2C, K*L)
        # 条件信息投影
        if self.conditoner_projection is not None:
            conditoner = self.conditoner_projection(conditoner) # (B, 2C, K*L)
            y = y + conditoner
        # 分割
        gate, filter = torch.chunk(y, 2, dim=1) # (B, C, K*L)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        # 输出投影
        y = self.output_projection(y) # (B, 2C, K*L)
        residual, skip = torch.chunk(y, 2, dim=1) # (B, C, K*L)
        # 形状恢复
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        
        return (x + residual) / math.sqrt(2.0), skip


class TDM_Residual_Block_S(nn.Module):
    def __init__(self, res_channels, c_step, dilation, unconditional, c_cond, seq_len, embed, freq, n_heads, e_layers, d_model, d_ff, 
                 activation, dropout, factor, distil, tau_kernel_size, tau_dilation, inverted_channel):
        super(TDM_Residual_Block_S, self).__init__()

        # 扩散步投影
        self.diffusion_projection = nn.Linear(c_step, res_channels)
        # 时间注意力
        self.temporal_attention = TemporalAttention1d(res_channels, kernel_size=tau_kernel_size, dilation=tau_dilation)
        # 特征注意力
        self.feature_attention = TransformerEncoder(res_channels, e_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq)
        # 膨胀投影
        self.dilated_conv = Conv1d(res_channels, 2 * res_channels, 3, padding=dilation, dilation=dilation)
        # 条件投影
        if not unconditional: # conditional model
            self.conditoner_projection = Conv1d(c_cond, 2 * res_channels, 1)
        else: # unconditional model
            self.conditoner_projection = None
        # 输出投影
        self.output_projection = Conv1d(res_channels, 2 * res_channels, 1)


    def forward(self, x, data_stamp, diffusion_step, conditoner=None):
        
        # 扩散步投影
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)  # (B, c_step) -> (B, C, 1)
        
        y = x + diffusion_step   # (B, C, L) or (B, C, K)
        
        # # 时间注意力
        # y = self.temporal_attention(y) # (B, C, L)
        
        # # 特征注意力
        # y = self.feature_attention(y.permute(0, 2, 1), data_stamp.permute(0, 2, 1)).permute(0, 2, 1) # (B, C, L)
        
        # 膨胀投影
        y = self.dilated_conv(y)  # (B, 2C, L) or (B, 2C, K)
        
        # 条件信息投影
        if self.conditoner_projection is not None: # using a unconditional model
            conditoner = self.conditoner_projection(conditoner)  # (B, 2C, L) or (B, 2C, K)
            y = y + conditoner
        
        # 分割
        gate, filter = torch.chunk(y, 2, dim=1)  # (B, C, L) or (B, C, K)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        # 输出投影
        y = self.output_projection(y)  # (B, 2C, L) or (B, 2C, K)
        residual, skip = torch.chunk(y, 2, dim=1)  # (B, C, L) or (B, C, K)

        return (x + residual) / math.sqrt(2.0), skip
    

class CSDI_Residual_Block(nn.Module):
    def __init__(self, res_channels, c_step,  c_cond,  n_heads, e_layers, d_ff, activation, dropout, factor, distil):
        super(CSDI_Residual_Block, self).__init__()

        # 扩散步投影
        self.diffusion_projection = nn.Linear(c_step, res_channels)
        # 时间注意力
        self.temporal_former =  TransformerEncoderWithoutEmbedding(e_layers, n_heads, res_channels, d_ff, dropout, activation, factor, distil)
        # 特征注意力
        self.feature_former =   TransformerEncoderWithoutEmbedding(e_layers, n_heads, res_channels, d_ff, dropout, activation, factor, distil)
        # 条件信息投影
        self.cond_projection = Conv1d(c_cond, 2*res_channels, 1)
        # 膨胀投影
        self.dilated_conv = Conv1d(res_channels, 2 * res_channels, 1)
        # 输出投影
        self.output_projection = Conv1d(res_channels, 2 * res_channels, 1)

    def forward_temporal(self, y, base_shape):
        B, C, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 2, 1, 3).reshape(B*K, C, L)
        y = self.temporal_former(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = y.reshape(B, K, C, L).permute(0, 2, 1, 3).reshape(B, C, K*L) 
        return y
    
    def forward_feature(self, y, base_shape):
        B, C, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 3, 1, 2).reshape(B*L, C, K)
        y = self.feature_former(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = y.reshape(B, L, C, K).permute(0, 2, 3, 1).reshape(B, C, K*L) 
        return y
    
    def forward(self, x,  diffusion_step, cond=None):
        B, C, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, C, K*L)
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1) 
        y = x + diffusion_step  
        y = self.forward_temporal(y, base_shape)
        y = self.forward_feature(y, base_shape)
        if self.cond_projection is None: 
            y = self.dilated_conv(y)
        else:
            cond = self.cond_projection(cond) 
            y = self.dilated_conv(y) + cond
        gate, filter = torch.chunk(y, 2, dim=1) 
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y) 
        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        
        return (x + residual) / math.sqrt(2.0), skip
    
    
class SSSD_Residual_block(nn.Module):
    def __init__(self, res_channels, skip_channels, 
                 diffusion_step_embed_dim_out, in_channels,
                s4_lmax,
                s4_d_state,
                s4_dropout,
                s4_bidirectional,
                s4_layernorm):
        super(SSSD_Residual_block, self).__init__()
        self.res_channels = res_channels

        # 时间步投影
        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)
        
        # 结构化状态转移层
        self.S41 = S4Layer(features=2*self.res_channels, 
                          lmax=s4_lmax,
                          N=s4_d_state,
                          dropout=s4_dropout,
                          bidirectional=s4_bidirectional,
                          layer_norm=s4_layernorm)
        # 膨胀投影
        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)

        self.S42 = S4Layer(features=2*self.res_channels, 
                          lmax=s4_lmax,
                          N=s4_d_state,
                          dropout=s4_dropout,
                          bidirectional=s4_bidirectional,
                          layer_norm=s4_layernorm)
        # 条件信息投影
        self.cond_conv = Conv(2*in_channels, 2*self.res_channels, kernel_size=1)  

        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        # 跳跃投影
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, input_data):
        x, cond, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.res_channels                      
                 
        part_t = self.fc_t(diffusion_step_embed)
        part_t = part_t.view([B, self.res_channels, 1])  
        h = h + part_t
        
        h = self.conv_layer(h)
        h = self.S41(h.permute(2,0,1)).permute(1,2,0)     
        
        assert cond is not None
        cond = self.cond_conv(cond)
        h += cond
        
        h = self.S42(h.permute(2,0,1)).permute(1,2,0)
        
        out = torch.tanh(h[:,:self.res_channels,:]) * torch.sigmoid(h[:,self.res_channels:,:])

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability


class SSSD_Residual_group(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers, 
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(SSSD_Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        # 扩散步投影
        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)
        
        # 残差块
        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(SSSD_Residual_block(res_channels, skip_channels, 
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       in_channels=in_channels,
                                                       s4_lmax=s4_lmax,
                                                       s4_d_state=s4_d_state,
                                                       s4_dropout=s4_dropout,
                                                       s4_bidirectional=s4_bidirectional,
                                                       s4_layernorm=s4_layernorm))

            
    def forward(self, input_data):

        noise, conditional, diffusion_steps = input_data

        # 扩散步嵌入
        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        # 扩散步投影
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        # 残差连接
        h = noise
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n]((h, conditional, diffusion_step_embed))  
            skip += skip_n  

        return skip * math.sqrt(1.0 / self.num_res_layers)  