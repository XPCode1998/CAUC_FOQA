import torch
import torch.nn as nn
from apps.ml_models.LGTDM.layers.Transformer_EncDec import Encoder, EncoderLayer, ConvLayer, TransformerBlock
from apps.ml_models.LGTDM.layers.SelfAttention_Family import FullAttention, ProbAttention, FlowAttention, AttentionLayer, XPAttentionLayer, TemporalAttention
from apps.ml_models.LGTDM.layers.Embed import DataEmbedding, DataEmbedding_inverted


class iTemporalFormer(nn.Module):
    def __init__(self, seq_len, num_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq):
        super(iTemporalFormer, self).__init__()
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)
        self.encoder = TransformerBlock(XPAttentionLayer, n_heads, d_model, d_ff)
        self.projector = nn.Linear(d_model, seq_len, bias=True)

    def forward(self, x_enc, x_mark_enc=None):
        _, _, K, N = x_enc.shape
        enc_out  = self.enc_embedding(x_enc, x_mark_enc) 
        enc_out = self.encoder(enc_out)
        enc_out = self.projector(enc_out)[:, :, :K, :]
        return enc_out

class TemporalFormer(nn.Module):
    def __init__(self, enc_in, num_layers, n_heads, d_model, d_ff, dropout, activation, factor, distill, embed, freq):
        super(TemporalFormer, self).__init__()
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.encoder = TransformerBlock(XPAttentionLayer, n_heads, d_model, d_ff)
        self.projector = nn.Linear(d_model, enc_in, bias=True)

    def forward(self, x_enc, x_mark_enc=None):
        _, _, K, N = x_enc.shape
        enc_out  = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out)
        enc_out = self.projector(enc_out)
        return enc_out

class TransformerEncoder(nn.Module):
    def __init__(self, enc_in,  num_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq):
        super(TransformerEncoder, self).__init__()
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        attn = FullAttention(mask_flag=False, factor=factor, attention_dropout=dropout, output_attention = False)
        attn_layer = AttentionLayer(attn, d_model, n_heads)
        encoder_layer = EncoderLayer(attn_layer, d_model, d_ff, dropout, activation )
        self.encoder = Encoder(
            [encoder_layer for i in range(num_layers)],
            norm_layer=torch.nn.LayerNorm(d_model)
        ) 
        self.projector = nn.Linear(d_model, enc_in, bias=True)

    def forward(self,x_enc, x_mark_enc=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # (B, L, K)
        enc_out, attns = self.encoder(enc_out, attn_mask=None) # (B, L, D)
        enc_out = self.projector(enc_out) # (B, L, K)
        return enc_out


class iTransformer(nn.Module):
    def __init__(self, seq_len, num_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq):
        super(iTransformer, self).__init__()
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout=dropout)
        attn = FullAttention(mask_flag=False, factor=factor, attention_dropout=dropout, output_attention = False)
        attn_layer = AttentionLayer(attn, d_model, n_heads)
        encoder_layer = EncoderLayer(attn_layer, d_model, d_ff, dropout, activation )
        self.encoder = Encoder(
            [encoder_layer for i in range(num_layers)],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projector = nn.Linear(d_model, seq_len, bias=True)
        
    def forward(self,x_enc, x_mark_enc=None): 
        B, L, K = x_enc.shape
        enc_out  = self.enc_embedding(x_enc, x_mark_enc) # (B, L, K) -> (B, K1+K2, D)
        enc_out, attns = self.encoder(enc_out, attn_mask=None) # (B, K1+K2, D)
        enc_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :K] # (B, L, K)
        return enc_out


class InformerEncoder(nn.Module):
    def __init__(self, enc_in, num_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq):
        super(InformerEncoder, self).__init__()
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        attn = ProbAttention(factor=factor, attention_dropout=dropout)
        attn_layer = AttentionLayer(attn, d_model, n_heads)
        encoder_layer = EncoderLayer(attn_layer, d_model, d_ff, dropout, activation )
        self.encoder = Encoder(
            [encoder_layer for i in range(num_layers)],
            [ConvLayer(d_model) for i in range(num_layers - 1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projector = nn.Linear(d_model, enc_in, bias=True)
        
    def forward(self,x_enc, x_mark_enc=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # (B, L, K)
        enc_out, attns = self.encoder(enc_out, attn_mask=None) # (B, L, D)
        enc_out = self.projector(enc_out) # (B, L, K)
        return enc_out


class FlowformerEncoder(nn.Module):
    def __init__(self, enc_in, num_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq):
        super(FlowformerEncoder, self).__init__()
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        attn = FlowAttention(attention_dropout=dropout)
        attn_layer = AttentionLayer(attn, d_model, n_heads)
        encoder_layer = EncoderLayer(attn_layer, d_model, d_ff, dropout, activation )
        self.encoder = Encoder(
            [encoder_layer for i in range(num_layers)],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projector = nn.Linear(d_model, enc_in, bias=True)
        
    def forward(self,x_enc, x_mark_enc=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # (B, L, K)
        enc_out, attns = self.encoder(enc_out, attn_mask=None) # (B, L, D)
        enc_out = self.projector(enc_out) # (B, L, K)
        return enc_out


class TransformerEncoderWithoutEmbedding(nn.Module):
    def __init__(self, num_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil):
        super(TransformerEncoderWithoutEmbedding, self).__init__()
        attn = FullAttention(mask_flag=False, factor=factor, attention_dropout=dropout, output_attention = False)
        attn_layer = AttentionLayer(attn, d_model, n_heads)
        encoder_layer = EncoderLayer(attn_layer, d_model, d_ff, dropout, activation )
        self.encoder = Encoder(
            [encoder_layer for i in range(num_layers)],
            norm_layer=torch.nn.LayerNorm(d_model)
        ) 
        

    def forward(self,x_enc, x_mark_enc=None):
        enc_out, attns = self.encoder(x_enc, attn_mask=None) # (B, L, C)
        return enc_out
    




# class XP_TAU(nn.Module):
#     def __init__(self, res_channels, tau_kernel_size, tau_dilation, e_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq, attn_shortcut=True):
#         super(XP_TAU, self).__init__()
#         # 时序注意力
        
#         self.tau = TemporalAttention(res_channels, kernel_size=tau_kernel_size, dilation=tau_dilation)

#         t_attn = FullAttention(mask_flag=False, factor=factor, attention_dropout=dropout, output_attention = False)
#         f_attn = FullAttention(mask_flag=False, factor=factor, attention_dropout=dropout, output_attention = False)
#         self.temporal_attention = AttentionLayer(t_attn, res_channels, n_heads)
#         self.feature_attention = AttentionLayer(f_attn, res_channels, n_heads)
#         self.dropout = nn.Dropout(dropout)

#         # # 时间维度注意力
#         # self.temporal_former = TransformerEncoder(res_channels, e_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq)
#         # # 特征维度注意力
#         # self.feature_former =  TransformerEncoder(res_channels, e_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq)

#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         B, C, K, L = base_shape = x.shape
        
#         x = self.tau(x)
       
#         x_t = x.permute(0, 2, 1, 3).reshape(B*K, C, L).permute(0, 2, 1)
#         x_f = x.permute(0, 3, 1, 2).reshape(B*L, C, K).permute(0, 2, 1)

#         # (B*K, L, C)
#         new_x_t, attn = self.temporal_attention(
#             x_t, x_t, x_t,
#             attn_mask=attn_mask,
#             tau=tau, delta=delta
#         ) 
#         # (B*L, K, C)
#         new_x_f, attn = self.feature_attention(
#             x_f, x_f, x_f,
#             attn_mask=attn_mask,
#             tau=tau, delta=delta
#         )

#         new_x_t = self.dropout(new_x_t)
#         new_x_f = self.dropout(new_x_f)
#         new_x_t = new_x_t.permute(0, 2, 1).reshape(B, K, C, L).permute(0, 2, 1, 3)
#         new_x_f = new_x_f.permute(0, 2, 1).reshape(B, L, C, K).permute(0, 2, 3, 1)

#         x = x + new_x_t + new_x_f

#         return x


# class XP_TAU(nn.Module):
#     def __init__(self, res_channels, tau_kernel_size, tau_dilation, e_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq, attn_shortcut=True):
#         super(XP_TAU, self).__init__()
        
#         # 时序注意力
#         self.tau = TemporalAttention(res_channels, kernel_size=tau_kernel_size, dilation=tau_dilation)
#         # 时间维度注意力
#         self.temporal_former = TransformerEncoder(res_channels, e_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq)
#         # 特征维度注意力
#         self.feature_former =  TransformerEncoder(res_channels, e_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq)

#         self.conv = nn.Conv1d(2*res_channels, res_channels, 1)


#     def forward_temporal(self, y, base_shape):
#         B, C, K, L = base_shape
#         if L == 1:
#             return y
#         y = y.reshape(B, C, K, L).permute(0, 2, 1, 3).reshape(B*K, C, L)
#         y = self.temporal_former(y.permute(0, 2, 1)).permute(0, 2, 1)
#         y = y.reshape(B, K, C, L).permute(0, 2, 1, 3).reshape(B, C, K*L) 
#         return y
    

#     def forward_feature(self, y, base_shape):
#         B, C, K, L = base_shape
#         if K == 1:
#             return y
#         y = y.reshape(B, C, K, L).permute(0, 3, 1, 2).reshape(B*L, C, K)
#         y = self.feature_former(y.permute(0, 2, 1)).permute(0, 2, 1)
#         y = y.reshape(B, L, C, K).permute(0, 2, 3, 1).reshape(B, C, K*L) 
#         return y


#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         B, C, K, L = base_shape = x.shape
#         x = x.reshape(base_shape)
#         x = self.tau(x)
#         x = x.reshape(B, C, K*L)

#         x_tmeporal = self.forward_temporal(x, base_shape)
#         x_feature = self.forward_feature(x, base_shape)

#         x = torch.cat([x_tmeporal, x_feature], 1)
#         x = self.conv(x)

#         return x   


class XP_TAU(nn.Module):
    def __init__(self, res_channels, tau_kernel_size, tau_dilation, e_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq, attn_shortcut=True):
        super(XP_TAU, self).__init__()
        
        # 时序注意力
        self.tau = TemporalAttention(res_channels, kernel_size=tau_kernel_size, dilation=tau_dilation)
        # 时间维度注意力
        self.temporal_former = TransformerEncoder(res_channels, e_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq)
        # 特征维度注意力
        self.feature_former =  TransformerEncoder(res_channels, e_layers, n_heads, d_model, d_ff, dropout, activation, factor, distil, embed, freq)

        self.conv = nn.Conv1d(2*res_channels, res_channels, 1)


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


    def forward(self, x, attn_mask=None, tau=None, delta=None):
        B, C, K, L = base_shape = x.shape
        x = x.reshape(base_shape)
        x = self.tau(x)
        x = x.reshape(B, C, K*L)

        x = self.forward_temporal(x, base_shape)
        x = self.forward_feature(x, base_shape)

        # x = torch.cat([x_tmeporal, x_feature], 1)
        # x = self.conv(x)

        return x   
