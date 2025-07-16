import torch
import torch.nn as nn
from apps.ml_models.LGTDM.modules import BRITS_modules


class Model(nn.Module):
    def __init__(self, args, seq_dim, num_label, device):
        super(Model, self).__init__()

        self.device = device

        self.seq_dim = seq_dim
        self.seq_len = args.seq_len
        self.rnn_hid_size = args.brits_rnn_hid_size
        
        self.build()


    def build(self):
        self.rits_f = BRITS_modules.Model(self.seq_len, self.seq_dim, self.rnn_hid_size, self.device)
        self.rits_b = BRITS_modules.Model(self.seq_len, self.seq_dim, self.rnn_hid_size, self.device)


    def parse_delta(self,mask):
        B, L, K = mask.size()
        delta = torch.ones_like(mask)
        for i in range(1, L):
            delta[:, i] = delta[:, i-1] * (1 - mask[:, i]) + 1
        return delta


    def set_forward_data(self, data, mask):
        delta = self.parse_delta(mask)
        return (data, mask, delta)
    

    def set_backward_data(self,data, mask):
        data = torch.flip(data, [1])
        mask = torch.flip(mask, [1])
        delta = self.parse_delta(mask)
        return (data, mask, delta)


    def forward(self, mode, input_data, label, loss_fn=None):
        data, obs_mask, gt_mask = [x for x in input_data]  # (B, L, K)
        if mode != 'test':
            mask = obs_mask
        else:
            mask = gt_mask
        forward_data = self.set_forward_data(data, mask) 
        backward_data = self.set_backward_data(data, mask)
        ret_f = self.rits_f(forward_data)
        ret_b = self.rits_b(backward_data)
        ret_b['imputations'] = torch.flip(ret_b['imputations'], [1])
        ret = self.merge_ret(ret_f, ret_b)
        if mode !=  'test':
            return ret['loss']
        else:
            return ret['imputations']


    # 返回值合并
    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])
        loss = loss_f + loss_b + loss_c
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2
        ret_f['loss'] = loss
        ret_f['imputations'] = imputations

        return ret_f


    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss


