import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from inspect import isfunction
from django.db.models import Count
from apps.ml_models.LGTDM.data_provider.data_factory import data_provider
from apps.core.models import QAR, QAR_Parameter_Attribute


def swish(x):
    return x * torch.sigmoid(x)


@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)


# 辅助函数
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d 


# 根据时间步t，计算出a(sqrt_alphas_cumprod等)对应t位置的值，返回为(batch,1,1,1)形状
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# 边缘噪声强度以余弦函数的方式增长
def cosine_beta_schedule(timesteps, s=0.008, beta_start=0.0001, beta_end=0.02):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


# 注入噪声的强度呈线性增长
def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def get_seq_dim(file_path):
    df_raw = pd.read_csv(file_path)
    if 'OT' in df_raw.columns:
        return len(df_raw.columns)-2
    else:
        return len(df_raw.columns)-1


def get_time_dim(embed, freq, enc_minute):
    time_dim_dict={'s':6, 't':5, 'h':4, 'd':3, 'b':3, 'w':2, 'm':1}
    if embed == 'timeF':
        time_dim = time_dim_dict[freq]
    else:
        if enc_minute:
            time_dim = 5
        else:
            time_dim = 4

    return time_dim


def get_variance_dimensions_count():
    """获取方差大于0.01的维度数量(排除qar_id, id和label属性)"""
    # 定义要排除的属性列表
    excluded_attributes = ['qar_id', 'id', 'label']
    
    # 从QARVariance模型中查询方差大于0.01且不在排除列表中的属性数量
    count = QAR_Parameter_Attribute.objects.filter(
        normalized_variance__gt=0.1
    ).exclude(
        parameter_name__in=excluded_attributes
    ).count()
    
    return count


def get_label_variety_count():
    """获取label的种类数"""
    # 使用annotate和values获取所有不同的label值
    label_counts = QAR.objects.values('label').annotate(count=Count('label')).order_by('label')
    return len(label_counts)


def get_data_info():
    seq_dim = get_variance_dimensions_count()
    num_label = get_label_variety_count()
    return seq_dim, num_label