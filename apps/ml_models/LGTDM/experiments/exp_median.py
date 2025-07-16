import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import os
from apps.ml_models.LGTDM.data_provider.data_factory import data_provider
from apps.ml_models.LGTDM.experiments.exe_basic import Exp_Basic


# do this before importing pylab or pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Exp_Median(Exp_Basic):

    def  __init__(self, args):
        super(Exp_Median, self).__init__(args)


    # 获取数据
    def _get_data(self, mode):
        data_set, data_loader = data_provider(self.args, mode)
        
        return data_set, data_loader
        
    # 测试
    def test(self, ):
      
        
        # 测试集
        test_data, test_loader = self._get_data(mode='test')
        
        # 评价指标
        total_mae = []
        total_mse = []
        total_rmse = []
        total_mape = []
        total_mspe = []

        i = 0
        count = 0
       
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, batch in enumerate(it, start=1):
                
                # 数据形状 (B, L, K)
                data, label, obs_mask, gt_mask = [x for x in batch]


                means, stds = test_data.get_scale()
                print(means)

                mask = obs_mask - gt_mask

                mae = (np.abs(data - means))*mask
                rmse = ((data-means)**2)*mask
                
                masked_mae_sum = torch.sum(mae)
                masked_rmse_sum = torch.sum(rmse)
                non_zero_elements = torch.sum(mask != 0)
                count += non_zero_elements
                mae_value = masked_mae_sum / non_zero_elements
                
                total_mae.append(mae_value)
                total_rmse.append(masked_rmse_sum)

        print(np.mean(total_mae))
        
        print(np.sqrt(np.sum(total_rmse)/count))

               
    
        
            
                

