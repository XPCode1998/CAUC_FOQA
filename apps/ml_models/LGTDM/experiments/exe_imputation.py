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
from apps.ml_models.LGTDM.utils.metrics import metric
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Exp_Imputation(Exp_Basic):

    def  __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

        
    # 初始化模型
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args, self.seq_dim, self.num_label, self.device).float()
        
        return model

    # 获取数据
    def _get_data(self, mode):
        data_set, data_loader = data_provider(self.args, mode)
        
        return data_set, data_loader
        

    # 损失函数
    def _select_criterion(self):
        
        if self.args.loss == 'l1':
            criterion = F.l1_loss
        elif self.args.loss == 'l2':
            criterion = F.mse_loss
        elif self.args.loss == 'huber':
            criterion = F.smooth_l1_loss
        else:
            raise NotImplementedError()
        
        return criterion
    

    # 优化器
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-6)
        
        return model_optim


    # 训练
    def train(self):

        # 训练集
        train_data, train_loader = self._get_data(mode='train')
        # 验证集
        val_data, val_loader = self._get_data(mode='val')
        # 优化器
        model_optim = self._select_optimizer()
        # 损失函数
        criterion = self._select_criterion()

        best_val_loss = float('inf')

        # 评价指标
        total_mae = []
        total_mse = []
        total_rmse = []
        total_mape = []
        total_mspe = []

        # 循环train_epochs次
        for epoch in range(1, self.args.train_epochs +1):
            # 单个epoch内的损失列表
            total_loss =[]

            total_d_loss = []
            total_g_loss = []

            total_c_loss = []
            
            # # 模型切换为训练模式
            self.model.train()
            with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, batch in enumerate(it, start=1):
                    # 数据形状 (B, L, K)
                    # data：含有缺失值的数据
                    # label：标签
                    # obs_mask：原始数据掩码
                    # gt_mask：手动掩码
                    data, label, obs_mask, gt_mask = [x.float().to(self.device) for x in batch]
                    
                    # 将label转换为torch.long类型
                    label = label.long().to(self.device)
                    
                    # 模型输入
                    model_input = (data, obs_mask, gt_mask)
                    
                    # 梯度清零
                    model_optim.zero_grad()
                    
                    # 计算损失
                    c_loss, d_loss, g_loss, loss =self.model('train', model_input, label, criterion, epoch)
                    total_loss.append(loss.item())
                    if c_loss:
                        total_c_loss.append(c_loss.item())
                    else:
                        total_c_loss.append(0)

                    total_d_loss.append(d_loss)
                    total_g_loss.append(g_loss)

                    # GAIN 的反向传播在模型内完成（ 涉及到生成器与判别器，更复杂）
                    if self.args.model != 'GAIN':
                        # 误差反向传播
                        loss.backward()
                        # 梯度更新
                        model_optim.step()

                    # 刷新tqdm中损失值显示
                    it.set_postfix(
                        ordered_dict={
                            'avg_epoch_loss': np.average(total_loss),
                            'c_loss': np.average(total_c_loss),
                            'd_loss': np.average(total_d_loss),
                            'g_loss': np.average(total_g_loss),
                            'epoch': epoch,
                        },
                        refresh=False,
                    )
            
            # 当前epoch的训练损失
            train_loss = np.average(total_loss)
            
            # 每val_per_epoch个epoch进行一次评估验证
            if epoch % self.args.val_per_epoch == 0:
                val_loss = self.val(val_data, val_loader, criterion, epoch)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # 保存模型参数
                    torch.save(self.model.state_dict(), 'apps/ml_models/LGTDM/save/model_weights.pth')
                    print(f'Best model saved at epoch {epoch} with validation loss: {best_val_loss:.4f}')
                
            # # test_epoch_start个epoch后，每test_per_epoch个epoch后进行一次测试
            # if epoch >= self.args.test_epoch_start and epoch % self.args.test_per_epoch ==0:
            #     mae, mse, rmse, mape, mspe = metric = self.test(epoch)
            #     # 评价指标列表
            #     total_mae.append(mae)
            #     total_mse.append(mse)
            #     total_rmse.append(rmse)
            #     total_mape.append(mape)
            #     total_mspe.append(mspe)

            
        return 


    # 验证
    def val(self, val_data, val_loader, criterion, epoch):
        
        # 验证集上的损失
        total_loss = []
        
        # 模型切换为评估模式
        self.model.eval()
        with torch.no_grad():
            with tqdm(val_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, batch in enumerate(it, start=1):
                    
                    # 数据形状 (B, L, K)
                    data, label, obs_mask, gt_mask = [x.float().to(self.device) for x in batch]
                    label = label.long().to(self.device)
                    model_input = (data, obs_mask, gt_mask)
                    
                    # 计算损失
                    loss =self.model('val', model_input, label, criterion)
                    total_loss.append(loss.item())

                    # 刷新tqdm中损失值显示
                    it.set_postfix(
                        ordered_dict={
                            'val_loss': np.average(total_loss),
                            'epoch': epoch,
                        },
                        refresh=False,
                    )
            
            # 验证集上的损失
            val_loss = np.average(total_loss)

        return val_loss


    # 测试
    def test(self, epoch=None):

        # 测试集
        test_data, test_loader = self._get_data(mode='test')

        # 评价指标
        total_mae = []
        total_mse = []
        total_rmse = []
        total_mape = []
        total_mspe = []

        i = 0
        # 模型切换为评估模式
        self.model.eval()
        with torch.no_grad():
            with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, batch in enumerate(it, start=1):
                   
                    # 数据形状 (B, L, K)
                    data, label, obs_mask, gt_mask = [x.float().to(self.device) for x in batch]

                    # 将label转换为torch.long类型
                    label = label.long().to(self.device)

                    model_input = (data, obs_mask, gt_mask)
                    

                    # 数据填充
                    if self.args.model == 'LGTDM':
                        data_imputation, samples, data_time_list = self.model('test', model_input, label)
                    else:
                        data_imputation = self.model('test', model_input, label)

                    # 评估点位
                    eval_point = obs_mask - gt_mask
                    eval_point_cpu = eval_point.bool().cpu()

                    # 评估点位的填充值与真实值
                    # 逆标准化, 还原为原始数据集的值
                    if self.args.inverse_transform:
                        preds = [test_data.inverse_transform(batch.cpu().numpy()) for batch in data_imputation]
                        preds = np.array(preds)
                        preds = preds[eval_point_cpu.numpy()]
                        trues = [test_data.inverse_transform(batch.cpu().numpy()) for batch in data]
                        trues = np.array(trues)
                        trues = trues[eval_point_cpu.numpy()]
                    else:
                        preds = np.array(data_imputation[eval_point_cpu].cpu())
                        trues = np.array(data[eval_point_cpu].cpu()) 

                    # 计算评价指标值
                    mae, mse, rmse, mape, mspe = metric(preds, trues)

                    total_mae.append(mae)
                    total_mse.append(mse)
                    total_rmse.append(rmse)
                    total_mape.append(mape)
                    total_mspe.append(mspe)

                    it.set_postfix(
                        ordered_dict={
                            'mae': np.average(total_mae),
                            'mse': np.average(total_mse),
                            'rmse': np.average(total_rmse),
                            'mape': np.average(total_mape),
                            'mspe': np.average(total_mspe),
                        },
                        refresh=False,
                    )

                     
        mae = np.average(total_mae)
        mse = np.average(total_mse)
        rmse = np.average(total_rmse)
        mape = np.average(total_mape)
        mspe = np.average(total_mspe)
        # 输出评估指标值
        print('mae: ', mae)   
        print('mse: ', mse)
        print('rmse: ', rmse)
        print('mape: ', mape)
        print('mspe: ', mspe)

        return mae, mse, rmse, mape, mspe
    

    def get_quantile(self,samples,q,dim=1):
        return torch.quantile(samples,q,dim=dim).cpu().numpy()
    


      
   
        
            
                

