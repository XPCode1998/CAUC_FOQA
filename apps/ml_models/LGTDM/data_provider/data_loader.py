import os
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import warnings
from concurrent.futures import ThreadPoolExecutor

from django.db import connection
from apps.core.models import QAR, QAR_Parameter_Attribute

warnings.filterwarnings("ignore")

class QARDataset(Dataset):
    def __init__(
        self, seq_len=36, split_len=240, mode="train", missing_ratio=0.1, 
        scale=True, random_seed=2025):

        # 模式
        assert mode in ["train", "val", "test"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[mode]
        self.mode = mode

        # 参数
        self.random_seed = random_seed
        self.seq_len = seq_len
        self.split_len = split_len
        self.scale = scale
        self.missing_ratio = missing_ratio

        # 数据容器
        self.data = []
        self.label = []
        self.obs_mask = []
        self.gt_mask = []

        # 读取数据
        self.__read_data__()


    def _read_batch(self, offset, batch_size):
        """读取单个批次数据的函数, 确保batch_size是300的整数倍"""
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM core_qar LIMIT {batch_size} OFFSET {offset}")
            columns = [col[0] for col in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)

    def _load_data_from_sql(self):
        """从SQL数据库并行加载数据, 每次读取量为300的整数倍"""
        print("Reading data from SQL database using multi-threading...")

        # 获取总记录数
        total_count = QAR.objects.count()
        print(f"总数据量: {total_count}")

        # 基础批次大小（300的整数倍）
        base_batch_size = 300
        # 可以根据性能调整倍数因子
        multiplier = 33  # 300 * 33 = 9900 ≈ 10,000（接近原代码的批次大小）
        batch_size = base_batch_size * multiplier  # 确保是300的整数倍

        # 计算需要的批次数量（向上取整）
        num_batches = (total_count + batch_size - 1) // batch_size

        # 调整最后一个批次的大小（确保不超过总记录数）
        last_batch_size = total_count % batch_size
        if last_batch_size > 0:
            # 调整最后一个批次为300的整数倍
            last_batch_size = ((last_batch_size + base_batch_size - 1) // base_batch_size) * base_batch_size

        # 使用线程池并行读取
        with ThreadPoolExecutor(max_workers=4) as executor:
            # 准备所有批次的offset和实际读取大小
            batch_params = []
            for i in range(num_batches):
                offset = i * batch_size
                current_batch_size = batch_size if i < num_batches - 1 else min(last_batch_size, batch_size)
                if current_batch_size > 0:  # 确保有数据才读取
                    batch_params.append((offset, current_batch_size))
            
            # 提交所有任务
            futures = [executor.submit(self._read_batch, offset, size) for offset, size in batch_params]
            
            # 收集结果
            df_list = [future.result() for future in futures]

        # 合并所有批次
        df_raw = pd.concat(df_list, ignore_index=True)
        print(f"数据库并行读取完毕，共读取 {len(df_raw)} 条记录")
        return df_raw

    def __read_data__(self):

        # 读取数据
        df_raw = self._load_data_from_sql()

        # 建立训练集、验证集和测试集的划分索引
        index_max = len(df_raw)
        index_list = [i for i in range(0, len(df_raw), self.split_len)]
        
        random.seed(self.random_seed)
        random.shuffle(index_list)
        
        num_train = int(len(index_list) * 0.7)
        num_val = int(len(index_list) * 0.2)
        num_test = len(index_list) - num_train - num_val
        border1s = [0, num_train, num_train + num_val]
        border2s = [num_train, num_train + num_val, len(index_list)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        index = index_list[border1:border2]
        
        # 移除非特征属性
        cols_data = list(df_raw.columns)
        cols_to_remove = ['id', 'qar_id', 'label']
        for col in cols_to_remove:
            if col in cols_data:
                cols_data.remove(col)
        df_data = df_raw[cols_data]

        # 数据标签
        label_list = df_raw['label'].values.tolist()

        select_fileds = QAR_Parameter_Attribute.objects.filter(
            normalized_variance__gt=0.1
        ).exclude(
            parameter_name__in=cols_to_remove
        )

        selected_features = [item.parameter_name for item in select_fileds]
        
        print(f"原始特征数: {len(cols_data)}, 筛选后特征数: {len(selected_features)}")
        print("筛选后的特征:", selected_features)
        
        # 更新数据只包含筛选后的特征
        df_data = df_data[selected_features]
        self.features = selected_features
        
        # 获取训练数据用于标准化
        train_index = index_list[border1s[0]:border2s[0]]
        train_data_list = [df_data.iloc[i:i + self.split_len] for i in train_index]
        train_data = pd.concat(train_data_list, ignore_index=True)
        
        # 数据掩码
        obs_mask = 1 - df_data.isnull().values
        mask = obs_mask.reshape(-1).copy()
        obs_indices = np.where(mask)[0].tolist()
        miss_indices = np.random.choice(
            obs_indices, (int)(len(obs_indices) * self.missing_ratio), replace=False
        )
        mask[miss_indices] = False
        gt_mask = mask.reshape(obs_mask.shape)

        # 标准化
        self.mean_values = train_data.mean(skipna=True).values
        self.std_values = train_data.std(skipna=True).values
        if self.scale:
            data = (df_data.fillna(0) - self.mean_values) / ((1e-8) + self.std_values)
            data = data.values * obs_mask
        else:
            data = df_data.fillna(0).values

        if self.mode == "train":
            self.data.extend([
                data[j:j + self.seq_len]
                for i in index
                for j in range(i, min(i + self.split_len, index_max) - self.seq_len + 1, 100)
            ])
            self.label.extend([
                label_list[j]
                for i in index
                for j in range(i, min(i + self.split_len, index_max) - self.seq_len + 1, 100)
            ])
            self.obs_mask.extend([
                obs_mask[j:j + self.seq_len]
                for i in index
                for j in range(i, min(i + self.split_len, index_max) - self.seq_len + 1, 100)
            ])
            self.gt_mask = self.obs_mask
        else:
            self.data.extend([
                data[j:j + self.seq_len]
                for i in index
                for j in range(i, min(i + self.split_len, index_max) - self.seq_len + 11, self.seq_len)
            ])
            self.label.extend([
                label_list[j]
                for i in index
                for j in range(i, min(i + self.split_len, index_max) - self.seq_len + 1, self.seq_len)
            ])
            self.obs_mask.extend([
                obs_mask[j:j + self.seq_len]
                for i in index
                for j in range(i, min(i + self.split_len, index_max) - self.seq_len + 1, self.seq_len)
            ])
            self.gt_mask.extend([
                gt_mask[j:j + self.seq_len]
                for i in index
                for j in range(i, min(i + self.split_len, index_max) - self.seq_len + 1, self.seq_len)
            ])

    def __getitem__(self, index):
        return (self.data[index], self.label[index], self.obs_mask[index], self.gt_mask[index])

    def __len__(self):
        return len(self.data)

    def inverse_transform(self, data):
        return data * (self.std_values + (1e-6)) + self.mean_values

    def get_scale(self):
        return self.mean_values, self.std_values

    def get_features(self):
        return self.features 

    def get_dim(self):
        return len(self.features)