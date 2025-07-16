import torch
from apps.ml_models.LGTDM.utils.tools import get_data_info
from apps.ml_models.LGTDM.model import CSDI, LGTDM, SSSD, BRITS, SAITS, GAIN, PriSTI


class Exp_Basic(object):
    def __init__(self, args):
        
        # 实验参数
        self.args = args
        
        # 模型字典
        self.model_dict = {
            'LGTDM' : LGTDM,
            'CSDI' : CSDI,
            'SSSD' : SSSD,
            'BRITS' : BRITS,
            'SAITS' : SAITS,
            'GAIN' : GAIN,
            'PriSTI': PriSTI
        }
        
        # 数据集文件路径字典
        file_path_dict = {
            'ETTh1':'./data/ETT-small/ETTh1.csv',
            'ETTh2':'./data/ETT-small/ETTh2.csv',
            'ETTm1':'./data/ETT-small/ETTm1.csv',
            'ETTm2':'./data/ETT-small/ETTm2.csv',
            'Electricity':'./data/electricity/electricity.csv',
            'KDD':'./data/pm25/Code/STMVL/SampleData/pm25_ground.txt',
            'Traffic':'./data/traffic/traffic.csv',
            'Weather':'./data/weather/weather.csv',
            'METR-LA':'./data/metr-la/METR-LA.csv',
            'PEMS-BAY':'./data/pems-bay/PEMS-BAY.csv',
            'FlightRisk':'./data/FlightRisk/'
        }
        
        # 数据维度
        self.seq_dim, self.num_label = get_data_info()
        # 运行设备
        self.device = self._acquire_device()
        # 模型构建
        self.model = self._build_model().to(self.device)


    # 构建模型
    def _build_model(self):
        raise NotImplementedError
        
        return None


    # 定义设备
    def _acquire_device(self):
        
        # 使用GPU
        if self.args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        # 使用CPU
        else:
            device = torch.device('cpu')
            print('Use CPU')
        
        return device


    def _get_data(self):
        pass


    def vali(self):
        pass


    def train(self):
        pass


    def test(self):
        pass
