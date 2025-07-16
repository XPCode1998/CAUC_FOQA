import torch
import datetime
from tabulate import tabulate
from apps.ml_models.LGTDM.experiments.exe_imputation import Exp_Imputation
from apps.ml_models.LGTDM.experiments.exp_median import Exp_Median
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
class LGTDMConfig:
    """Configuration class that holds all parameters with their default values"""
    def __init__(self):
        # Basic config
        self.is_training = 1
        self.model = "LGTDM"
        self.dataset = "QAR"
        self.missing_ratio = 0.1
        
        # Data loader
        self.dataset_type = "QARDataset"
        self.seq_len = 150
        self.split_len = 300
        self.scale = True
        
        # Flight risk data loader
        # self.sample_interval = 20
        # self.dim_auto_select = 1
        # self.dim_manual_select = 1
        # self.select_threshold = 0.001
        self.random_seed = 2025
        self.num_workers = 8
        
        # Model define - diff part
        self.diff_steps = 30
        self.diff_layers = 4
        
        # Diff noise part
        self.beta_start = 0.0001
        self.beta_end = 0.5
        self.beta_schedule = "quad"
        
        # Diff step part
        self.step_emb_dim = 64
        self.c_step = 128
        
        # Conditional part
        self.unconditional = False
        self.c_cond = 128
        self.pos_emb_dim = 0
        self.date_emb_dim = 128
        self.fea_emb_dim = 0
        self.label_emb_dim = 128
        self.is_label_emb = 1
        
        # Label classifier
        self.lcm_hidden_dim = 64
        
        # Input part
        self.inverted_channel = False
        
        # Residual part
        self.res_channels = 64
        self.skip_channels = 64
        self.dilation = 2
        self.d_hidden_dim = 64
        self.gan_loss_ratio = 0.1
        self.classifier_loss_ratio = 0.005
        
        # TAM part
        self.n_heads = 8
        self.e_layers = 1
        self.d_layers = 2
        self.d_model = 128
        self.d_ff = 256
        self.factor = 3
        self.distil = True
        self.dropout = 0.1
        self.activation = "gelu"
        
        # Tau part
        self.tau_kernel_size = 21
        self.tau_dilation = 2
        
        # Train part
        self.only_generate_missing = True
        self.train_missing_ratio_fixed = False
        
        # Test/val part
        self.val_per_epoch = 10
        self.test_per_epoch = 10
        self.test_epoch_start = 10
        self.n_samples = 50
        self.select_k = True
        self.inverse_transform = True
        self.incremental = True
        self.noised_unmasked = True
        
        # Resample part (deprecated)
        self.resample = False
        self.jump_length = 5
        self.jump_n_samples = 5
        
        # Optimization
        self.itr = 1
        self.train_epochs = 1
        self.batch_size = 4
        self.loss = "l2"
        self.learning_rate = 0.001
        self.lradj = "type1"
        self.patience = 3
        
        # ISM part
        self.cls_model = "GRU"
        self.cls_n_heads = 8
        self.cls_e_layers = 1
        self.cls_d_layers = 2
        self.cls_d_model = 128
        self.cls_d_ff = 256
        self.cls_factor = 3
        self.cls_distil = True
        self.cls_dropout = 0.1
        self.cls_activation = "gelu"
        self.cls_gru_hidden_size = 128
        self.cls_gru_num_layers = 4
        self.pretrain_missing_ratio_fixed = True
        self.pretrain_epochs = 2
        self.cls_begin_ratio = 0.02
        self.cls_end_ratio = 0.04
        
        # SSSD part
        self.s4_lmax = 100
        self.s4_d_state = 64
        self.s4_dropout = 0.0
        self.s4_bidirectional = 1
        self.s4_layernorm = 1
        
        # BRITS part
        self.brits_rnn_hid_size = 64
        
        # GAIN part
        self.gain_hint_rate = 0.9
        self.gain_alpha = 1
        
        # SAITS part
        self.saits_diagonal_attention_mask = True
        self.saits_n_groups = 2
        self.saits_n_group_inner_layers = 1
        self.saits_MIT = True
        self.saits_d_model = 256
        self.saits_d_inner = 128
        self.saits_n_head = 4
        self.saits_d_k = 64
        self.saits_d_v = 64
        self.saits_dropout = 0.1
        self.saits_input_with_mask = True
        self.saits_param_sharing_strategy = "inner_group"
        self.saits_imputation_loss_weight = 1
        self.saits_reconstruction_loss_weight = 1
        
        # PriSTI part
        self.is_lr_decay = 1
        self.is_adp = 1
        self.is_cross_t = 1
        self.is_cross_s = 1
        self.proj_t = 64
        self.use_guide = 1
        
        # GPU
        self.use_gpu = True
        self.gpu = 0
        
        # Runtime properties
        self.cur_time = datetime.datetime.now().strftime("%m-%d %H-%M-%S")

    def update(self, params):
        """Update configuration with custom parameters"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Handle GPU availability
        self.use_gpu = torch.cuda.is_available() and self.use_gpu

def run_lgtdm(is_training=None):
   
    # Initialize configuration
    config = LGTDMConfig()
    
    if is_training:
        config.is_training = int(is_training)
    
    # Print configuration
    print("LGTDM Configuration:")
    config_list = [[attr, getattr(config, attr)] for attr in dir(config) 
                  if not attr.startswith('_') and not callable(getattr(config, attr))]
    print(tabulate(config_list, headers=["Parameter", "Value"], tablefmt="grid"))
    
    # Execute main logic
    Exp = Exp_Imputation

    if config.is_training:
        for _ in range(config.itr):
            setting = f"{config.model}_{config.dataset}_{config.missing_ratio}"
            print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp = Exp(config)
            exp.train()
            torch.cuda.empty_cache()
    else:
        setting = f"{config.model}_{config.dataset}_{config.missing_ratio}"
        print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp = Exp(config)
        exp.imputate()
        torch.cuda.empty_cache()
    
    return {"status": "success", "time": config.cur_time}
    

# Maintain original command-line interface
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LGTDM")
    
    # Add all arguments (same as original)
    # ... [all the original argparse code] ...
    
    args = parser.parse_args()
    run_lgtdm(vars(args))