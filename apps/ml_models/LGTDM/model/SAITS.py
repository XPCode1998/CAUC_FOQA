from apps.ml_models.LGTDM.modules.SAITS_modules import *
from apps.ml_models.LGTDM.utils.metrics import masked_mae_cal

# def __init__(self, args, seq_dim, time_dim, device) :

class Model(nn.Module):
    def __init__(self, args, seq_dim, num_label, device):
        super().__init__()

        self.device = device
        self.d_time = args.seq_len
        self.d_feature = seq_dim

        self.diagonal_attention_mask = args.saits_diagonal_attention_mask
        self.n_groups = args.saits_n_groups
        self.n_group_inner_layers = args.saits_n_group_inner_layers

        self.MIT = args.saits_MIT
        self.d_model = args.saits_d_model
        self.d_inner = args.saits_d_inner
        self.n_head = args.saits_n_head
        self.d_k = args.saits_d_k
        self.d_v = args.saits_d_v
        self.dropout = args.saits_dropout

        self.input_with_mask = args.saits_input_with_mask 
        self.actual_d_feature = self.d_feature * 2 if self.input_with_mask else self.d_feature
        self.param_sharing_strategy = args.saits_param_sharing_strategy

        self.attn_dropout = 0

        self.train_missing_ratio_fixed = args.train_missing_ratio_fixed,
        self.missing_ratio = args.missing_ratio

        self.imputation_loss_weight = args.saits_imputation_loss_weight
        self.reconstruction_loss_weight = args.saits_reconstruction_loss_weight
        
        
        if self.param_sharing_strategy == "between_group":
            # For between_group, only need to create 1 group and repeat n_groups times while forwarding
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(
                        self.d_time,
                        self.actual_d_feature,
                        self.d_model,
                        self.d_inner,
                        self.n_head,
                        self.d_k,
                        self.d_v,
                        self.dropout,
                        self.attn_dropout,
                        self.diagonal_attention_mask,
                        self.device,
                    )
                    for _ in range(self.n_group_inner_layers)
                ]
            )
            self.layer_stack_for_second_block = nn.ModuleList(
                [
                    EncoderLayer(
                        self.d_time,
                        self.actual_d_feature,
                        self.d_model,
                        self.d_inner,
                        self.n_head,
                        self.d_k,
                        self.d_v,
                        self.dropout,
                        self.attn_dropout,
                        self.diagonal_attention_mask,
                        self.device,
                    )
                    for _ in range(self.n_group_inner_layers)
                ]
            )
        else:  # then inner_group，inner_group is the way used in ALBERT
            # For inner_group, only need to create n_groups layers
            # and repeat n_group_inner_layers times in each group while forwarding
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(
                        self.d_time,
                        self.actual_d_feature,
                        self.d_model,
                        self.d_inner,
                        self.n_head,
                        self.d_k,
                        self.d_v,
                        self.dropout,
                        self.attn_dropout,
                        self.diagonal_attention_mask,
                        self.device,
                    )
                    for _ in range(self.n_group_inner_layers)
                ]
            )
            self.layer_stack_for_second_block = nn.ModuleList(
                [
                    EncoderLayer(
                        self.d_time,
                        self.actual_d_feature,
                        self.d_model,
                        self.d_inner,
                        self.n_head,
                        self.d_k,
                        self.d_v,
                        self.dropout,
                        self.attn_dropout,
                        self.diagonal_attention_mask,
                        self.device,
                    )
                    for _ in range(self.n_group_inner_layers)
                ]
            )

        self.dropout = nn.Dropout(p=self.dropout)
        self.position_enc = PositionalEncoding(self.d_model, n_position=self.d_time)
        # for the 1st block
        self.embedding_1 = nn.Linear(self.actual_d_feature, self.d_model)
        self.reduce_dim_z = nn.Linear(self.d_model, self.d_feature)
        # for the 2nd block
        self.embedding_2 = nn.Linear(self.actual_d_feature, self.d_model)
        self.reduce_dim_beta = nn.Linear(self.d_model, self.d_feature)
        self.reduce_dim_gamma = nn.Linear(self.d_feature, self.d_feature)
        # for the 3rd block
        self.weight_combine = nn.Linear(self.d_feature + self.d_time, self.d_feature)

    def impute(self, X, masks):
        # X, masks = inputs["X"], inputs["missing_mask"]
        # the first DMSA block
        input_X_for_first = torch.cat([X, masks], dim=2) if self.input_with_mask else X
        input_X_for_first = self.embedding_1(input_X_for_first)
        enc_output = self.dropout(
            self.position_enc(input_X_for_first)
        )  # namely term e in math algo
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_first_block:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_first_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = masks * X + (1 - masks) * X_tilde_1

        # the second DMSA block
        input_X_for_second = (
            torch.cat([X_prime, masks], dim=2) if self.input_with_mask else X_prime
        )
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(
            input_X_for_second
        )  # namely term alpha in math algo
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_second_block:
                    enc_output, attn_weights = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_second_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, attn_weights = encoder_layer(enc_output)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # the attention-weighted combination block
        attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in math algo
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

        combining_weights = F.sigmoid(
            self.weight_combine(torch.cat([masks, attn_weights], dim=2))
        )  # namely term eta
        # combine X_tilde_1 and X_tilde_2
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        # replace non-missing part with original data
        X_c = masks * X + (1 - masks) * X_tilde_3
        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]


    # 随机掩码
    def get_mask_rm(self, obs_mask, train_missing_ratio_fixed=False, missing_ratio = None):
        random_mask = torch.rand_like(obs_mask) * obs_mask
        random_mask = random_mask.reshape(len(random_mask), -1)
        for i in range(len(obs_mask)):
            if not train_missing_ratio_fixed:
                missing_ratio = np.random.rand()
            num_observed = obs_mask[i].sum().item()
            num_masked = round(num_observed * missing_ratio)
            random_mask[i][random_mask[i].topk(num_masked).indices] = -1
        random_mask = (random_mask > 0).reshape(obs_mask.shape).float()

        return random_mask




    # X: Incomplete Data
    # X_hat : Incomplete Data with Artificially Missing Values 
    def forward(self, mode, input_data, label, loss_fn=None):
        # inputs, stage
        # loss_fn = None
        # 数据形状变换
        data, obs_mask, gt_mask = [x for x in input_data]  # (B, L, K)
        X = data
        if mode == 'train':
            rm_mask = self.get_mask_rm(obs_mask, self.train_missing_ratio_fixed, self.missing_ratio)
            X_hat = data * rm_mask
            mask = rm_mask
        else:
            X_hat = data * gt_mask
            mask = gt_mask
        indicating_mask = obs_mask - mask

        
        # data, masks = inputs["X"], inputs["missing_mask"]
        reconstruction_loss = 0
        imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self.impute(X_hat, mask)

        reconstruction_loss += masked_mae_cal(X_tilde_1, X_hat, mask)
        reconstruction_loss += masked_mae_cal(X_tilde_2, X_hat, mask)
        final_reconstruction_MAE = masked_mae_cal(X_tilde_3, X_hat, mask)
        reconstruction_loss += final_reconstruction_MAE
        reconstruction_loss /= 3

        

        if (self.MIT or mode == "val") and mode != "test":
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(X_tilde_3, X, indicating_mask)
        else:
            imputation_MAE = torch.tensor(0.0)

        total_loss = imputation_MAE * self.imputation_loss_weight + reconstruction_loss * self.reconstruction_loss_weight

        if mode == 'train':
            return total_loss
        elif mode == 'val':
            return total_loss
        elif mode == 'test':
            return imputed_data