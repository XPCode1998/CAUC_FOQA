from apps.ml_models.LGTDM.layers.PriITS_layers import *
from apps.ml_models.LGTDM.layers.generate_adj import *
from apps.ml_models.LGTDM.layers.Embed import DiffusionEmbedding


class Guide_diff(nn.Module):
    def __init__(self, inputdim, target_dim, is_itp, res_channels, diff_layers, diff_steps, step_emb_dim, c_cond, n_heads, proj_t, is_cross_t, is_cross_s, device, is_adp, adj_file):
        super().__init__()
        self.channels = res_channels
        self.is_itp = is_itp
        self.itp_channels = None
        if self.is_itp:
            self.itp_channels = res_channels
            self.itp_projection = Conv1d_with_init(inputdim-1, self.itp_channels, 1)

            self.itp_modeling = GuidanceConstruct(channels=self.itp_channels, nheads=n_heads, target_dim=target_dim,
                                            order=2, include_self=True, device=device, is_adp=is_adp,
                                            adj_file=adj_file, proj_t=proj_t)
            self.cond_projection = Conv1d_with_init(c_cond, self.itp_channels, 1)
            self.itp_projection2 = Conv1d_with_init(self.itp_channels, 1, 1)

        self.diffusion_embedding = DiffusionEmbedding(num_steps=diff_steps, embedding_dim=step_emb_dim, projection_dim=step_emb_dim)

        # if adj_file == 'KDD':
        #     self.adj = get_similarity(thr=0.1)
        # self.device = device
        # self.support = compute_support_gwn(self.adj, device=device)
        # self.is_adp = is_adp
        # if self.is_adp:
        #     node_num = self.adj.shape[0]
        #     self.nodevec1 = nn.Parameter(torch.randn(node_num, 10).to(self.device), requires_grad=True).to(self.device)
        #     self.nodevec2 = nn.Parameter(torch.randn(10, node_num).to(self.device), requires_grad=True).to(self.device)
        #     self.support.append([self.nodevec1, self.nodevec2])

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=c_cond,
                    channels=self.channels,
                    diffusion_embedding_dim=step_emb_dim,
                    nheads=n_heads,
                    target_dim=target_dim,
                    proj_t=proj_t,
                    is_adp=is_adp,
                    device=device,
                    adj_file=adj_file,
                    is_cross_t=is_cross_t,
                    is_cross_s=is_cross_s,
                )
                for _ in range(diff_layers)
            ]
        )


    def forward(self, x, side_info, t, itp_x=None, cond_mask=None):
        if self.is_itp:
            x = torch.cat([x, itp_x], dim=1)
            # print('guide')
        B, inputdim, K, L = x.shape
        # print('x shape', x.shape)

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        
        x = x.reshape(B, self.channels, K, L) # (B, C, K, L)
       
        if self.is_itp:
            itp_x = itp_x.reshape(B, inputdim-1, K * L)
            itp_x = self.itp_projection(itp_x)
            itp_cond_info = side_info.reshape(B, -1, K * L)
            itp_cond_info = self.cond_projection(itp_cond_info)
            itp_x = itp_x + itp_cond_info
            # itp_x = self.itp_modeling(itp_x, [B, self.itp_channels, K, L], None)
            itp_x = F.relu(itp_x)
            itp_x = itp_x.reshape(B, self.itp_channels, K, L)
        # print('itp_x shape', itp_x.shape)
        diffusion_emb = self.diffusion_embedding(t)

        skip = []
        for i in range(len(self.residual_layers)):
            x, skip_connection = self.residual_layers[i](x, side_info, diffusion_emb, itp_x, None)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class NoiseProject(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, target_dim, proj_t, order=2, include_self=True,
                 device=None, is_adp=False, adj_file=None, is_cross_t=False, is_cross_s=True):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.forward_time = TemporalLearning(channels=channels, nheads=nheads, is_cross=is_cross_t)
        # self.forward_feature = SpatialLearning(channels=channels, nheads=nheads, target_dim=target_dim,
        #                                        order=order, include_self=include_self, device=device, is_adp=is_adp,
        #                                        adj_file=adj_file, proj_t=proj_t, is_cross=is_cross_s)


    def forward(self, x, side_info, diffusion_emb, itp_info, support):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape, itp_info)
        # y = self.forward_feature(y, base_shape, support, itp_info)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, side_dim, _, _ = side_info.shape
        side_info = side_info.reshape(B, side_dim, K * L)
        side_info = self.cond_projection(side_info)  # (B,2*channel,K*L)
        y = y + side_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip

