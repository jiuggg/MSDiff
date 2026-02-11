import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class CAM_AE_multihops(nn.Module):
    """
    修改后的版本，支持多种融合机制，可通过 fusion_mode 参数控制。
    """

    def __init__(self, d_model, num_heads, num_layers, in_dims, emb_size, num_mashups_dim, hidden_dims,
                 use_2hop=True,
                 use_contextual_api_api_sim=True,
                 use_api_compl=True, use_mashup_text_sim=True,
                 norm=False, dropout=0.5,
                 # ================= 新增：接收融合模式参数 =================
                 fusion_mode='attention',
                 dim_inters=256,  # 新增：接收dim_inters参数
                 # =======================================================
                 softmax_temperature=1.0):
        super(CAM_AE_multihops, self).__init__()
        self.in_dims = in_dims
        self.num_mashups_dim = num_mashups_dim
        self.norm = norm
        self.num_layers = num_layers

        # ================= 新增：保存融合模式 =================
        self.fusion_mode = fusion_mode
        # ===================================================

        self.time_emb_dim = emb_size
        self.d_model = d_model
        self.dim_inters = dim_inters # 使用传入的参数

        self.use_2hop = use_2hop
        self.use_contextual_api_api_sim = use_contextual_api_api_sim
        self.use_api_compl = use_api_compl
        self.use_mashup_text_sim = use_mashup_text_sim

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        # --- 编码器定义 ---
        self.encoder_main_api_dim = nn.Linear(self.in_dims, self.dim_inters)
        self.encoder_2hop_mashup_dim = nn.Linear(self.num_mashups_dim, self.dim_inters)
        self.encoder_mashup_text_sim_mashup_dim = nn.Linear(self.num_mashups_dim, self.dim_inters)

        self.main_path_to_d_model_embedding = nn.Linear(self.dim_inters + self.time_emb_dim, self.d_model)

        context_branch_input_dim = self.dim_inters + self.time_emb_dim

        if self.use_2hop:
            self.second_hop_branch_specific_embedding = nn.Linear(context_branch_input_dim, self.d_model)
        else:
            self.second_hop_branch_specific_embedding = None
        if self.use_contextual_api_api_sim:
            self.contextual_api_api_sim_branch_specific_embedding = nn.Linear(context_branch_input_dim, self.d_model)
        else:
            self.contextual_api_api_sim_branch_specific_embedding = None
        if self.use_api_compl:
            self.api_compl_branch_specific_embedding = nn.Linear(context_branch_input_dim, self.d_model)
        else:
            self.api_compl_branch_specific_embedding = None
        if self.use_mashup_text_sim:
            self.mashup_text_sim_branch_specific_embedding = nn.Linear(context_branch_input_dim, self.d_model)
        else:
            self.mashup_text_sim_branch_specific_embedding = None

        self.self_attentions = nn.ModuleList([
            nn.MultiheadAttention(self.d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(self.num_layers)
        ])
        self.forward_layers = nn.ModuleList([
            nn.Linear(self.d_model, self.d_model)
            for _ in range(self.num_layers)
        ])

        self.drop_encoder_output = nn.Dropout(dropout)
        self.drop_attn_output = nn.Dropout(dropout)
        self.drop_ff_input = nn.Dropout(dropout)

        # ================= 根据融合模式，条件性地创建融合层 =================
        if self.fusion_mode == 'attention':
            self.fusion_attention = nn.MultiheadAttention(self.d_model, num_heads, dropout=dropout, batch_first=True)
            print("融合模式: 动态注意力 (Attention)")
        elif self.fusion_mode == 'concat':
            num_active_branches = sum([use_2hop, use_contextual_api_api_sim, use_api_compl, use_mashup_text_sim])
            if num_active_branches > 0:
                self.fusion_projection = nn.Linear(self.d_model * num_active_branches, self.d_model)
            else:
                self.fusion_projection = None
            print(f"融合模式: 拼接-线性 (Concat-Linear), 激活分支数: {num_active_branches}")
        elif self.fusion_mode == 'mean':
            print("融合模式: 平均池化 (Mean Pooling)")
        elif self.fusion_mode == 'max':
            print("融合模式: 最大池化 (Max Pooling)")
        # ================================================================

        self.projection_fused_to_main_dim = nn.Linear(self.d_model, self.dim_inters + self.time_emb_dim)

        # --- 深度MLP模块定义 ---
        mlp_input_dim = self.dim_inters + self.time_emb_dim
        mlp_layers = []
        current_dim = mlp_input_dim
        for h_dim in hidden_dims:
            mlp_layers.append(nn.Linear(current_dim, h_dim))
            mlp_layers.append(nn.Tanh())
            mlp_layers.append(nn.Dropout(dropout))
            current_dim = h_dim

        self.mid_layers = nn.Sequential(*mlp_layers)
        self.decoder = nn.Linear(current_dim, self.in_dims)

    def forward(self, x, x_mashup_2hop, x_contextual_api_api_sim, x_api_compl, x_mashup_text_sim, timesteps):
        # 编码和信息交互部分保持不变
        time_embedding = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        time_emb_processed = self.emb_layer(time_embedding)

        x_encoded = self.encoder_main_api_dim(x)
        if self.norm:
            x_encoded = F.normalize(x_encoded, dim=-1)
        x_encoded_dropped = self.drop_encoder_output(x_encoded)

        if x_encoded_dropped.ndim == 2:
            x_encoded_dropped = x_encoded_dropped.unsqueeze(1)

        num_items = x_encoded_dropped.size(1)
        time_emb_expanded = time_emb_processed.unsqueeze(1).expand(-1, num_items, -1)

        h_main_concat = torch.cat([x_encoded_dropped, time_emb_expanded], dim=-1)
        h_main_embedded_for_kv = self.main_path_to_d_model_embedding(h_main_concat)

        context_sources_configs = []
        if self.use_2hop and x_mashup_2hop is not None and self.second_hop_branch_specific_embedding is not None:
            context_sources_configs.append(
                {'name': '2hop', 'input_tensor': x_mashup_2hop, 'encoder_layer': self.encoder_2hop_mashup_dim,
                 'embed_layer': self.second_hop_branch_specific_embedding})
        if self.use_contextual_api_api_sim and x_contextual_api_api_sim is not None and self.contextual_api_api_sim_branch_specific_embedding is not None:
            context_sources_configs.append({'name': 'context_api_sim', 'input_tensor': x_contextual_api_api_sim,
                                            'encoder_layer': self.encoder_main_api_dim,
                                            'embed_layer': self.contextual_api_api_sim_branch_specific_embedding})
        if self.use_api_compl and x_api_compl is not None and self.api_compl_branch_specific_embedding is not None:
            context_sources_configs.append(
                {'name': 'api_compl', 'input_tensor': x_api_compl, 'encoder_layer': self.encoder_main_api_dim,
                 'embed_layer': self.api_compl_branch_specific_embedding})
        if self.use_mashup_text_sim and x_mashup_text_sim is not None and self.mashup_text_sim_branch_specific_embedding is not None:
            context_sources_configs.append({'name': 'mashup_text_sim', 'input_tensor': x_mashup_text_sim,
                                            'encoder_layer': self.encoder_mashup_text_sim_mashup_dim,
                                            'embed_layer': self.mashup_text_sim_branch_specific_embedding})

        active_branch_d_model_outputs = []
        active_branch_names_for_dynamic_weights = []

        if context_sources_configs:
            for source_config in context_sources_configs:
                ctx_input = source_config['input_tensor']
                ctx_encoder_layer = source_config['encoder_layer']
                ctx_branch_embed_layer = source_config['embed_layer']

                ctx_encoded = ctx_encoder_layer(ctx_input)
                if self.norm:
                    ctx_encoded = F.normalize(ctx_encoded, dim=-1)

                if ctx_encoded.ndim == 2:
                    ctx_encoded = ctx_encoded.unsqueeze(1)

                ctx_concat_with_time = torch.cat([ctx_encoded, time_emb_expanded], dim=-1)

                if ctx_branch_embed_layer is not None:
                    h_ctx_embedded_for_q = ctx_branch_embed_layer(ctx_concat_with_time)
                else:
                    h_ctx_embedded_for_q = torch.zeros_like(h_main_embedded_for_kv)

                current_branch_representation = h_ctx_embedded_for_q

                for i in range(self.num_layers):
                    attention_layer = self.self_attentions[i]
                    feed_forward_layer = self.forward_layers[i]

                    attention_output, _ = attention_layer(query=current_branch_representation,
                                                          key=h_main_embedded_for_kv, value=h_main_embedded_for_kv)

                    if self.norm:
                        attention_output = F.normalize(attention_output, dim=-1)
                    attention_output_dropped = self.drop_attn_output(attention_output)

                    current_branch_representation = current_branch_representation + attention_output_dropped

                    h_branch_ff_input_dropped = self.drop_ff_input(current_branch_representation)
                    ff_output = feed_forward_layer(h_branch_ff_input_dropped)

                    current_branch_representation = current_branch_representation + ff_output

                    if i != self.num_layers - 1:
                        current_branch_representation = torch.tanh(current_branch_representation)

                active_branch_d_model_outputs.append(current_branch_representation)
                active_branch_names_for_dynamic_weights.append(source_config['name'])

        fused_multihop_features_d_model = None
        dynamic_weights_for_return = None

        # ================== 核心修改：根据 fusion_mode 执行不同逻辑 ==================
        if active_branch_d_model_outputs:
            if self.fusion_mode == 'attention':
                query_for_fusion = h_main_embedded_for_kv
                B_shape, S_shape, D_shape = query_for_fusion.shape
                mha_query = query_for_fusion.reshape(B_shape * S_shape, 1, D_shape)
                stacked_key_value_sources = torch.stack(active_branch_d_model_outputs, dim=1)
                K_shape = stacked_key_value_sources.shape[1]
                mha_key_value = stacked_key_value_sources.permute(0, 2, 1, 3).reshape(B_shape * S_shape, K_shape,
                                                                                      D_shape)
                attn_output, attn_weights_raw = self.fusion_attention(mha_query, mha_key_value, mha_key_value)
                fused_multihop_features_d_model = attn_output.reshape(B_shape, S_shape, D_shape)

                if attn_weights_raw is not None:
                    dynamic_weights_for_return = attn_weights_raw.reshape(B_shape, S_shape, K_shape)

            elif self.fusion_mode == 'mean':
                stacked_features = torch.stack(active_branch_d_model_outputs, dim=1)
                fused_multihop_features_d_model = torch.mean(stacked_features, dim=1)

            elif self.fusion_mode == 'max':
                stacked_features = torch.stack(active_branch_d_model_outputs, dim=1)
                fused_multihop_features_d_model = torch.max(stacked_features, dim=1).values

            elif self.fusion_mode == 'concat':
                if self.fusion_projection is not None:
                    # 每个分支输出是 (B, S, D), 在最后一个维度拼接
                    concatenated_features = torch.cat(active_branch_d_model_outputs, dim=-1)
                    fused_multihop_features_d_model = self.fusion_projection(concatenated_features)
                else:  # 如果没有激活任何分支，则没有projection层，返回0
                    fused_multihop_features_d_model = torch.zeros_like(h_main_embedded_for_kv)

        # ==============================================================================

        if fused_multihop_features_d_model is None:
            fused_multihop_features_d_model = torch.zeros_like(h_main_embedded_for_kv)

        projected_fused_d_model = self.projection_fused_to_main_dim(fused_multihop_features_d_model)
        final_representation_before_mlp = h_main_concat + self.drop_attn_output(projected_fused_d_model)

        h = self.mid_layers(final_representation_before_mlp)
        output = self.decoder(h)

        if output.ndim == 3 and output.shape[1] == 1:
            output = output.squeeze(1)

        final_active_branch_names = active_branch_names_for_dynamic_weights if active_branch_d_model_outputs else []

        return output, dynamic_weights_for_return, final_active_branch_names


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding