import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse, remove_self_loops


def adj2adj(graph, batch_size, window_size, zdim):
    graph1 = graph.squeeze(0).squeeze(0).repeat(batch_size, window_size, 1, 1) \
        .reshape(-1, graph.shape[-2], graph.shape[-1])
    adj0, adj1, fea = [], [], []
    node_adj = dense_to_sparse(graph1)[0]
    node_efea = graph.unsqueeze(-1).repeat(1, 1, zdim)
    for num in range(node_adj.shape[1]):
        idx = torch.argwhere(node_adj[1] == num)
        idy = torch.argwhere(node_adj[0] == num)
        adj0.append(idx.repeat(1, idy.shape[0]).reshape(-1))
        adj1.append(idy.repeat(idx.shape[0], 1).reshape(-1))
        fea.append(torch.ones(
            idy.shape[0] * idx.shape[0], device=graph.device) * num)

    adj = torch.stack([torch.concat(adj0), torch.concat(adj1)], dim=0)
    fea = torch.concat(fea)
    edge_adj, edge_efea = remove_self_loops(adj, fea)
    return node_adj, node_efea, edge_adj, edge_efea


# =====================================================================
#  创新点1: 动态图结构学习器 (Dynamic Graph Structure Learning)
# =====================================================================

class DynamicGraphLearner(nn.Module):
    """基于多模态节点特征动态生成边权重（软注意力）。
    
    优化设计:
    - 不改变图的拓扑结构（始终使用静态图的边），只动态调整边权重
    - 这样避免了每次forward都需调用adj2adj_dynamic的开销
    - 通过可学习的λ参数控制动态权重与静态权重（全1）的融合
    """
    def __init__(self, node_dim, log_dim, hidden_dim=16, num_nodes=5):
        super(DynamicGraphLearner, self).__init__()
        input_dim = node_dim + log_dim
        self.num_nodes = num_nodes
        
        # 源/目标节点投影
        self.proj_src = nn.Linear(input_dim, hidden_dim)
        self.proj_tgt = nn.Linear(input_dim, hidden_dim)
        
        # 可学习的融合权重 λ，初始化为 0（sigmoid(0)=0.5）
        self.graph_lambda = nn.Parameter(torch.zeros(1))
        
        # 温度参数
        self.temperature = 0.5
        
    def forward(self, x_node, x_log, static_graph):
        """
        Args:
            x_node: [B, W, N, F_node]
            x_log:  [B, W, N, F_log]
            static_graph: [N, N] 静态邻接矩阵
        Returns:
            edge_weights: [N, N] 动态边权重矩阵（与静态图同尺寸）
            graph_reg_loss: 正则化损失
        """
        # 将 node 和 log 特征拼接后取全局均值
        h = torch.cat([x_node, x_log], dim=-1).mean(dim=1).mean(dim=0)  # [N, input_dim]
        
        src = self.proj_src(h)  # [N, hidden]
        tgt = self.proj_tgt(h)  # [N, hidden]
        
        # 相似度
        sim = torch.mm(src, tgt.t()) / (src.shape[-1] ** 0.5)  # [N, N]
        
        # Sigmoid得到连续权重（不用Gumbel采样，节省计算）
        if self.training:
            noise = torch.zeros_like(sim).uniform_(1e-6, 1 - 1e-6)
            noise = torch.log(noise) - torch.log(1 - noise)
            adj_dynamic = torch.sigmoid((sim + noise) / self.temperature)
        else:
            adj_dynamic = torch.sigmoid(sim / self.temperature)
        
        # 去除自环
        adj_dynamic = adj_dynamic * (1 - torch.eye(self.num_nodes, device=sim.device))
        
        # 只在静态图有边的位置生成权重，其他位置置零
        edge_weights = adj_dynamic * static_graph.float()
        
        # 可学习 λ 融合：lambda * dynamic_weight + (1-lambda) * static_weight(1.0)
        lam = torch.sigmoid(self.graph_lambda)
        fused_weights = lam * edge_weights + (1 - lam) * static_graph.float()
        
        # 正则化损失（轻量计算）
        sparse_loss = edge_weights.mean()
        static_flat = static_graph.float().flatten().clamp(1e-6, 1 - 1e-6)
        dynamic_flat = edge_weights.flatten().clamp(1e-6, 1 - 1e-6)
        kl_loss = F.kl_div(dynamic_flat.log(), static_flat, reduction='batchmean')
        graph_reg_loss = sparse_loss + 0.1 * kl_loss
        
        return fused_weights, graph_reg_loss


# =====================================================================
#  创新点2: 门控跨模态融合 (Gated Cross-Modal Fusion)
# =====================================================================

class GatedCrossModalFusion(nn.Module):
    """自适应门控机制实现跨模态注意力融合。
    
    替代原始 Temporal_Attention 中的固定 trace2pod 矩阵乘法。
    使用全局均值池化做跨模态信息聚合（不依赖具体的 N/E 数量），
    然后通过可学习的门控网络动态决定各模态注意力的融合权重。
    """
    def __init__(self, num_nodes, num_edges, window_size, num_heads):
        super(GatedCrossModalFusion, self).__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        
        # 注意力权重维度: [heads, W, W]
        att_dim = num_heads * window_size * window_size
        
        # 门控网络：输入 = [self_att; cross_modal_mean]，输出 = gate
        self.gate_node = nn.Sequential(
            nn.Linear(att_dim * 2, att_dim),
            nn.Sigmoid()
        )
        self.gate_edge = nn.Sequential(
            nn.Linear(att_dim * 2, att_dim),
            nn.Sigmoid()
        )
        self.gate_log = nn.Sequential(
            nn.Linear(att_dim * 2, att_dim),
            nn.Sigmoid()
        )
        
        # 跨模态信息投影
        self.cross_proj_node = nn.Linear(att_dim, att_dim)
        self.cross_proj_edge = nn.Linear(att_dim, att_dim)
        self.cross_proj_log = nn.Linear(att_dim, att_dim)
        
    def forward(self, att_n, att_t, att_l):
        """
        Args:
            att_n: [B, N, heads, W, W]
            att_t: [B, E, heads, W, W]
            att_l: [B, N, heads, W, W]
        Returns:
            fused_node_att, fused_edge_att, fused_log_att
        """
        B = att_n.shape[0]
        N = att_n.shape[1]
        E = att_t.shape[1]
        
        att_n_flat = att_n.reshape(B, N, -1)
        att_t_flat = att_t.reshape(B, E, -1)
        att_l_flat = att_l.reshape(B, N, -1)
        
        # 全局均值池化
        att_n_global = att_n_flat.mean(dim=1)
        att_t_global = att_t_flat.mean(dim=1)
        att_l_global = att_l_flat.mean(dim=1)
        
        # Node融合
        cross_for_node = self.cross_proj_node((att_t_global + att_l_global) / 2)
        cross_for_node = cross_for_node.unsqueeze(1).expand_as(att_n_flat)
        gate_n = self.gate_node(torch.cat([att_n_flat, cross_for_node], dim=-1))
        fused_n = gate_n * att_n_flat + (1 - gate_n) * cross_for_node
        
        # Edge融合
        cross_for_edge = self.cross_proj_edge((att_n_global + att_l_global) / 2)
        cross_for_edge = cross_for_edge.unsqueeze(1).expand_as(att_t_flat)
        gate_e = self.gate_edge(torch.cat([att_t_flat, cross_for_edge], dim=-1))
        fused_e = gate_e * att_t_flat + (1 - gate_e) * cross_for_edge
        
        # Log融合
        cross_for_log = self.cross_proj_log((att_n_global + att_t_global) / 2)
        cross_for_log = cross_for_log.unsqueeze(1).expand_as(att_l_flat)
        gate_l = self.gate_log(torch.cat([att_l_flat, cross_for_log], dim=-1))
        fused_l = gate_l * att_l_flat + (1 - gate_l) * cross_for_log
        
        fused_node_att = fused_n.reshape(att_n.shape)
        fused_edge_att = fused_e.reshape(att_t.shape)
        fused_log_att = fused_l.reshape(att_l.shape)
        
        return fused_node_att, fused_edge_att, fused_log_att


# =====================================================================
#  创新点3: 对比学习损失 (Contrastive Learning Loss)
# =====================================================================

class ContrastiveLoss(nn.Module):
    """基于节点表示的多模态对比学习损失。"""
    def __init__(self, node_dim, edge_dim, log_dim, proj_dim=32, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
        self.proj_node = nn.Sequential(
            nn.Linear(node_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.proj_edge = nn.Sequential(
            nn.Linear(edge_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.proj_log = nn.Sequential(
            nn.Linear(log_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
    def info_nce_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        B = z1.shape[0]
        logits = torch.mm(z1, z2.t()) / self.temperature
        labels = torch.arange(B, device=z1.device)
        return F.cross_entropy(logits, labels)
    
    def forward(self, enc_node, enc_edge, enc_log, dec_node, dec_edge, dec_log):
        enc_n = self.proj_node(enc_node.mean(dim=(1,2)))
        dec_n = self.proj_node(dec_node.mean(dim=(1,2)))
        enc_e = self.proj_edge(enc_edge.mean(dim=(1,2)))
        dec_e = self.proj_edge(dec_edge.mean(dim=(1,2)))
        enc_l = self.proj_log(enc_log.mean(dim=(1,2)))
        dec_l = self.proj_log(dec_log.mean(dim=(1,2)))
        
        loss_node = self.info_nce_loss(enc_n, dec_n)
        loss_edge = self.info_nce_loss(enc_e, dec_e)
        loss_log = self.info_nce_loss(enc_l, dec_l)
        
        return (loss_node + loss_edge + loss_log) / 3


# =====================================================================
#  基础模块
# =====================================================================

class FFN(nn.Module):
    def __init__(self, node_embedding_dim, edge_embedding_dim, log_embedding_dim, dropout=0.1):
        super(FFN, self).__init__()
        self.ffn_node = nn.Sequential(nn.Linear(node_embedding_dim, node_embedding_dim * 4),
                                      nn.LeakyReLU(inplace=True),
                                      nn.Linear(node_embedding_dim * 4, node_embedding_dim))

        self.ffn_edge = nn.Sequential(nn.Linear(edge_embedding_dim, edge_embedding_dim * 4),
                                      nn.LeakyReLU(inplace=True),
                                      nn.Linear(edge_embedding_dim * 4, edge_embedding_dim))

        self.ffn_log = nn.Sequential(nn.Linear(log_embedding_dim, log_embedding_dim * 4),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(log_embedding_dim * 4, log_embedding_dim))

        self.add_node = AddNorm(node_embedding_dim, dropout)
        self.add_edge = AddNorm(edge_embedding_dim, dropout)
        self.add_log = AddNorm(log_embedding_dim, dropout)

    def forward(self, x_node, x_trace, x_log):
        x_node = self.add_node(x_node, self.ffn_node(x_node))
        x_trace = self.add_edge(x_trace, self.ffn_edge(x_trace))
        x_log = self.add_log(x_log, self.ffn_log(x_log))
        return x_node, x_trace, x_log


class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, x, y):
        return self.ln(self.dropout(y) + x)


class AddALL(nn.Module):
    def __init__(self, node_embedding_dim, edge_embedding_dim, log_embedding_dim, dropout=0.1):
        super(AddALL, self).__init__()
        self.add_node = AddNorm(node_embedding_dim, dropout)
        self.add_edge = AddNorm(edge_embedding_dim, dropout)
        self.add_log = AddNorm(log_embedding_dim, dropout)

    def forward(self, x_node, x_trace, x_log, y_node, y_trace, y_log):
        x_node = self.add_node(x_node, y_node)
        x_trace = self.add_edge(x_trace, y_trace)
        x_log = self.add_log(x_log, y_log)
        return x_node, x_trace, x_log


class Temporal_Attention(nn.Module):
    def __init__(self, node_embedding_dim, edge_embedding_dim, log_embedding_dim, trace2pod,
                 heads_node=4, heads_edge=4, heads_log=4, dropout=0.1,
                 window_size=16, batch_size=10, num_nodes=5, num_edges=None):
        super(Temporal_Attention, self).__init__()
        self.window_size = window_size
        self.batch_size = batch_size
        self.trace2pod = trace2pod
        
        self.attention_node = nn.MultiheadAttention(embed_dim=node_embedding_dim, num_heads=heads_node,
                                                    dropout=dropout, batch_first=True)
        self.attention_trace = nn.MultiheadAttention(embed_dim=edge_embedding_dim, num_heads=heads_edge,
                                                     dropout=dropout, batch_first=True)
        self.attention_log = nn.MultiheadAttention(embed_dim=log_embedding_dim, num_heads=heads_log,
                                                     dropout=dropout, batch_first=True)

        self.vff_node = nn.Linear(node_embedding_dim, node_embedding_dim)
        self.vff_trace = nn.Linear(edge_embedding_dim, edge_embedding_dim)
        self.vff_log = nn.Linear(log_embedding_dim, log_embedding_dim)
        self.headff_node = nn.Linear(heads_node * window_size, window_size)
        self.headff_trace = nn.Linear(heads_edge * window_size, window_size)
        self.headff_log = nn.Linear(heads_log * window_size, window_size)

        self.softmax = nn.Softmax(dim=-1)
        
        # === 创新点2: 门控跨模态融合 ===
        if num_edges is None:
            num_edges = trace2pod.shape[0]
        self.gated_fusion = GatedCrossModalFusion(
            num_nodes=num_nodes,
            num_edges=num_edges,
            window_size=window_size,
            num_heads=heads_node
        )
        if heads_edge != heads_node:
            self.edge_head_proj = nn.Linear(heads_edge, heads_node)
        else:
            self.edge_head_proj = None
        if heads_log != heads_node:
            self.log_head_proj = nn.Linear(heads_log, heads_node)
        else:
            self.log_head_proj = None

    def forward(self, x_node, x_trace, x_log, mask=False):
        x_node = x_node.permute(0, 2, 1, 3).reshape(-1, self.window_size, x_node.shape[-1])
        x_trace = x_trace.permute(0, 2, 1, 3).reshape(-1, self.window_size, x_trace.shape[-1])
        x_log = x_log.permute(0, 2, 1, 3).reshape(-1, self.window_size, x_log.shape[-1])

        if mask:
            mask_att = (torch.triu(torch.ones(self.window_size, self.window_size, device=x_node.device)) == 1).transpose(0, 1)
            mask_att = mask_att.float().masked_fill(mask_att == 0, float('-inf')).masked_fill(mask_att == 1, float(0.0))

            att_n = self.attention_node(x_node, x_node, x_node, attn_mask=mask_att, average_attn_weights=False)[1]
            att_t = self.attention_trace(x_trace, x_trace, x_trace, attn_mask=mask_att, average_attn_weights=False)[1]
            att_l = self.attention_log(x_log, x_log, x_log, attn_mask=mask_att, average_attn_weights=False)[1]
        else:
            att_n = self.attention_node(x_node, x_node, x_node, average_attn_weights=False)[1]
            att_t = self.attention_trace(x_trace, x_trace, x_trace, average_attn_weights=False)[1]
            att_l = self.attention_log(x_log, x_log, x_log, average_attn_weights=False)[1]

        # reshape to [B, entity, heads, W, W]
        att_n = att_n.reshape(self.batch_size, -1, att_n.shape[-3], att_n.shape[-2], att_n.shape[-1])
        att_t = att_t.reshape(self.batch_size, -1, att_t.shape[-3], att_t.shape[-2], att_t.shape[-1])
        att_l = att_l.reshape(self.batch_size, -1, att_l.shape[-3], att_l.shape[-2], att_l.shape[-1])
        
        # 对齐 heads 维度
        att_t_aligned = att_t
        if self.edge_head_proj is not None:
            att_t_aligned = self.edge_head_proj(att_t.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        att_l_aligned = att_l
        if self.log_head_proj is not None:
            att_l_aligned = self.log_head_proj(att_l.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        
        # 门控融合
        fused_n, fused_t, fused_l = self.gated_fusion(att_n, att_t_aligned, att_l_aligned)
        
        att_node = fused_n
        att_edge = fused_t
        if self.edge_head_proj is not None:
            att_edge = att_t
        att_log = fused_l

        x_node = torch.bmm(
            self.softmax(att_node + att_n).reshape(att_n.shape[0] * att_n.shape[1], att_n.shape[2] * att_n.shape[3],
                                              att_n.shape[-1]), self.vff_node(x_node))
        x_trace = torch.bmm(
            self.softmax(att_edge + att_t).reshape(att_t.shape[0] * att_t.shape[1], att_t.shape[2] * att_t.shape[3],
                                              att_t.shape[-1]), self.vff_trace(x_trace))
        x_log = torch.bmm(
            self.softmax(att_log + att_l).reshape(att_l.shape[0] * att_l.shape[1], att_l.shape[2] * att_l.shape[3],
                                              att_l.shape[-1]), self.vff_log(x_log))
    
        x_node = self.headff_node(x_node.permute(0, 2, 1)).permute(0, 2, 1) \
            .reshape(self.batch_size, -1, self.window_size, x_node.shape[-1]).permute(0, 2, 1, 3)
        x_trace = self.headff_trace(x_trace.permute(0, 2, 1)).permute(0, 2, 1) \
            .reshape(self.batch_size, -1, self.window_size, x_trace.shape[-1]).permute(0, 2, 1, 3)
        x_log = self.headff_log(x_log.permute(0, 2, 1)).permute(0, 2, 1) \
            .reshape(self.batch_size, -1, self.window_size, x_log.shape[-1]).permute(0, 2, 1, 3)
        return x_node, x_trace, x_log


class Spatial_Attention(nn.Module):
    """空间注意力 —— 支持动态边权重。"""
    def __init__(self, node_embedding_dim, edge_embedding_dim, log_embedding_dim, heads_n2e=4, heads_e2n=4, dropout=0.1, batch_size=10,
                 window_size=16):
        super(Spatial_Attention, self).__init__()

        self.batch_size = batch_size
        self.window_size = window_size

        self.node2node = GATv2Conv(in_channels=node_embedding_dim + log_embedding_dim,
                                   out_channels=int((node_embedding_dim + log_embedding_dim) / heads_n2e),
                                   heads=heads_n2e, dropout=dropout, edge_dim=edge_embedding_dim, add_self_loops=False)
        self.egde2node = GATv2Conv(in_channels=edge_embedding_dim, out_channels=int(edge_embedding_dim / heads_e2n),
                                   heads=heads_e2n, dropout=dropout, edge_dim=node_embedding_dim + log_embedding_dim,
                                   add_self_loops=False)

    def forward(self, x_node, x_trace, x_log, node_adj, edge_adj, edge_efea):
        node = torch.concat([x_node, x_log], dim=-1)
        node = node.reshape(-1, node.shape[-1])
        x_trace = x_trace.reshape(-1, x_trace.shape[-1])
        
        node = self.node2node(node, node_adj, x_trace)
        x_trace = self.egde2node(x_trace, edge_adj, node[edge_efea.long()])

        x_node = node[:, :x_node.shape[-1]].reshape(self.batch_size, self.window_size, -1, x_node.shape[-1])
        x_trace = x_trace.reshape(self.batch_size, self.window_size, -1, x_trace.shape[-1])
        x_log = node[:, x_node.shape[-1]:].reshape(self.batch_size, self.window_size, -1, x_log.shape[-1])
        return x_node, x_trace, x_log


class Encoder_Decoder_Attention(nn.Module):
    def __init__(self, node_embedding_dim, edge_embedding_dim, log_embedding_dim, heads_node=4, heads_edge=4, heads_log=4, dropout=0.1):
        super(Encoder_Decoder_Attention, self).__init__()
        self.attention_node = nn.MultiheadAttention(
            embed_dim=node_embedding_dim, num_heads=heads_node, batch_first=True, dropout=dropout)
        self.attention_trace = nn.MultiheadAttention(
            embed_dim=edge_embedding_dim, num_heads=heads_edge, batch_first=True, dropout=dropout)
        self.attention_log = nn.MultiheadAttention(
            embed_dim=log_embedding_dim, num_heads=heads_log, batch_first=True, dropout=dropout)

    def forward(self, x_node, x_trace, x_log, z_node, z_trace, z_log):
        # 保存原始形状用于恢复
        node_shape = x_node.shape
        trace_shape = x_trace.shape
        log_shape = x_log.shape
        
        x_node = x_node.reshape(x_node.shape[0], -1, x_node.shape[-1])
        z_node = z_node.reshape(z_node.shape[0], -1, z_node.shape[-1])
        x_node = self.attention_node(x_node, z_node, z_node)[0]

        x_trace = x_trace.reshape(x_trace.shape[0], -1, x_trace.shape[-1])
        z_trace = z_trace.reshape(z_trace.shape[0], -1, z_trace.shape[-1])
        x_trace = self.attention_trace(x_trace, z_trace, z_trace)[0]

        x_log = x_log.reshape(x_log.shape[0], -1, x_log.shape[-1])
        z_log = z_log.reshape(z_log.shape[0], -1, z_log.shape[-1])
        x_log = self.attention_log(x_log, z_log, z_log)[0]

        # 恢复原始4D形状
        x_node = x_node.reshape(node_shape)
        x_trace = x_trace.reshape(trace_shape)
        x_log = x_log.reshape(log_shape)

        return x_node, x_trace, x_log


class Encoder(nn.Module):
    """编码器 —— 使用静态图拓扑 + 动态边权重（性能优化版）。"""
    def __init__(self, graph, node_embedding, edge_embedding, log_embedding, node_heads, log_heads, edge_heads,
                 n2e_heads, e2n_heads, dropout, batch_size, window_size, num_layer, trace2pod,
                 graph_hidden=16, num_nodes=5):
        super(Encoder, self).__init__()
        # 始终使用静态图的拓扑结构（只在init时计算一次）
        self.node_adj, self.node_efea, self.edge_adj, self.edge_efea = adj2adj(graph, batch_size, window_size, edge_embedding)
        self.graph = graph
        self.L = num_layer
        self.batch_size = batch_size
        self.window_size = window_size
        self.edge_embedding = edge_embedding

        self.spatial_attention = nn.ModuleList(
            [Spatial_Attention(node_embedding, edge_embedding, log_embedding,
                               heads_n2e=n2e_heads, heads_e2n=e2n_heads, dropout=dropout, batch_size=batch_size,
                               window_size=window_size) for _ in range(self.L)])
        self.sa_add = nn.ModuleList([AddALL(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])
        
        num_edges = trace2pod.shape[0]
        self.temporal_attention = nn.ModuleList(
            [Temporal_Attention(node_embedding, edge_embedding, log_embedding, trace2pod,
                                heads_node=node_heads, heads_edge=edge_heads, heads_log=log_heads, dropout=dropout, window_size=window_size,
                                batch_size=batch_size, num_nodes=num_nodes, num_edges=num_edges) for _ in range(self.L)])
        self.ta_add = nn.ModuleList([AddALL(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])
        self.ffn = nn.ModuleList([FFN(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])


    def forward(self, e_node, e_edge, e_log):
        # 使用静态图的拓扑结构筛选边特征（只做一次masked_select）
        e_edge = torch.masked_select(e_edge, self.node_efea.byte()) \
            .reshape(e_edge.shape[0], e_edge.shape[1], -1, e_edge.shape[-1])

        for i in range(self.L):
            e_node, e_edge, e_log = self.sa_add[i](e_node, e_edge, e_log, *self.spatial_attention[i](e_node, e_edge, e_log, self.node_adj, self.edge_adj, self.edge_efea))
            e_node, e_edge, e_log = self.ta_add[i](e_node, e_edge, e_log, *self.temporal_attention[i](e_node, e_edge, e_log))
            e_node, e_edge, e_log = self.ffn[i](e_node, e_edge, e_log)
        return e_node, e_edge, e_log


class Decoder(nn.Module):
    """解码器 —— 使用静态图拓扑 + 动态边权重（性能优化版）。"""
    def __init__(self, graph, node_embedding, edge_embedding, log_embedding, node_heads, log_heads, edge_heads,
                 n2e_heads, e2n_heads, dropout, batch_size, window_size, num_layer, trace2pod,
                 graph_hidden=16, num_nodes=5):
        super(Decoder, self).__init__()
        self.node_adj, self.node_efea, self.edge_adj, self.edge_efea = adj2adj(graph, batch_size, window_size, edge_embedding)
        self.graph = graph
        self.L = num_layer
        self.batch_size = batch_size
        self.window_size = window_size
        self.edge_embedding = edge_embedding

        self.spatial_attention = nn.ModuleList(
            [Spatial_Attention(node_embedding, edge_embedding, log_embedding,
                              heads_n2e=n2e_heads, heads_e2n=e2n_heads, dropout=dropout, batch_size=batch_size,
                               window_size=window_size) for _ in range(self.L)])
        self.sa_add = nn.ModuleList([AddALL(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])
        
        num_edges = trace2pod.shape[0]
        self.temporal_attention = nn.ModuleList(
            [Temporal_Attention(node_embedding, edge_embedding, log_embedding, trace2pod,
                                heads_node=node_heads, heads_edge=edge_heads, heads_log=log_heads, dropout=dropout, window_size=window_size,
                                batch_size=batch_size, num_nodes=num_nodes, num_edges=num_edges) for _ in range(self.L)])
        self.ta_add = nn.ModuleList([AddALL(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])
        self.cross_attention = nn.ModuleList(
            [Encoder_Decoder_Attention(node_embedding, edge_embedding, log_embedding, 
                                        heads_node=node_heads, heads_edge=edge_heads, heads_log=log_heads, dropout=dropout)
                                        for _ in range(self.L)])
        self.ca_add = nn.ModuleList([AddALL(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])     
        self.ffn = nn.ModuleList([FFN(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])

    def forward(self, d_node, d_edge, d_log, z_node, z_edge, z_log):
        d_edge = torch.masked_select(d_edge, self.node_efea.byte()) \
            .reshape(d_edge.shape[0], d_edge.shape[1], -1, d_edge.shape[-1])
        
        for i in range(self.L):
            d_node, d_edge, d_log = self.sa_add[i](d_node, d_edge, d_log, *self.spatial_attention[i](d_node, d_edge, d_log, self.node_adj, self.edge_adj, self.edge_efea))
            d_node, d_edge, d_log = self.ta_add[i](d_node, d_edge, d_log, *self.temporal_attention[i](d_node, d_edge, d_log, mask=True))
            d_node, d_edge, d_log = self.ca_add[i](d_node, d_edge, d_log, *self.cross_attention[i](d_node, d_edge, d_log, z_node, z_edge, z_log))
            d_node, d_edge, d_log = self.ffn[i](d_node, d_edge, d_log)
        return d_node, d_edge, d_log



class Embed(nn.Module):
    def __init__(self, raw_dim, embedding_dim, max_len=1000, dim=4):
        super(Embed, self).__init__()
        self.linear = nn.Linear(raw_dim, embedding_dim)
        self.dim = dim
        pe = torch.zeros((1, max_len, embedding_dim))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000,
                                                                                  torch.arange(0, embedding_dim, 2,
                                                                                               dtype=torch.float32) / embedding_dim)
        pe[:, :, 0::2] = torch.sin(X)
        pe[:, :, 1::2] = torch.cos(X)
        if dim == 4:
            pe = pe.unsqueeze(2)
        elif dim == 5:
            pe = pe.unsqueeze(2).unsqueeze(2)
        self.register_buffer('pe', pe)

    def forward(self, X):
        X = self.linear(X)
        if self.dim == 4:
                padding = (0, 0, 0, 0, 1, 0)
                X_new = F.pad(X, padding, "constant", 0)
                return X + self.pe[:, :X.shape[1], :, :], X_new[:, :X.shape[1], :, :] + self.pe[:, :X.shape[1], :, :]
        else:
                padding = (0, 0, 0, 0, 0, 0, 1, 0)
                X_new = F.pad(X, padding, "constant", 0)
                return X + self.pe[:, :X.shape[1], :, :, :], X_new[:, :X.shape[1], :, :, :] + self.pe[:, :X.shape[1], :, :, :]