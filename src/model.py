import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from src.model_util import *


class MyModel(nn.Module):
	def __init__(self, graph, **args):
		super(MyModel, self).__init__()
		self.name = 'my'
		self.device = torch.device('cuda' if args.get('gpu', False) and torch.cuda.is_available() else 'cpu')
		self.graph = torch.tensor(graph).to(self.device)
		self.label_weight = args['label_weight']
		num_nodes = args.get('num_nodes', graph.shape[0])

		adj = dense_to_sparse(self.graph)[0]
		trace2pod = torch.nn.functional.one_hot(adj[0], num_classes=graph.shape[0]) \
			+ torch.nn.functional.one_hot(adj[1], num_classes=graph.shape[0])
		trace2pod = trace2pod / trace2pod.sum(axis=0, keepdim=True)
		trace2pod = torch.where(torch.isnan(
			trace2pod), torch.full_like(trace2pod, 0), trace2pod)

		self.encoder = Encoder(graph=self.graph, node_embedding=args['feature_node'], edge_embedding=args['feature_edge'], log_embedding=args['feature_log'],
                         node_heads=args['num_heads_node'], log_heads=args['num_heads_log'], edge_heads=args['num_heads_edge'],
                         n2e_heads=args['num_heads_n2e'], e2n_heads=args['num_heads_e2n'],
                         dropout=args['dropout'], batch_size=args['batch_size'], window_size=args['window'], num_layer=args['num_layer'], trace2pod=trace2pod,
                         graph_hidden=args.get('graph_hidden', 16), num_nodes=num_nodes)
		self.decoder = Decoder(graph=self.graph, node_embedding=args['feature_node'], edge_embedding=args['feature_edge'], log_embedding=args['feature_log'],
                         node_heads=args['num_heads_node'], log_heads=args['num_heads_log'], edge_heads=args['num_heads_edge'],
                         n2e_heads=args['num_heads_n2e'], e2n_heads=args['num_heads_e2n'],
                         dropout=args['dropout'], batch_size=args['batch_size'], window_size=args['window'], num_layer=args['num_layer'], trace2pod=trace2pod,
                         graph_hidden=args.get('graph_hidden', 16), num_nodes=num_nodes)

		self.node_emb = Embed(args['raw_node'], args['feature_node'], dim=4)
		self.log_emb = Embed(args['log_len'], args['feature_log'], dim=4)
		self.egde_emb = Embed(args['raw_edge'], args['feature_edge'], dim=5)

		self.trace2pod = torch.nn.functional.one_hot(adj[0], num_classes=self.graph.shape[0]) \
			+ torch.nn.functional.one_hot(adj[1], num_classes=self.graph.shape[0])
		self.trace2pod = self.trace2pod / 2
		self.register_buffer('edge_src', adj[0].long())
		self.register_buffer('edge_dst', adj[1].long())

		self.dense_node = nn.Linear(args['feature_node'], args['raw_node'])
		self.dense_log = nn.Linear(args['feature_log'], args['log_len'])
		self.dense_edge = nn.Linear(args['feature_edge'], args['raw_edge'])

		self.show = nn.Sequential(nn.Linear(args['raw_node'] + args['raw_edge'] + args['log_len'], (args['raw_node'] + args['raw_edge'] + args['log_len']) // 2),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear((args['raw_node'] + args['raw_edge'] + args['log_len']) // 2, 2))

		# === 创新点1: 动态图学习器（Encoder/Decoder共享） ===
		self.dynamic_graph_learner = DynamicGraphLearner(
			node_dim=args['feature_node'],
			log_dim=args['feature_log'],
			hidden_dim=args.get('graph_hidden', 16),
			num_nodes=num_nodes,
			summary_mode=args.get('graph_summary_mode', 'last')
		)
		self.graph_sparse_weight = args.get('graph_sparse_weight', 1e-3)
		self.graph_update_steps = max(1, int(args.get('graph_update_steps', 4)))
		self._cached_edge_weights = None
		self._cached_graph_reg_loss = None
		self._cached_graph_step = None
		self._graph_cache_hits = 0
		self._graph_cache_refreshes = 0

		# === 创新点3: 对比学习 ===
		self.contrastive_loss = ContrastiveLoss(
			node_dim=args['feature_node'],
			edge_dim=args['feature_edge'],
			log_dim=args['feature_log'],
			proj_dim=args.get('contrast_proj_dim', 32),
			temperature=args.get('contrast_temp', 0.1),
			summary_mode=args.get('contrast_summary_mode', 'last')
		)

	def reset_dynamic_graph_cache(self, reset_stats=False):
		self._cached_edge_weights = None
		self._cached_graph_reg_loss = None
		self._cached_graph_step = None
		if reset_stats:
			self._graph_cache_hits = 0
			self._graph_cache_refreshes = 0

	def get_dynamic_graph_cache_stats(self):
		total = self._graph_cache_hits + self._graph_cache_refreshes
		hit_rate = self._graph_cache_hits / total if total > 0 else 0.0
		return {
			'hits': self._graph_cache_hits,
			'refreshes': self._graph_cache_refreshes,
			'hit_rate': hit_rate
		}

	def _compute_dynamic_graph(self, x_node, x_log):
		edge_weights, graph_reg_loss = self.dynamic_graph_learner(x_node, x_log, self.graph)
		return edge_weights, graph_reg_loss * self.graph_sparse_weight

	def _get_dynamic_graph(self, x_node, x_log, evaluate=False, global_step=None):
		if evaluate or (not self.training) or self.graph_update_steps <= 1:
			if self.training and self.graph_update_steps <= 1:
				self._graph_cache_refreshes += 1
			return self._compute_dynamic_graph(x_node, x_log)

		should_refresh = (
			self._cached_edge_weights is None or
			global_step is None or
			(global_step % self.graph_update_steps == 0)
		)
		if should_refresh:
			edge_weights, total_graph_reg = self._compute_dynamic_graph(x_node, x_log)
			self._cached_edge_weights = edge_weights.detach()
			self._cached_graph_reg_loss = total_graph_reg.detach()
			self._cached_graph_step = global_step
			self._graph_cache_refreshes += 1
			return edge_weights, total_graph_reg

		self._graph_cache_hits += 1
		return self._cached_edge_weights, self._cached_graph_reg_loss

	def forward(self, x, evaluate=False, global_step=None, compute_contrast=True, return_eval_aux=False):
		x_node, d_node = self.node_emb(x['data_node'])
		x_edge, d_edge = self.egde_emb(x['data_edge'])
		x_log, d_log = self.log_emb(x['data_log'])

		# === 创新点1: 动态图低频更新（训练时缓存 K step） ===
		edge_weights, total_graph_reg = self._get_dynamic_graph(
			x_node, x_log, evaluate=evaluate, global_step=global_step
		)

		# 将动态学习到的边权重（软注意力掩膜）应用到边特征上
		# edge_weights shape: [N, N]
		# x_edge shape: [B, W, N, N, F] -> expand 权重进行广播
		weight_mask = edge_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
		x_edge_dynamic = x_edge * weight_mask

		# Encoder/Decoder 使用静态图拓扑，但特征已被动态权重调制（性能+效果双赢）
		z_node, z_edge, z_log = self.encoder(x_node, x_edge_dynamic, x_log)
		node, edge, log = self.decoder(d_node, d_edge, d_log, z_node, z_edge, z_log)
        
		# 对于 l_edge (用于计算重构损失的标注)，保持原始的 x_edge 特征不受缩放影响
		# 这类似于自编码器目标：尽管中间被掩膜/压缩，最终仍需努力还原真实的边特征
		l_edge = x['data_edge'][:, :, self.edge_src, self.edge_dst, :]

		rec_node = torch.square(self.dense_node(node) - x['data_node'])
		rec_edge1 = torch.square(self.dense_edge(edge) - l_edge)
		rec_log = torch.square(self.dense_log(log) - x['data_log'])

		rec_edge = torch.matmul(rec_edge1.permute(
			0, 1, 3, 2), self.trace2pod.float()).permute(0, 1, 3, 2)
		rec = torch.concat([rec_node, rec_log, rec_edge], dim=-1)

		if evaluate:
			rec = rec[:, -1].squeeze()
			rec_score = torch.sum(rec, dim=-1)
			cls_result = torch.softmax(self.show(rec), dim=-1)
			if return_eval_aux:
				return cls_result, x['groundtruth_cls'], rec_score
			return cls_result, x['groundtruth_cls']
		else:
			cls_label = x['groundtruth_cls']

			#cls_label
			rec = rec[:, -1].squeeze()
			cls_result = self.show(rec)
			cls_result = cls_result.reshape(-1, cls_result.shape[-1])
			cls_label = cls_label.reshape(-1, cls_label.shape[-1])

			if cls_label.shape[-1] == 3:
				mask = cls_label[:, -1]
				cls_result, cls_label = cls_result[mask == 0], cls_label[mask == 0]
				cls_label = cls_label[:, :cls_result.shape[-1]]

			# rec_loss
			label_pod = torch.argmax(x['groundtruth_cls'], dim=-1)  # B*N

			node_rec = torch.sum(rec, dim=-1)
			node_right = torch.where(label_pod == 0, node_rec,
			                         torch.zeros_like(node_rec).to(node_rec.device))
			node_wrong = torch.where(label_pod == 1, torch.pow(node_rec, torch.tensor(
				-1, device=node_rec.device)), torch.zeros_like(node_rec).to(node_rec.device))
			node_unkown = torch.where(label_pod == 2, self.label_weight *
			                          node_rec, torch.zeros_like(node_rec).to(node_rec.device))
			rec_loss = [node_right, node_wrong, node_unkown]

			param = label_pod.shape[0] * label_pod.shape[1]
			rec_loss = list(map(lambda x: x.sum() / param, rec_loss))

			if compute_contrast:
				contrast_loss = self.contrastive_loss(z_node, z_edge, z_log, node, edge, log)
			else:
				contrast_loss = rec.new_zeros(())

			return rec_loss, cls_result, cls_label, contrast_loss, total_graph_reg
