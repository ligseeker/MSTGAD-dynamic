# MSTGAD-Dynamic 技术文档

## 0. 文档定位

本文档面向以下场景：

1. 科研汇报：快速说明项目目标、baseline、创新点和实验设计。
2. 论文整理：明确当前实现到底落在“哪一种方法变体”上。
3. 项目交接：帮助新成员沿着真实训练/推理主链路理解仓库，而不是只看目录猜测。

当前仓库实现对应的方法组合为：

`方案 C = 动态图 + 门控融合 + 对比学习`

但需要强调一个实现层面的事实：

1. 动态图模块并没有改变图拓扑，只是对静态图上的边做动态重加权。
2. “动态图”是在每个 batch 上基于 `node+log` 的全局均值生成一个共享边权矩阵，而不是为每个时间步单独生成 `graph_t`。
3. 对比学习不是“正常/异常样本对比”或“数据增强双视图对比”，而是“编码器表示 vs 解码器表示”的模态级一致性对比。

---

## 1. 项目概览

### 1.1 项目目标与任务

本项目面向微服务系统异常检测，输入是多模态时序观测：

1. `metric`：服务节点级 KPI 指标。
2. `trace`：服务调用/链路相关统计。
3. `log`：日志模板计数。

目标是在滑动时间窗口内，联合建模服务节点之间的空间依赖、时间依赖和多模态关系，最终输出每个服务节点在窗口末时刻的异常/正常预测。

从训练目标看，本项目是一个半监督异常检测模型：

1. 通过重构损失学习正常模式。
2. 通过分类损失利用已标注异常。
3. 通过未知标签掩码保留部分样本为“未标注”。

### 1.2 MSTGAD baseline 的基本思路

从初始版本代码和当前仍保留的骨架看，baseline MSTGAD 的核心是：

1. 双图建模：
   - Node Graph：服务节点图。
   - Edge Graph：调用边图。
2. 三模态输入：
   - node/metric 走节点侧。
   - edge/trace 走边侧。
   - log 与 node 同属于节点侧观测。
3. Encoder-Decoder 架构：
   - Encoder：空间注意力 + 时间注意力 + FFN。
   - Decoder：空间注意力 + 带因果 mask 的时间注意力 + 编解码交叉注意力 + FFN。
4. 异常检测方式：
   - 先重构 metric/log/trace。
   - 将最后一个时间步的重构误差拼接。
   - 再通过一个小型 MLP 分类头输出正常/异常概率。

### 1.3 当前改进版的整体方法

当前改进版在 baseline 上加入了三个关键增强：

1. 动态图重加权：
   - 用 `DynamicGraphLearner` 根据当前 batch 的节点和日志表示生成边权。
   - 用该边权对 `x_edge` 做软掩膜，再送入原有 Encoder/Decoder。
2. 门控跨模态融合：
   - 将原始 `Temporal_Attention` 里基于固定 `trace2pod` 的均值融合，替换为 `GatedCrossModalFusion`。
   - 让 node/edge/log 的时间注意力图之间以门控方式自适应融合。
3. 对比学习正则：
   - 将编码器输出和解码器输出投影到同一空间。
   - 对 node/edge/log 三个模态分别做 InfoNCE。

因此，当前方法可以概括为：

`静态拓扑上的动态边权重 + 门控时序跨模态融合 + 编解码表示一致性对比学习`

---

## 2. 代码结构分析

### 2.1 主要目录与核心文件

| 路径 | 作用 | 是否主链路 |
| --- | --- | --- |
| `main.py` | MSDS 数据集训练/评估入口 | 是 |
| `main_GAIA.py` | GAIA 数据集训练/评估入口 | 是 |
| `src/model.py` | 总模型封装，定义 embedding、encoder、decoder、loss 输出 | 是 |
| `src/model_util.py` | 主要网络组件，含 baseline 主干与三项创新模块 | 是 |
| `util/train.py` | 训练器、优化器、调度器、保存与评估逻辑 | 是 |
| `util/util.py` | 指标计算、日志目录管理、参数存取、随机种子 | 是 |
| `util/MSDS/data_MSDS.py` | MSDS 数据加载与窗口化 | 是 |
| `util/GAIA/data_GAIA.py` | GAIA 数据加载与窗口化 | 是 |
| `util/MSDS/parser_MSDS.py` | MSDS 训练参数 | 是 |
| `util/GAIA/parser_GAIA.py` | GAIA 训练参数 | 是 |
| `util/MSDS/pre_MSDS.py` | MSDS 原始数据预处理 | 预处理链路 |
| `util/GAIA/pre_GAIA.py` | GAIA 原始数据预处理 | 预处理链路 |
| `merge_dataset.py` | 将分散的 `pkl` 窗口合并为 `dataset.pkl` | 辅助脚本 |
| `summarize_results.py` | 汇总 `result/` 下实验指标 | 辅助脚本 |
| `innovation_analysis.md` | 创新点方案说明，不参与运行 | 说明文档 |

### 2.2 训练/推理入口

MSDS 入口：

1. [`main.py`](/home/zll/MultiModal/MSTGAD-dynamic/main.py)
2. 参数来自 [`util/MSDS/parser_MSDS.py`](/home/zll/MultiModal/MSTGAD-dynamic/util/MSDS/parser_MSDS.py)
3. 数据来自 [`util/MSDS/data_MSDS.py`](/home/zll/MultiModal/MSTGAD-dynamic/util/MSDS/data_MSDS.py)

GAIA 入口：

1. [`main_GAIA.py`](/home/zll/MultiModal/MSTGAD-dynamic/main_GAIA.py)
2. 参数来自 [`util/GAIA/parser_GAIA.py`](/home/zll/MultiModal/MSTGAD-dynamic/util/GAIA/parser_GAIA.py)
3. 数据来自 [`util/GAIA/data_GAIA.py`](/home/zll/MultiModal/MSTGAD-dynamic/util/GAIA/data_GAIA.py)

### 2.3 模块依赖关系

主依赖链如下：

```text
main.py / main_GAIA.py
  -> parser_* 获取 args
  -> Process(...) 构造滑动窗口数据集与 graph
  -> DataLoader
  -> MyModel(graph, **args)
       -> Embed
       -> Encoder
            -> Spatial_Attention
            -> Temporal_Attention
                 -> GatedCrossModalFusion
       -> Decoder
            -> Spatial_Attention
            -> Temporal_Attention(mask=True)
            -> Encoder_Decoder_Attention
       -> DynamicGraphLearner
       -> ContrastiveLoss
  -> train.MY.fit()
       -> AdaBelief + StepLR
       -> total loss
  -> train.MY.evaluate()
       -> util.calc_index()
```

---

## 3. 核心架构

### 3.1 整体 pipeline

完整 pipeline 可拆成 6 个阶段：

1. 原始数据预处理：
   - 生成 `metric.csv`、`log.csv`、`trace.csv`、`trace_path.pkl`、标签文件。
2. 窗口化：
   - 将连续时序切成 `[window, ...]` 的样本。
3. 模态嵌入：
   - `data_node/data_edge/data_log -> embedding + position encoding`
4. 编码：
   - 空间注意力建模图结构依赖。
   - 时间注意力建模窗口内时序依赖。
5. 解码：
   - 在因果时间 mask 下重构多模态表示。
   - 通过 cross-attention 使用编码器记忆。
6. 异常判别：
   - 计算重构误差。
   - 取窗口最后时刻误差向量，送入分类头。

### 3.2 模型主要组件

| 组件 | 文件/类 | 职责 | baseline / 新增 |
| --- | --- | --- | --- |
| 输入嵌入 | `src/model_util.py::Embed` | 原始特征线性映射 + 位置编码 | baseline |
| 图邻接展开 | `src/model_util.py::adj2adj` | 静态图转 node/edge 图索引 | baseline |
| 空间建模 | `src/model_util.py::Spatial_Attention` | 用 GATv2Conv 在 node-edge 双图上传播 | baseline |
| 时间建模 | `src/model_util.py::Temporal_Attention` | 模态内时间自注意力 + 跨模态融合 | baseline，已被增强 |
| 编码器 | `src/model_util.py::Encoder` | 多层空间/时间/FFN 堆叠 | baseline |
| 解码器 | `src/model_util.py::Decoder` | 带因果 mask 的解码与 cross-attn | baseline |
| 动态图 | `src/model_util.py::DynamicGraphLearner` | 生成动态边权软掩膜 | 新增 |
| 门控融合 | `src/model_util.py::GatedCrossModalFusion` | 自适应融合三模态时间注意力 | 新增 |
| 对比学习 | `src/model_util.py::ContrastiveLoss` | 编码器-解码器表示一致性约束 | 新增 |
| 总模型 | `src/model.py::MyModel` | 串联所有组件并输出损失 | baseline 主封装，已增强 |

### 3.3 输入 -> 模型 -> 输出

输入样本字典的核心字段为：

1. `data_node`: `[B, W, N, F_node_raw]`
2. `data_edge`: `[B, W, N, N, F_edge_raw]`
3. `data_log`: `[B, W, N, F_log_raw]`
4. `groundtruth_cls`: `[B, N, 3]`，训练标签，类别为正常/异常/未知
5. `groundtruth_real`: `[B, N, 2]`，真实评估标签，类别为正常/异常

模型输出有两种模式：

训练模式：

1. `rec_loss`: 三部分重构损失列表
2. `cls_result`: 分类 logits
3. `cls_label`: 分类标签
4. `contrast_loss`: 对比损失
5. `graph_reg_loss`: 动态图正则

评估模式：

1. `cls_result`: softmax 后的二分类概率
2. `groundtruth_cls`: 返回标签占位，评估时实际使用的是 `groundtruth_real`

### 3.4 baseline 与新增模块的边界

属于 baseline 的部分：

1. `Embed`
2. `Encoder`
3. `Decoder`
4. `Spatial_Attention`
5. `Encoder_Decoder_Attention`
6. 重构误差 + 分类头 `show`
7. 半监督标签掩码思路

属于新增模块的部分：

1. `DynamicGraphLearner`
2. `GatedCrossModalFusion`
3. `ContrastiveLoss`
4. 训练总损失中的 `contrast_loss` 和 `graph_reg_loss`
5. 相关超参数：`graph_hidden`、`graph_sparse_weight`、`contrast_*`

---

## 4. 数据流分析

### 4.1 数据读取与预处理

#### MSDS

主文件：[`util/MSDS/data_MSDS.py`](/home/zll/MultiModal/MSTGAD-dynamic/util/MSDS/data_MSDS.py)

读取内容：

1. `label.pkl` -> `label` 与 `label_mask`
2. `metric.csv` -> 节点指标
3. `log.csv` -> 日志模板计数
4. `trace.csv` -> 链路统计
5. `trace_path.pkl` -> 静态图邻接矩阵

关键操作：

1. 指标缺失时间戳补齐后前向填充。
2. 日志按模板计数并归一化。
3. trace 聚合为 `[T, N, N, F_trace]`。
4. 滑动窗口产生样本。

#### GAIA

主文件：[`util/GAIA/data_GAIA.py`](/home/zll/MultiModal/MSTGAD-dynamic/util/GAIA/data_GAIA.py)

相对 MSDS 的差异：

1. 支持自动检测 `raw_node/log_len/raw_edge`。
2. metric 使用分块读取和缓存 `metric_aggregated.pkl`。
3. 根据时间间隔检测连续段，避免跨长间隔拼窗口。
4. 支持 `max_timesteps` 限制加载规模。

### 4.2 窗口样本结构

单个窗口样本格式为：

```python
{
  "data_node": [W, N, F_node_raw],
  "data_log": [W, N, F_log_raw],
  "data_edge": [W, N, N, F_edge_raw],
  "groundtruth_cls": [N, 3],
  "groundtruth_real": [N, 2],
}
```

时间标签取窗口最后一个时刻。

### 4.3 前向传播 tensor 流程

#### Step 1. embedding

在 [`src/model.py`](/home/zll/MultiModal/MSTGAD-dynamic/src/model.py) 中：

```text
data_node [B,W,N,F_n_raw] -> node_emb -> x_node [B,W,N,F_n]
data_edge [B,W,N,N,F_e_raw] -> edge_emb -> x_edge [B,W,N,N,F_e]
data_log  [B,W,N,F_l_raw] -> log_emb  -> x_log  [B,W,N,F_l]
```

同时 `Embed` 还会额外返回一份前面补 0 的解码器输入 `d_node/d_edge/d_log`。

#### Step 2. 动态图权重

`DynamicGraphLearner` 输入：

1. `x_node [B,W,N,F_n]`
2. `x_log [B,W,N,F_l]`
3. `static_graph [N,N]`

执行：

```text
h = mean_W(mean_B(cat(x_node, x_log))) -> [N, F_n+F_l]
src = Linear(h)
tgt = Linear(h)
sim = src @ tgt^T / sqrt(d)
adj_dynamic = sigmoid(sim / T)  (+ train noise)
edge_weights = adj_dynamic * static_graph
fused_weights = λ * edge_weights + (1-λ) * static_graph
```

然后用 `weight_mask` 广播到边特征：

```text
x_edge_dynamic = x_edge * fused_weights[None,None,:,:,None]
```

#### Step 3. 编码器

编码器内部先基于静态图把 `[N,N]` 的 dense edge 特征筛到“真实边集合”：

```text
[B,W,N,N,F_e] -> masked_select(static_graph) -> [B,W,E,F_e]
```

每一层依次执行：

1. `Spatial_Attention`
2. `Temporal_Attention`
3. `FFN`

输出：

1. `z_node [B,W,N,F_n]`
2. `z_edge [B,W,E,F_e]`
3. `z_log [B,W,N,F_l]`

#### Step 4. 解码器

解码器输入是补 0 后的 `d_*`，并重复多层：

1. `Spatial_Attention`
2. `Temporal_Attention(mask=True)`
3. `Encoder_Decoder_Attention`
4. `FFN`

输出重构隐表示：

1. `node`
2. `edge`
3. `log`

#### Step 5. 重构误差

`MyModel.forward()` 中：

1. `dense_node(node)` 回到原始 node 维度
2. `dense_edge(edge)` 回到原始 edge 维度
3. `dense_log(log)` 回到原始 log 维度

逐元素平方误差：

```text
rec_node = (recon_node - data_node)^2
rec_edge1 = (recon_edge - l_edge)^2
rec_log  = (recon_log  - data_log)^2
```

然后通过 `trace2pod` 把边误差映射回节点空间：

```text
rec_edge = matmul(rec_edge1, trace2pod)
rec = concat([rec_node, rec_log, rec_edge], dim=-1)
```

#### Step 6. 分类输出

仅使用窗口最后一个时间步：

```text
rec_last = rec[:, -1]         # [B, N, F_concat]
show(rec_last) -> [B, N, 2]
```

因此最终异常打分并不是“直接用 reconstruction error 阈值”，而是“用 reconstruction error 向量作为分类器输入得到异常概率”。

### 4.4 loss 计算

训练总损失位于 [`util/train.py`](/home/zll/MultiModal/MSTGAD-dynamic/util/train.py)。

#### 分类损失

`BCEWithLogitsLoss`，并对异常类使用更高权重：

```text
cls_loss = BCEWithLogitsLoss(weight=[1, abnormal_weight])
```

未知标签会在 `MyModel.forward()` 中被过滤掉，不进入分类损失。

#### 重构损失

重构损失不是简单的统一 MSE，而是按标签区分：

1. 正常节点：最小化重构误差。
2. 异常节点：最小化 `1 / 重构误差`，等价于鼓励异常点重构误差更大。
3. 未知节点：最小化 `label_weight * 重构误差`。

这是一种显式“拉开正常/异常重构行为”的半监督设计。

#### 对比损失

对 `enc_node/dec_node`、`enc_edge/dec_edge`、`enc_log/dec_log` 分别做 InfoNCE，再取均值。

#### 图正则

动态图正则为：

```text
graph_reg = sparse_loss + 0.1 * KL(dynamic || static)
```

训练总损失：

```text
loss = (1 - para) * cls_loss + para * rec_loss + gamma * contrast_loss + graph_reg_loss
```

其中：

1. `para` 会随 epoch 衰减，控制前期更偏重重构、后期更偏重分类。
2. `gamma` 对比损失有 warm-up。

### 4.5 anomaly score 生成方式

推理阶段：

1. 前向时走 `evaluate=True`。
2. 分类头输出经过 `softmax`。
3. `util.calc_index()` 在二分类情况下对预测概率做 `argmax`，再计算指标。

因此当前仓库的 anomaly score 本质上是：

`P(anomaly | reconstruction-error features)`

而不是直接把 `rec_loss` 作为异常分数。

---

## 5. baseline MSTGAD 方法解析

结合初始提交版本与当前仍保留的 baseline 主干，可将 MSTGAD baseline 概括为以下四点。

### 5.1 图建模

1. 静态图来自 `trace_path.pkl`。
2. 图拓扑通过 `adj2adj()` 转换成：
   - `node_adj`
   - `edge_adj`
   - `edge_efea`
3. node graph 与 edge graph 通过 GATv2Conv 实现双向消息传播。

### 5.2 时序建模

1. 每个模态独立做窗口内 Multi-Head Self-Attention。
2. 原始 baseline 在 `Temporal_Attention` 中通过 `trace2pod` 将 edge 注意力映射到 node，或反向映射。
3. 多模态注意力图通过固定映射后的均值进行融合。

baseline 原始逻辑可简化为：

```text
att_n = self-attn(node)
att_t = self-attn(trace)
att_l = self-attn(log)

att_tn = trace2pod(att_t)
att_nn = trace2pod^T(att_n)
att_ln = trace2pod^T(att_l)

att_node = mean(att_n, att_tn, att_l)
att_edge = mean(att_nn, att_t, att_ln)
```

### 5.3 训练目标

baseline 目标主要由两部分组成：

1. 重构损失：
   - 通过 Encoder-Decoder 重构多模态输入。
   - 正常点倾向于小误差，异常点倾向于大误差。
2. 分类损失：
   - 用最后时刻重构误差特征做正常/异常二分类。

### 5.4 异常评分机制

最终异常评分来自分类头 `show` 的输出，而不是单独使用 reconstruction error 阈值。

因此，baseline 更像是：

`重构引导的判别式异常检测`

而不是传统纯重构式异常检测。

---

## 6. 创新点实现分析

## 6.1 动态图（Dynamic Graph）

### 实现位置

1. `src/model_util.py::DynamicGraphLearner`
2. `src/model.py::MyModel.forward`

### 输入输出

输入：

1. `x_node [B,W,N,F_n]`
2. `x_log [B,W,N,F_l]`
3. `static_graph [N,N]`

输出：

1. `fused_weights [N,N]`
2. `graph_reg_loss []`

### 关键计算逻辑

核心逻辑：

1. 将 node 与 log 表示拼接。
2. 沿 batch 和 time 取均值，得到节点级全局表示。
3. 通过 `src/tgt` 两个线性投影做双边相似度。
4. 通过 sigmoid 变成连续边权。
5. 与静态图逐元素相乘，只保留静态拓扑已有边。
6. 用可学习 `λ` 在动态权重和静态全 1 边权之间融合。

伪代码：

```python
h = mean_B(mean_W(concat(x_node, x_log)))   # [N, D]
src = W_src(h)
tgt = W_tgt(h)
sim = src @ tgt.T / sqrt(D)
adj_dynamic = sigmoid(sim / T)
edge_weights = adj_dynamic * static_graph
fused_weights = sigmoid(lambda_) * edge_weights + (1-sigmoid(lambda_)) * static_graph
```

### 与 baseline 的区别

baseline：

1. 图只有静态邻接关系。
2. edge 特征直接进入 encoder。

当前实现：

1. 图拓扑仍是静态的。
2. 但边特征进入 encoder 前先乘上动态边权软掩膜。

### 需要特别说明的实现事实

1. 当前实现是“动态边权学习”，不是“动态图拓扑重建”。
2. 边权是 batch 级共享的，不是每个窗口位置独立的时间动态图。
3. 动态图只影响 `x_edge_dynamic`，`adj2adj()` 生成的图索引仍完全来自静态图。

### 潜在收益

1. 允许不同 batch 下边重要性变化。
2. 通过图稀疏正则约束，减少噪声边的影响。
3. 相比每次重建图索引，代价更低。

---

## 6.2 门控融合（Gated Fusion）

### 实现位置

1. `src/model_util.py::GatedCrossModalFusion`
2. `src/model_util.py::Temporal_Attention`

### 输入输出

输入：

1. `att_n [B,N,H,W,W]`
2. `att_t [B,E,H,W,W]`
3. `att_l [B,N,H,W,W]`

输出：

1. `fused_node_att`
2. `fused_edge_att`
3. `fused_log_att`

### 关键计算逻辑

1. 将注意力图展平为 `[entity, H*W*W]`。
2. 对每个模态做 entity 维均值池化，得到全局注意力摘要。
3. 对于 node，取 `(edge_global + log_global)/2` 作为跨模态信息。
4. 拼接 `self_att` 与 `cross_modal_info`，通过 sigmoid 门控网络得到逐维门值。
5. 用 `gate * self + (1-gate) * cross` 得到融合结果。

伪代码：

```python
att_n_global = mean_entity(att_n)
att_t_global = mean_entity(att_t)
att_l_global = mean_entity(att_l)

cross_for_node = proj((att_t_global + att_l_global) / 2)
gate_n = sigmoid(MLP([att_n_flat, cross_for_node]))
fused_n = gate_n * att_n_flat + (1 - gate_n) * cross_for_node
```

### 与 baseline 的区别

baseline：

1. 用固定 `trace2pod` 把 attention 从 edge 映射回 node。
2. 多模态信息以固定平均方式融合。

当前实现：

1. 不再依赖固定 one-hot 映射完成主要融合。
2. 直接对注意力图做可学习门控融合。
3. 允许不同模态在不同 batch 中占比不同。

### 需要特别说明的实现事实

1. `trace2pod` 仍存在，但主要用于后续边误差回投到节点，不再是 Temporal_Attention 的核心融合方式。
2. 门控融合发生在注意力图层面，不是特征层面。
3. 若 `heads_edge != heads_node`，代码会先做 head 对齐；但默认参数下三者 head 数相同。

### 潜在收益

1. 避免固定映射过于刚性。
2. 能适应不同故障下 metric/log/trace 信息贡献不同。
3. 有利于解释“哪种模态在当前异常检测中更重要”。

---

## 6.3 对比学习（Contrastive Learning）

### 实现位置

1. `src/model_util.py::ContrastiveLoss`
2. `src/model.py::MyModel.forward`
3. `util/train.py::MY.fit`

### 输入输出

输入：

1. `enc_node, enc_edge, enc_log`
2. `dec_node, dec_edge, dec_log`

输出：

1. 单标量 `contrast_loss`

### 关键计算逻辑

1. 对每个模态沿时间与实体维做均值池化，得到 batch 级表示。
2. 通过投影头投到同一对比空间。
3. 使用归一化余弦相似度构造 InfoNCE logits。
4. 正样本是“同一 batch index 的 enc/dec 表示对”。

伪代码：

```python
enc_n = proj(mean(enc_node, dim=(1,2)))
dec_n = proj(mean(dec_node, dim=(1,2)))
loss_node = InfoNCE(enc_n, dec_n)
loss_edge = InfoNCE(enc_e, dec_e)
loss_log  = InfoNCE(enc_l, dec_l)
contrast_loss = (loss_node + loss_edge + loss_log) / 3
```

### 与 baseline 的区别

baseline：

1. 没有显式表示对齐损失。
2. 只靠重构与分类学习判别边界。

当前实现：

1. 强制编码器和解码器的同模态语义保持一致。
2. 在训练总损失中作为附加正则项。
3. 通过 `contrast_warmup` 避免训练初期不稳定。

### 需要特别说明的实现事实

1. 这不是论文中常见的“多视图增强对比学习”。
2. 也不是“正常样本 vs 异常样本”的监督式对比。
3. 它更准确的名字应是“编解码表示一致性对比正则”。

### 潜在收益

1. 稳定 latent 表示空间。
2. 减少 encoder 与 decoder 表示漂移。
3. 提高多模态重构表征的一致性。

---

## 7. 相对 baseline 的改动总结

| 改动模块 | baseline 做法 | 当前做法 | 修改动机 | 潜在收益 |
| --- | --- | --- | --- | --- |
| 图建模 | 固定静态图 | 静态拓扑 + 动态边权 | 图强度随状态变化 | 更灵活地刻画依赖关系 |
| 时间跨模态融合 | `trace2pod` 固定映射 + 平均 | 门控注意力融合 | 不同模态贡献不同 | 自适应多模态权重 |
| 表示学习约束 | 无对比损失 | encoder-decoder InfoNCE | 提高表示区分度 | 更稳定的 latent 空间 |
| 训练目标 | 分类 + 重构 | 分类 + 重构 + 对比 + 图正则 | 提升鲁棒性与判别性 | 更强泛化与结构先验 |
| 参数系统 | baseline 参数 | 新增 `graph_*`、`contrast_*` | 支持新模块实验 | 方便做消融 |
| 数据支持 | MSDS | MSDS + GAIA | 扩展适用数据集 | 更广泛实验基础 |

---

## 8. 训练机制

### 8.1 训练流程

训练主链路见 [`util/train.py`](/home/zll/MultiModal/MSTGAD-dynamic/util/train.py)。

流程如下：

1. 初始化权重。
2. 使用 `AdaBelief` 优化器。
3. 使用 `StepLR` 调整学习率。
4. 每个 epoch：
   - 将 batch 转到设备上。
   - 前向得到 `rec_loss/cls_loss/contrast_loss/graph_reg_loss`。
   - 按权重组装总损失。
   - 反向传播。
   - 梯度裁剪。
5. 保存两类 best checkpoint：
   - `loss` 最优
   - `f1` 最优

### 8.2 loss 组成

总损失：

```text
L_total = (1 - para) * L_cls + para * L_rec + gamma * L_contrast + L_graph
```

其中：

1. `para = 1 / (epoch // rec_down + 1)`，并受 `para_low` 下界限制。
2. `gamma` 由 `contrast_warmup` 控制线性升温。

### 8.3 optimizer / scheduler

1. Optimizer：`AdaBelief`
2. Scheduler：`torch.optim.lr_scheduler.StepLR`
3. Gradient clip：`max_norm=10`

### 8.4 实验配置方式

MSDS：

1. 通过 CLI 参数读取。
2. 参数文件保存在实验目录 `params.json`。

GAIA：

1. 参数定义在 `parser_GAIA.py`。
2. 但当前实现使用 `parse_args([])`，这意味着命令行参数实际上不会生效。

这是一个重要实现注意事项：

1. `main_GAIA.py --epochs 2` 这种写法按当前代码不会真正覆盖默认值。
2. 若要严格做 GAIA 实验管理，需要先修复 `parser_GAIA.py`。

---

## 9. 推理与异常检测

### 9.1 推理流程

推理与训练共用 `MyModel.forward()`，差异仅在 `evaluate=True`：

1. 仍然会走 embedding、动态图重加权、encoder、decoder、重构误差。
2. 但不再返回损失项。
3. 仅输出 `softmax(self.show(rec_last))`。

### 9.2 anomaly score

当前代码中的 anomaly score 等价于：

```text
score = softmax(show(reconstruction_error_features))[:, :, 1]
```

即窗口最后时刻每个节点属于异常类的概率。

### 9.3 评估指标

由 [`util/util.py`](/home/zll/MultiModal/MSTGAD-dynamic/util/util.py) 的 `calc_index()` 计算：

1. Precision
2. Recall
3. ROC-AUC
4. Average Precision
5. F1

注意：

1. 评估时使用的是 `groundtruth_real`。
2. 若预测是二分类概率，先对最后一维做 `argmax`。

---

## 10. 数据流 / 调用链总结

## 10.1 训练调用链

```text
main.py / main_GAIA.py
  -> Process(...)
  -> dataset(list of dict)
  -> DataLoader
  -> MyModel.forward(batch)
       -> Embed
       -> DynamicGraphLearner
       -> Encoder
            -> Spatial_Attention
            -> Temporal_Attention
                 -> GatedCrossModalFusion
            -> FFN
       -> Decoder
            -> Spatial_Attention
            -> Temporal_Attention(mask=True)
                 -> GatedCrossModalFusion
            -> Encoder_Decoder_Attention
            -> FFN
       -> reconstruction error
       -> classifier head
       -> ContrastiveLoss
  -> train.MY.fit()
       -> total loss
       -> optimizer.step()
```

## 10.2 前向伪代码

```python
def forward(batch):
    x_node, d_node = node_emb(batch["data_node"])
    x_edge, d_edge = edge_emb(batch["data_edge"])
    x_log,  d_log  = log_emb(batch["data_log"])

    edge_weights, graph_reg = dynamic_graph_learner(x_node, x_log, static_graph)
    x_edge_dynamic = x_edge * edge_weights

    z_node, z_edge, z_log = encoder(x_node, x_edge_dynamic, x_log)
    node, edge, log = decoder(d_node, d_edge, d_log, z_node, z_edge, z_log)

    rec_node = mse(dense_node(node), batch["data_node"])
    rec_edge = mse(dense_edge(edge), real_edge_on_static_graph)
    rec_log  = mse(dense_log(log), batch["data_log"])

    rec_edge = project_edge_error_to_node(rec_edge, trace2pod)
    rec = concat(rec_node, rec_log, rec_edge)
    rec_last = rec[:, -1]

    logits = show(rec_last)

    if evaluate:
        return softmax(logits)

    rec_loss = semi_supervised_reconstruction_loss(rec_last, labels)
    contrast = contrastive_loss(z_node, z_edge, z_log, node, edge, log)
    return rec_loss, logits, cls_label, contrast, graph_reg
```

---

## 11. 核心模块总结表

| 模块 | 输入 | 输出 | 关键逻辑 | 备注 |
| --- | --- | --- | --- | --- |
| `Embed` | 原始模态特征 | 编码特征 + decoder 输入 | 线性映射 + 位置编码 + 前置零填充 | baseline |
| `DynamicGraphLearner` | `x_node`, `x_log`, `graph` | `edge_weights`, `graph_reg` | 节点相似度生成动态边权 | 新增 |
| `Spatial_Attention` | `x_node`, `x_edge`, `x_log` | 更新后的三模态表示 | GATv2Conv 在 node-edge 双图传播 | baseline |
| `Temporal_Attention` | 三模态时序表示 | 更新后的三模态表示 | 模态内 self-attn + 跨模态融合 | baseline 主体，已增强 |
| `GatedCrossModalFusion` | 三模态 attention map | 融合后的 attention map | 门控控制 self/cross 权重 | 新增 |
| `Encoder_Decoder_Attention` | decoder 状态 + encoder 记忆 | 对齐后的 decoder 状态 | cross-attention | baseline |
| `ContrastiveLoss` | enc/dec 模态表示 | 标量损失 | 三模态 InfoNCE | 新增 |
| `show` 分类头 | 最后时刻重构误差向量 | 2 类 logits | MLP 判别 | baseline |

---

## 12. 当前实现的协同关系分析

### 12.1 动态图 + 门控融合 + 对比学习如何协同

三者的分工可以理解为：

1. 动态图：
   - 解决“哪些边更重要”。
2. 门控融合：
   - 解决“哪些模态更重要”。
3. 对比学习：
   - 解决“编码器和解码器表征是否稳定一致”。

对应三个层次：

1. 图结构层：边权调整
2. 注意力层：模态融合调整
3. 表示层：latent 对齐约束

所以它们不是重复改进，而是分别作用于不同层面。

### 12.2 当前实现最可能的性能贡献点

从代码实际作用范围看，最可能带来收益的顺序大致是：

1. 门控融合
   - 直接替换了 baseline 最核心的 temporal cross-modal 融合逻辑。
2. 动态图重加权
   - 在进入 encoder 前修改了 edge 特征强度。
3. 对比学习
   - 更多是辅助正则，收益可能依赖超参数与数据规模。

如果实验结果明显提升，最值得优先归因的通常会是前两者。

### 12.3 值得做的消融实验

建议至少做以下消融：

1. Baseline
2. Baseline + Dynamic Graph
3. Baseline + Gated Fusion
4. Baseline + Contrastive Learning
5. Baseline + Dynamic Graph + Gated Fusion
6. Baseline + Dynamic Graph + Contrastive Learning
7. Baseline + Gated Fusion + Contrastive Learning
8. Full Model

建议同时观察：

1. F1
2. AUC
3. AP
4. 收敛曲线
5. 不同异常类型上的表现

### 12.4 未来可优化部分

#### 动态图

1. 做真正的 time-step 级动态图，而不是 batch 级共享图。
2. 让动态图直接参与 message passing 拓扑，而不是只重加权边特征。
3. 引入稀疏化或 top-k 结构学习。

#### 门控融合

1. 从 attention-level 门控扩展到 feature-level cross-attention。
2. 显式输出模态权重，便于可解释性分析。
3. 在 node/edge/log 三个方向使用不同的跨模态路由。

#### 对比学习

1. 引入增强视图，构造真正的多视图对比。
2. 加入正常/异常监督对比。
3. 在时间维度构造同节点跨时刻对比。

#### 检测机制

1. 将分类概率与重构误差联合成最终 anomaly score。
2. 引入阈值搜索或 calibrated score。
3. 做异常类型识别而不仅是二分类。

---

## 13. 相对 baseline 的改动清单

1. 在 [`src/model.py`](/home/zll/MultiModal/MSTGAD-dynamic/src/model.py) 中新增 `DynamicGraphLearner` 与 `ContrastiveLoss`，并把前向返回值从 3 项扩展到 5 项。
2. 在 [`src/model_util.py`](/home/zll/MultiModal/MSTGAD-dynamic/src/model_util.py) 中新增 `DynamicGraphLearner`、`GatedCrossModalFusion`、`ContrastiveLoss`。
3. 在 `Temporal_Attention` 中，将 baseline 的固定 `trace2pod` 跨模态均值融合替换为门控融合。
4. 在 [`util/train.py`](/home/zll/MultiModal/MSTGAD-dynamic/util/train.py) 中新增对比损失 warm-up 与图正则损失聚合。
5. 在参数文件中新增 `graph_hidden`、`graph_sparse_weight`、`contrast_weight`、`contrast_temp`、`contrast_proj_dim`、`contrast_warmup`。
6. 仓库新增 GAIA 数据处理与训练入口。
7. 数据缓存从散落小 `pkl` 兼容到统一 `dataset.pkl`。

---

## 14. 需要在汇报或交接中明确说明的实现细节

这些点很容易被口头描述“讲大了”，建议在科研汇报中提前说清楚。

1. 当前动态图不改变邻接拓扑，只改变边特征权重。
2. 当前动态图是 batch 级共享，不是时刻级动态图。
3. 当前对比学习是 enc-dec 一致性对比，不是正常/异常样本对比。
4. 当前 anomaly score 来自分类头，而不是直接用重构误差。
5. GAIA 数据分支中的 trace 特征聚合使用的是 `service_name` 单节点统计，邻接矩阵来自 parent-child 调用关系，但加载时的 `trace_data` 写入为 `trace_data[ts, svc_idx, svc_idx, sc_idx]`，因此 GAIA 的边特征表达明显弱于 MSDS 的显式 `src-dst` trace。
6. `parser_GAIA.py` 当前使用 `parse_args([])`，命令行实验配置不会生效。

---

## 15. 后续实验与优化建议

### 15.1 近期建议

1. 先完成 8 组消融实验，明确三个创新点的独立收益。
2. 单独汇报动态图模块时，务必写成“dynamic edge weighting”而不是“full dynamic graph topology learning”。
3. 对门控融合做可视化，展示不同异常类型下 node/edge/log 权重变化。

### 15.2 中期建议

1. 修复 GAIA CLI 参数不可覆盖的问题。
2. 将 GAIA 的 trace 从“服务级自环统计”升级到“显式 caller-callee 边统计”。
3. 在日志中输出 `graph_lambda`、门控均值和对比损失曲线，增强实验可解释性。

### 15.3 论文写作建议

方法命名建议更贴近实现：

`Dynamic Edge-Reweighted MSTGAD with Gated Temporal Cross-Modal Fusion and Encoder-Decoder Contrastive Regularization`

如果论文中直接声称“动态图结构学习”，审稿时很可能会被追问是否真正学习了时变拓扑，因此建议措辞谨慎。

---

## 16. 一句话总结

当前仓库不是简单的“MSTGAD 复现”，而是一个在 MSTGAD 双图多模态骨架上，加入动态边权、门控时序融合和编解码对比正则的增强版异常检测系统；它保留了 baseline 的主体建模范式，但在图权、模态权和表示约束三个层次上做了系统性强化。
