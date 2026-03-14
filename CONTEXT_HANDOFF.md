# MSTGAD-dynamic Context Handoff

本文件用于在新对话中快速恢复当前仓库状态、已完成优化、验证结果和下一步建议。

## 1. 项目背景

- 项目基于 baseline `MSTGAD` 做改进。
- 当前实现主线是：`方案 C = 动态图 + 门控融合 + 对比学习`。
- 代码仓库当前已经是改进后的版本，不是原始 baseline。
- 本地 `GAIA` 数据集是截断版，只要求“预处理和训练主链路可跑通”，不适合据此下正式效果结论。
- `MSDS` 数据集是完整的，可以用于验证真实训练/测试行为。

## 2. 已完成的重要代码优化

### 2.1 训练与工程优化

已修改文件：

- [util/train.py](/home/zll/MultiModal/MSTGAD-dynamic/util/train.py)
- [main.py](/home/zll/MultiModal/MSTGAD-dynamic/main.py)
- [main_GAIA.py](/home/zll/MultiModal/MSTGAD-dynamic/main_GAIA.py)
- [util/MSDS/parser_MSDS.py](/home/zll/MultiModal/MSTGAD-dynamic/util/MSDS/parser_MSDS.py)
- [util/GAIA/parser_GAIA.py](/home/zll/MultiModal/MSTGAD-dynamic/util/GAIA/parser_GAIA.py)

已做内容：

- 去掉输入和标签上无意义的 `requires_grad=True`
- 优化 device 搬运
- 支持 `eval_interval`
- 恢复训练集指标评估，并新增 `train_eval_interval` 开关
- 移除 `AMP` 混合精度训练，统一使用 `float32`
- DataLoader 支持 `num_workers / pin_memory / persistent_workers`
- 修复 `GAIA parser` 命令行覆盖问题
- 训练日志增加更多配置输出

注意：

- 真实 `MSDS` 上 `AMP` 会触发 `loss is not finite`，当前代码已移除 `AMP` 训练路径。
- 当前 parser 默认 `train_eval_interval=1`
  - `1` 表示每个 epoch 后都评估训练集指标
  - `0` 表示关闭训练集指标评估

### 2.2 动态图与对比学习加速/调度

已修改文件：

- [src/model.py](/home/zll/MultiModal/MSTGAD-dynamic/src/model.py)
- [src/model_util.py](/home/zll/MultiModal/MSTGAD-dynamic/src/model_util.py)
- [util/train.py](/home/zll/MultiModal/MSTGAD-dynamic/util/train.py)

已做内容：

- 动态图缓存与低频刷新：`graph_update_steps`
- 对比学习延后启用：`contrast_start_epoch`
- 对比学习 warmup：`contrast_warmup`
- 动态图和对比学习都支持 `summary_mode = last/mean`
- 增加动态图缓存统计日志

当前更偏效果的默认配置大致是：

- `graph_update_steps=2`
- `contrast_start_epoch=1`
- `contrast_warmup=2`
- `graph_summary_mode=last`
- `contrast_summary_mode=last`
- `score_fusion_alpha=0.7`

### 2.3 推理分数融合

已修改文件：

- [util/train.py](/home/zll/MultiModal/MSTGAD-dynamic/util/train.py)
- [src/model.py](/home/zll/MultiModal/MSTGAD-dynamic/src/model.py)

已做内容：

- 推理时支持分类概率与重构能量融合
- `score_fusion_alpha` 控制分类分数权重

## 3. MSDS 验证结论

### 3.1 真实 MSDS 已跑通

已使用 baseline 仓库中的完整 MSDS 数据进行真实验证。

重要结果目录：

- [MSTGAD-MSDS-save-e4e2917d-1773488580](/home/zll/MultiModal/MSTGAD-dynamic/result/MSTGAD-MSDS-save-e4e2917d-1773488580)
- [MSTGAD-MSDS-save-b55737ce-1773490537](/home/zll/MultiModal/MSTGAD-dynamic/result/MSTGAD-MSDS-save-b55737ce-1773490537)
- [MSTGAD-MSDS-save-67f8507c-1773491289](/home/zll/MultiModal/MSTGAD-dynamic/result/MSTGAD-MSDS-save-67f8507c-1773491289)

关键结论：

- 工程优化后，训练速度明显改善
- “效果优先”配置在短程验证里比“纯提速”配置更好
- 真实 `MSDS` 上 `AMP` 会出现 `loss is not finite`，因此训练现已固定为 `float32`

## 4. GAIA 预处理已完成的优化

重点文件：

- [util/GAIA/pre_GAIA.py](/home/zll/MultiModal/MSTGAD-dynamic/util/GAIA/pre_GAIA.py)
- [util/GAIA/data_GAIA.py](/home/zll/MultiModal/MSTGAD-dynamic/util/GAIA/data_GAIA.py)

### 4.1 Trace 已优化

当前状态：

- [data/GAIA-pre/trace.csv](/home/zll/MultiModal/MSTGAD-dynamic/data/GAIA-pre/trace.csv) 已保留真实边字段：
  - `src_service`
  - `dst_service`
  - `status_code`
  - `duration_sum`
  - `duration_mean`
  - `count`
- loader 会将其转成真正的边张量：
  - `trace_data[t, src_idx, dst_idx, sc_idx]`
- 不再退化成旧版 self-loop 边特征

验证结论：

- [data/GAIA-save/dataset.pkl](/home/zll/MultiModal/MSTGAD-dynamic/data/GAIA-save/dataset.pkl) 中存在大量非对角边
- 截断版 GAIA 主链路已证明 directed edge feature 生效

### 4.2 Metric 已优化

当前实现逻辑：

1. 固定从 `./data/GAIA/MicroSS` 读取原始数据
2. 读取服务 docker 指标
3. 将 `system_*` 指标按 IP 映射到对应服务节点
4. 对重复时间戳做语义化聚合
   - `network/bytes/packets/errors/count/...` 倾向 `sum`
   - `max/peak/...` 倾向 `max`
   - 其他默认 `mean`
   - 优先避免一个非零值被多个零值稀释
5. 汇总多核 CPU 特征
6. 先做质量筛选，再取稳定共享交集
7. 做短间隔前向填充和 Min-Max

当前结果：

- [data/GAIA-pre/metric.csv](/home/zll/MultiModal/MSTGAD-dynamic/data/GAIA-pre/metric.csv)
- 最终为 `63` 维/节点，即 `630` 个 metric 列 + `timestamp`

验证日志中关键统计：

- `constant=1906`
- `low_coverage=776`
- `low_dynamic=119`
- 最终稳定共享交集：`63`

重要认识：

- 当前 MSTGAD 风格模型要求各节点输入特征同维，且最好语义对齐
- 因此不能简单对 `GAIA metric` 使用全并集
- 目前更合理的主输入是“10 个服务共享的 docker canonical feature space”
- host/middleware 指标更适合作为未来结构升级时的辅助信息，而不是硬塞回当前 `metric.csv`

### 4.3 Log 已优化

当前状态：

- [data/GAIA-pre/log.csv](/home/zll/MultiModal/MSTGAD-dynamic/data/GAIA-pre/log.csv) 除模板计数外，新增：
  - `level_INFO`
  - `level_WARNING`
  - `level_ERROR`
  - `level_DEBUG`
  - `level_UNKNOWN`
  - `log_total`

对应 loader 也已更新：

- `log_len` 现在会自动读取 `template + level + total`
- 在本地截断版 GAIA 上已验证：
  - `log_len = 42`
  - 其中 `36` 个模板 + `5` 个 level + `1` 个 total

### 4.4 Label 已优化

现有训练主标签文件仍保留：

- [data/GAIA-pre/label.csv](/home/zll/MultiModal/MSTGAD-dynamic/data/GAIA-pre/label.csv)

新增辅助标签文件：

- [data/GAIA-pre/label_type.csv](/home/zll/MultiModal/MSTGAD-dynamic/data/GAIA-pre/label_type.csv)
  - 每个 `timestamp x service` 的异常类型字符串
  - 默认 `[normal]`
  - 多异常类型用 `|` 拼接

辅助标签只用于分析，不影响当前训练链路。

## 5. 当前 GAIA 主链路验证状态

本地截断版 GAIA 已完成：

- 预处理成功
- `GAIA-save` 构建成功
- 最小训练 smoke test 成功

关键产物：

- [data/GAIA-pre](/home/zll/MultiModal/MSTGAD-dynamic/data/GAIA-pre)
- [data/GAIA-save/dataset.pkl](/home/zll/MultiModal/MSTGAD-dynamic/data/GAIA-save/dataset.pkl)

最近一次截断版 GAIA smoke test 结果目录：

- [MSTGAD-GAIA-save-69c38446-1773495313](/home/zll/MultiModal/MSTGAD-dynamic/result/MSTGAD-GAIA-save-69c38446-1773495313)
- [MSTGAD-GAIA-save-755ca97b-1773498344](/home/zll/MultiModal/MSTGAD-dynamic/result/MSTGAD-GAIA-save-755ca97b-1773498344)

注意：

- 截断版 GAIA 的指标只用于确认“代码可跑通”和“特征确实接进模型”
- 不应用于正式实验结论
- 最近一次 smoke test 已验证每个 epoch 结束后会打印一行 `[train] pr/rc/auc/ap/f1`

## 6. 当前最重要的结论

### 6.1 关于 GAIA metric

- 节点特征在当前模型里应保持同维，且最好语义对齐
- 不能简单做全并集，否则会造成高稀疏和语义错位
- 当前最合理的是保留共享的 docker canonical space 作为主输入

### 6.2 关于 GAIA trace

- `trace.csv` 已保留 `src_service/dst_service`
- 训练张量中不再保存字符串列名，而是转成 `src_idx/dst_idx`
- 当前主链路里 directed edge feature 已生效

### 6.3 关于 GAIA log

- 仅靠模板计数不够
- `log level` 和 `log_total` 对 GAIA 是高价值增强

### 6.4 关于 GAIA label

- 当前异常检测训练仍建议使用二值 `label.csv`
- 异常类型更适合作为辅助分析信号，而不是直接改成训练主标签

## 7. 建议的新对话起点

如果在新对话里继续优化，推荐优先级如下：

1. 按 anomaly type 做评估拆分
   - 使用 [label_type.csv](/home/zll/MultiModal/MSTGAD-dynamic/data/GAIA-pre/label_type.csv)
   - 看不同异常类型更依赖 metric / trace / log 中哪一模态

2. 做 GAIA 模态消融
   - `metric only`
   - `metric + trace`
   - `metric + log`
   - `metric + trace + log`

3. 做 GAIA 的类型级 score 分析
   - 看分类头概率和重构能量在不同异常类型上的表现差异

4. 如果后续准备升级模型结构
   - 再考虑引入 host/middleware auxiliary metric
   - 不建议直接并入当前统一节点特征向量

## 8. 推荐给新对话的简短提示词

可以在新对话中直接说明：

`请先阅读 CONTEXT_HANDOFF.md，再继续优化 GAIA。当前 trace、metric、log、label 辅助标签都已经优化过，优先做按 anomaly type 的评估与分析。`
