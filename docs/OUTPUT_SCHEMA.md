# 输入与输出说明

## 输入要求

每行必须是一个 JSON object，并包含：

| 字段 | 类型 | 含义 |
|---|---|---|
| `premise` | string | NLI 前提 |
| `hypothesis` | string | NLI 假设 |
| `label` | int/string | 期望关系 |

数字标签采用 `0=entailment`、`1=neutral`、`2=contradiction`。其他字段不会被修改。

## `predictions.jsonl`

每条输入记录新增一个 `nli_quality` object：

| 字段 | 含义 |
|---|---|
| `row_position` | 输入文件中的零起始记录位置 |
| `expected_label` | 归一化后的数据标签 |
| `predicted_label` / `predicted_label_id` | 模型 top-1 预测 |
| `agreement` | top-1 预测是否与数据标签相同 |
| `confidence` | top-1 softmax 概率 |
| `margin` | top-1 与 top-2 概率差 |
| `gold_probability` | 模型赋给数据标签的概率 |
| `gold_rank` | 数据标签按模型概率排序后的名次 |
| `negative_log_likelihood` | `-log(gold_probability)` |
| `normalized_entropy` | 三分类预测熵除以 `log(3)` |
| `probabilities` | entailment、neutral、contradiction 三类概率 |
| `input_token_count` | 截断后模型输入 token 数 |
| `at_max_length` | token 数是否达到 `max_length`，用于提示潜在截断 |

## 其他文件

### `disagreements.jsonl`

`agreement=false` 的完整记录视图。它是人工复核候选，不代表这些记录必然错误。

### `high_confidence_agreements.jsonl`

同时满足 `agreement=true` 和 `confidence >= --confidence-threshold` 的记录视图。

### `pair_quality.jsonl`

按 `pair_id` 聚合，包含 source/增强记录数量、各类组内一致状态、组内最低及平均置信度、原记录位置和 MR 列表。输入缺少 `pair_id` 时，每条记录会形成独立的临时分组。

### `summary.json`

包含：

- 输入路径、SHA-256 和记录数；
- 模型名称、请求 revision、实际解析 commit、logits 标签顺序和参数量；
- 运行设备、依赖版本、batch size、最大长度和截断提示；
- overall、source、augmented、label、MR type、MR id 分组指标；
- gold-to-prediction 混淆矩阵；
- 0.50、0.70、0.80、0.90、0.95、0.99 置信度阈值统计；
- 增强记录相对于同一 `pair_id` source 记录的条件统计；
- 输出文件位置。

## 可复现性

复现实验时至少应保留：输入 SHA-256、模型的 `resolved_revision`、模型标签顺序、PyTorch/Transformers 版本、fp16 状态、`max_length` 和 `batch_size`。
