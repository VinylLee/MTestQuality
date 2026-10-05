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

## Ensemble 输入与输出

Ensemble 额外要求 `pair_id` 和 boolean `is_source`。没有合法且唯一 source 的增强记录保留并标记 `invalid_pair_structure`。按 JSONL 非空记录的零起始 `row_position` join，`idx` 可重复；原字段必须与输入逐条一致。保留记录的相对顺序不变。

`ensemble_predictions.jsonl`、`filtered.jsonl`、`removed.jsonl` 都保留原始字段，并增加 `ensemble_quality`：

| 字段 | 含义 |
|---|---|
| `policy` | `conservative_consensus_v1` |
| `row_position` | 原始输入记录位置 |
| `decision` / `reason` | `keep` 或 `drop`，以及具体理由 |
| `eligible_voter_count` | source 预测正确且达到 source 阈值的模型数 |
| `wrong_consensus_label` / `wrong_consensus_count` | 高置信错误票的最大同向共识及票数；无错误票时标签为 null |
| `high_conf_gold_support_count` | 合格且高置信支持增强 gold 的模型数 |
| `supporting_models` | 最大同向错误票的模型 id；KEEP 时也可能存在错误票 |
| `per_model` | source eligibility、source 与增强预测/置信度、错误票及 gold 支持票 |

source 的 reason 始终为 `source_preserved`，per-model 中记录其 source 预测和置信度。其他 KEEP 理由为 `invalid_pair_structure`、`incomplete_model_results`、`insufficient_eligible_voters`、`high_confidence_gold_support` 或 `insufficient_wrong_consensus`；DROP 理由为 `high_confidence_wrong_consensus`。异常结构没有可用 source 证据，per-model 可为空。

`ensemble_summary.json` 保存输入 SHA-256、配置 fingerprint、policy 参数、source/增强数量、保留/删除数量及 removal rate。`by` 分别按 label、mr_type、mr_id、dataset、wrong_consensus_label、decision_reason 和 eligible_voter_count 分组；各组分别统计 source 与增强数量。`source_auditor_health` 包含 accuracy、confidence ≥ source threshold 的 accuracy/coverage 和逐标签 accuracy。空分组 accuracy 为 null。

`threshold_sensitivity` 在固定 source gating 下分别用 0.90、0.95、0.99 重算增强错误票与 gold 支持票，只记录模拟删除统计。默认正式输出始终使用配置中的 0.95。

`model_manifest.json` 按模型保存 name、requested/resolved revision、label order、映射来源、运行环境、输入哈希、单模型 predictions/summary 哈希，以及三份 ensemble JSONL 的哈希。Ensemble summary 不包含运行时钟信息，因此相同输入、配置和预测生成相同 decisions 与 summary。

`summary_all_datasets.json` 包含四数据集总表、全部数据集 summary、汇总计数、MR/gold label removal rate 及三个阈值的敏感性。它只在所有配置数据集成功后发布。
