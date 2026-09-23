# SNLI 增强数据 NLI 质量审计

## 结论

使用 `cross-encoder/nli-deberta-v3-large` 对 23,205 条 premise-hypothesis 进行三分类复核。该模型在 SNLI + MultiNLI 上训练，模型卡报告 SNLI test accuracy 为 92.20%。本次 source 数据一致率同样为 92.20%，说明标签映射和推理实现与模型预期吻合。

- 全部数据：21,318 / 23,205 一致，**91.87%**。
- source：9,058 / 9,824 一致，**92.20%**。
- 增强数据：12,260 / 13,381 一致，**91.62%**。
- 增强数据只比 source 低 **0.58 个百分点**，整体质量良好。
- 最大问题是 `conditional_clause`（76.87%），其次是 `adding_contradiction`（87.00%）。
- 其他增强 MR 的一致率为 92.41%–98.18%。

模型使用精确 revision `bab4bc7178836f731dcfd18c06ca9def0a137712`，输入文件 SHA-256 为 `acb67f77853b4d63329a85cec8cbbcb0a878d453bd69d08f6754169918a87e53`。所有 23,205 条输入均未达到 256-token 截断上限。

## 按 MR 统计

`≥0.90 且一致`是可直接作为宽松候选池的条数，不代表人工确认后的最终正确数。

| MR | 条数 | 一致数 | 一致率 | 平均置信度 | ≥0.90 且一致 |
|---|---:|---:|---:|---:|---:|
| conditional_clause | 2,546 | 1,957 | 76.87% | 94.61% | 1,778 |
| adding_contradiction | 877 | 763 | 87.00% | 97.16% | 741 |
| composite_inv | 145 | 134 | 92.41% | 98.25% | 131 |
| synonym_replacement | 183 | 170 | 92.90% | 98.75% | 168 |
| pronoun_substitution | 1,600 | 1,493 | 93.31% | 97.00% | 1,431 |
| uninformative | 3,285 | 3,127 | 95.19% | 99.34% | 3,102 |
| antonym_substitution | 67 | 64 | 95.52% | 98.95% | 63 |
| composite_neutral | 1,002 | 958 | 95.61% | 99.38% | 949 |
| negation_flip | 2,279 | 2,225 | 97.63% | 99.63% | 2,220 |
| composite_flip | 1,177 | 1,153 | 97.96% | 99.56% | 1,149 |
| voice_switch | 220 | 216 | 98.18% | 99.42% | 216 |

按 MR 类型，`flip` 为 95.57%，`inv`（包含 source）为 92.47%，`neutral` 为 88.42%。neutral 偏低主要由 `conditional_clause` 拉低。

## 相对 source 的变化

- source 被模型判对的增强记录有 12,521 条，其中增强后仍一致 11,567 条（92.38%）。
- source 被模型判错的增强记录有 860 条，其中增强后判对 693 条（80.58%）。
- 在 source 原本一致时，引入新分歧最多的是 `conditional_clause`（547 条）、`uninformative`（133 条）和 `adding_contradiction`（107 条）。
- `voice_switch` 没有在 source 原本一致的情况下引入新分歧；但它只有 220 条，样本量较小。

## 置信度与候选池

| 阈值 | 所有高置信预测 | 其中一致 | 高置信预测的一致率 | 可保留增强数据（高置信且一致） |
|---:|---:|---:|---:|---:|
| ≥0.90 | 22,273 | 20,841 | 93.57% | 11,948 |
| ≥0.95 | 21,532 | 20,354 | 94.53% | 11,641 |
| ≥0.99 | 18,468 | 17,929 | 97.08% | 10,255 |

推荐后续构建测试集时：

1. 宽松候选池使用“标签一致且 confidence ≥ 0.90”；严格候选池使用 ≥ 0.99。
2. `conditional_clause` 单独分层抽样或人工复核，不要与其他 MR 使用同一阈值后直接混合。
3. 按 `pair_id` 划分 train/dev/test，避免同一 source 及其增强变体跨集合泄漏。
4. 从高置信一致、低置信、以及高置信不一致三组分别人工抽样，估计真正 precision；不要把 softmax confidence 当作已校准的正确概率。
5. 若测试集用于衡量其他模型，保留各 label/MR 的目标配比，避免阈值筛选造成 easy-example 偏置。

## 输出文件

- `predictions.jsonl`：完整 23,205 条。保留所有原字段，新增 `nli_quality`；这是后续自定义筛选的主文件。
- `high_confidence_agreements.jsonl`：本次阈值 0.90 下的一致记录，共 20,841 条（含 source）。
- `disagreements.jsonl`：1,887 条不一致记录，适合人工复核和错误分析。
- `pair_quality.jsonl`：9,824 个 `pair_id` 的组级信息，可用于整组筛选和防止切分泄漏。
- `summary.json`：总体、source/增强、label、MR type、MR id、置信度阈值、混淆矩阵、source 条件分析和运行溯源。

每条 `nli_quality` 包含：期望标签、模型预测标签、三类概率、最大概率 confidence、top-1/top-2 margin、gold probability、gold rank、negative log-likelihood、normalized entropy、token 数、是否达到截断上限、是否一致。

## 限制

这是单一模型的自动审计，不是人工金标。该模型在 SNLI 上训练，适合作为本次用户指定的 in-domain 强筛选器，但其错误模式会影响筛选结果。检查高置信分歧可见两种情况同时存在：部分增强文本确有明显关系错误；部分清晰样本也会被模型误判。因此不建议自动删除全部分歧项，特别是计划构造高可信测试集时，应再做人审或第二模型交叉验证。
