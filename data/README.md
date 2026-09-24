# 内置数据

本目录保存可直接交给 `evaluate_nli_quality.py` 的数据，不依赖 MTrain 中的外部路径。

## SNLI v3.3

| 文件 | 记录数 | SHA-256 | 说明 |
|---|---:|---|---|
| `snli_v3_3/source/snli.jsonl` | 9,824 | `22e815c868a5377317ea557d2d2da7ec2fe912895c8c5379e0cf8fd1c5c5f241` | 增强前 source 数据 |
| `snli_v3_3/augmented/snli.jsonl` | 23,205 | `acb67f77853b4d63329a85cec8cbbcb0a878d453bd69d08f6754169918a87e53` | source 与 13,381 条增强记录组成的完整审计输入 |
| `snli_v3_3/augmented/snli.report.json` | — | `82fdc71a6a69b518faf90935d1d9e9ac98eb2534f5315bb224b538c2bf921ab7` | 增强过程配置与统计报告 |

## MNLI-mismatched v3.3

`mnlimm` 对应 MultiNLI 1.0 的 `validation_mismatched` split。

| 文件 | 记录数 | SHA-256 | 说明 |
|---|---:|---|---|
| `mnlimm_v3_3/source/mnlimm.jsonl` | 9,832 | `beec9e0786474e6a00fa3dad90365c8b662094b85bf68d735e00aed9e21079c7` | 增强前 source 数据 |
| `mnlimm_v3_3/augmented/mnlimm.jsonl` | 20,257 | `b1848583ab00e6a0cbdbfde85e51e86e429e83df3976fb5bfab22d42e0f125d1` | source 与 10,425 条增强记录组成的完整待审计输入 |
| `mnlimm_v3_3/augmented/mnlimm.report.json` | — | `50d5ee03eb07e31362021ccb06d4c33df851f7ba9a72bb1aee0aed499a233bad` | 状态为 `complete` 的增强过程配置与统计报告 |

标签数字映射为 `0=entailment`、`1=neutral`、`2=contradiction`。

## 数据关系

- 每个数据集的 `source/*.jsonl` 都与对应 `augmented/*.jsonl` 中的 `is_source=true` 子集逐条一致。
- `pair_id` 将一条 source 与其生成变体关联起来。
- `mr_id` 和 `mr_type` 描述增强操作；部分 composite 记录还包含有序的 `component_mrs`。
- 当前完整审计结果位于 `../outputs/snli_v3_3_cross_encoder_deberta_v3_large/`。

## 完整性检查

```bash
sha256sum data/snli_v3_3/source/snli.jsonl \
  data/snli_v3_3/augmented/snli.jsonl \
  data/snli_v3_3/augmented/snli.report.json \
  data/mnlimm_v3_3/source/mnlimm.jsonl \
  data/mnlimm_v3_3/augmented/mnlimm.jsonl \
  data/mnlimm_v3_3/augmented/mnlimm.report.json

wc -l data/snli_v3_3/source/snli.jsonl \
  data/snli_v3_3/augmented/snli.jsonl \
  data/mnlimm_v3_3/source/mnlimm.jsonl \
  data/mnlimm_v3_3/augmented/mnlimm.jsonl
```
