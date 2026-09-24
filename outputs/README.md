# 审计输出

`snli_v3_3_cross_encoder_deberta_v3_large/` 是仓库内置 SNLI v3.3 数据使用 `cross-encoder/nli-deberta-v3-large` 生成的完整结果，包括：

- `predictions.jsonl`：23,205 条完整逐行预测；
- `disagreements.jsonl`：1,887 条标签不一致记录；
- `high_confidence_agreements.jsonl`：20,841 条 confidence ≥ 0.90 且标签一致的记录；
- `pair_quality.jsonl`：9,824 个 pair 的组级质量信息；
- `summary.json`：总体与分组统计、运行环境和模型 revision；
- `REPORT.md`：中文结果解读和筛选建议。

`summary.json` 中的绝对路径记录的是首次审计实际发生的位置，用于历史溯源。仓库内对应输入是 `data/snli_v3_3/augmented/snli.jsonl`，其 SHA-256 与 summary 中记录一致。

重新运行时建议写入新目录，避免覆盖这份基线结果：

```bash
python evaluate_nli_quality.py data/snli_v3_3/augmented/snli.jsonl \
  --output-dir outputs/snli_v3_3_rerun \
  --device cuda:0
```
