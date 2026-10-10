# Ensemble filter V1.1 — offline refilter report

No model was loaded or rerun. All evidence comes from persisted V1 predictions.

## Overall

| dataset | augmented | V1 removed | V1.1 removed | rescued by V1.1 | V1.1 removal rate |
|---|---:|---:|---:|---:|---:|
| SNLI | 12724 | 355 | 346 | 9 | 2.7193% |
| MNLIM | 12608 | 445 | 303 | 142 | 2.4032% |
| MNLIMM | 12584 | 464 | 311 | 153 | 2.4714% |
| SICK | 5749 | 125 | 68 | 57 | 1.1828% |
| TOTAL | 43665 | 1389 | 1028 | 361 | 2.3543% |

Source records preserved: 34377; removed source: 0; KEEP→DROP: 0.

## Policy changes

Source gating remains correct source prediction with confidence ≥0.80; same-wrong-label votes remain ≥0.95. Gold-support blocker changes 0.95 → 0.90 (eligible auditors only). SNLI retains ≥3/4. MNLIM/MNLIMM additionally require the actual `alisawuffles/roberta-large-wanli` checkpoint among wrong supporters, resolved by model manifest identity, not a fixed model id. SICK changes 3/4 → unanimous 4/4, with all four source-eligible. Exact `mr_type` OR `mr_id == conditional_clause` disables auto-drop.

Decision priority: source → invalid/incomplete evidence → conditional protection → gold blocker → dataset-specific rule → insufficient consensus.

## Rescued samples (V1 DROP → V1.1 KEEP)

Primary protection reasons are mutually exclusive due to decision priority; counts are not causal ablations.

### By reason

| value | rescued |
|---|---:|
| conditional_clause_auto_drop_disabled | 43 |
| gold_support_block | 21 |
| mnli_wrong_consensus_missing_wanli_anchor | 245 |
| sick_requires_unanimous_consensus | 52 |

### By dataset

| value | rescued |
|---|---:|
| mnlim | 142 |
| mnlimm | 153 |
| sick | 57 |
| snli | 9 |

### By mr_type

| value | rescued |
|---|---:|
| flip | 274 |
| inv | 25 |
| neutral | 62 |

### By mr_id

| value | rescued |
|---|---:|
| adding_contradiction | 234 |
| antonym_substitution | 32 |
| conditional_clause | 43 |
| negation_flip | 8 |
| pronoun_substitution | 3 |
| synonym_replacement | 16 |
| uninformative | 19 |
| voice_switch | 6 |

### By gold_label

| value | rescued |
|---|---:|
| contradiction | 284 |
| entailment | 4 |
| neutral | 73 |

## Remaining removed samples

### By dataset

| value | removed |
|---|---:|
| mnlim | 303 |
| mnlimm | 311 |
| sick | 68 |
| snli | 346 |

### By mr_type

| value | removed |
|---|---:|
| flip | 859 |
| inv | 94 |
| neutral | 75 |

### By mr_id

| value | removed |
|---|---:|
| adding_contradiction | 730 |
| antonym_substitution | 114 |
| negation_flip | 15 |
| pronoun_substitution | 31 |
| synonym_replacement | 50 |
| uninformative | 75 |
| voice_switch | 13 |

### By gold_label

| value | removed |
|---|---:|
| contradiction | 901 |
| entailment | 17 |
| neutral | 110 |

### By wrong_consensus_label

| value | removed |
|---|---:|
| contradiction | 47 |
| entailment | 208 |
| neutral | 773 |

### By consensus_size

| value | removed |
|---|---:|
| 3/4 | 379 |
| 4/4 | 649 |

### Supporting-vote confidence ranges

Ranges cover only auditors in the winning wrong-label consensus, not cross-model averaged probabilities.

| dataset | source min–max | augmented min–max | votes |
|---|---|---|---:|
| snli | 0.80094087–0.99997783 | 0.9500702–0.99996948 | 1212 |
| mnlim | 0.80219316–0.99997091 | 0.95029622–0.99995375 | 1111 |
| mnlimm | 0.80921066–0.99995506 | 0.95039648–0.99996114 | 1138 |
| sick | 0.8067081–0.99990559 | 0.95048952–0.99990153 | 272 |

## Eligibility diagnostics

| dataset | eligible 0 | eligible 1 | eligible 2 | eligible 3 | eligible 4 | eligible <3 |
|---|---:|---:|---:|---:|---:|---:|
| snli | 448 (3.52%) | 519 (4.08%) | 797 (6.26%) | 1712 (13.45%) | 9248 (72.68%) | 1764 |
| mnlim | 632 (5.01%) | 508 (4.03%) | 817 (6.48%) | 2196 (17.42%) | 8455 (67.06%) | 1957 |
| mnlimm | 561 (4.46%) | 506 (4.02%) | 789 (6.27%) | 1864 (14.81%) | 8864 (70.44%) | 1856 |
| sick | 416 (7.24%) | 368 (6.40%) | 273 (4.75%) | 568 (9.88%) | 4124 (71.73%) | 1057 |

KEEP does not mean the ensemble proved the label correct. This filter only deletes records with strong consistent contrary evidence. Confidence is uncalibrated model softmax confidence, not a probability that the label is wrong.

## Wrong-vote threshold sensitivity

Source threshold=0.80 and gold blocker=0.90 stay fixed; anchor, unanimity, and conditional protection stay enabled. Actual filtered files always use wrong threshold=0.95.

| wrong threshold | SNLI | MNLIM | MNLIMM | SICK | TOTAL |
|---|---:|---:|---:|---:|---:|
| 0.90 | 412 | 349 | 366 | 82 | 1209 |
| 0.95 | 346 | 303 | 311 | 68 | 1028 |
| 0.99 | 209 | 171 | 172 | 32 | 584 |

## Conditional-clause diagnostic (not applied)

| dataset | protected | would drop without protection | would drop with 4/4 at 0.99 |
|---|---:|---:|---:|
| snli | 1781 | 3 | 1 |
| mnlim | 1736 | 17 | 3 |
| mnlimm | 1684 | 16 | 4 |
| sick | 739 | 0 | 0 |

## Integrity and offline execution

All 132 V1 files were checked before/after: file count, SHA-256, size, and mtime_ns unchanged. Original data files were checked the same way. Detailed receipts: `source_v1_integrity` and `original_data_integrity` in `summary_all_datasets.json`.

The refilter imports only Python standard-library modules; it never imports the evaluator, audit runner, torch, transformers, or any model loader. Individual model JSONL files are joined only if ensemble prediction fields are missing. Per-dataset summaries record any offline join model ids.

## Artifacts and reproduction

Each dataset contains `ensemble_predictions.jsonl`, `filtered.jsonl`, `removed.jsonl`, `rescued_from_v1.jsonl`, `ensemble_summary.json`, `policy_diff_from_v1.json`, and a provenance `model_manifest.json`. Raw fields and the complete V1 `ensemble_quality` remain unchanged; only `ensemble_quality_v1_1` is added. All subsets preserve original order.

Use `outputs/ensemble_filter_v1_1/{snli,mnlim,mnlimm,sick}/filtered.jsonl` preferentially for future experiments; V1 remains available for policy comparisons. Root `diff_v1_vs_v1_1.json` contains per-dataset and TOTAL transitions and rescued groups.

```bash
# Existing output roots are refused; use a fresh directory to reproduce.
python refilter_ensemble_v1_1.py --output-root outputs/ensemble_filter_v1_1_reproduced
python -m unittest discover -s tests -v
```
