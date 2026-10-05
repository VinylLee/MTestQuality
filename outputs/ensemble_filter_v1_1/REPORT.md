# Ensemble filter V1.1 — offline refilter report

No model was loaded or rerun. All evidence comes from persisted V1 predictions.

## Overall

| dataset | augmented | V1 removed | V1.1 removed | rescued by V1.1 | V1.1 removal rate |
|---|---:|---:|---:|---:|---:|
| SNLI | 13381 | 59 | 53 | 6 | 0.3961% |
| MNLIM | 10434 | 68 | 38 | 30 | 0.3642% |
| MNLIMM | 10425 | 53 | 32 | 21 | 0.3070% |
| SICK | 6019 | 10 | 4 | 6 | 0.0665% |
| TOTAL | 40259 | 190 | 127 | 63 | 0.3155% |

Source records preserved: 34377; removed source: 0; KEEP→DROP: 0.

## Policy changes

Source gating remains correct source prediction with confidence ≥0.80; same-wrong-label votes remain ≥0.95. Gold-support blocker changes 0.95 → 0.90 (eligible auditors only). SNLI retains ≥3/4. MNLIM/MNLIMM additionally require the actual `alisawuffles/roberta-large-wanli` checkpoint among wrong supporters, resolved by model manifest identity, not a fixed model id. SICK changes 3/4 → unanimous 4/4, with all four source-eligible. Exact `mr_type` OR `mr_id == conditional_clause` disables auto-drop.

Decision priority: source → invalid/incomplete evidence → conditional protection → gold blocker → dataset-specific rule → insufficient consensus.

## Rescued samples (V1 DROP → V1.1 KEEP)

Primary protection reasons are mutually exclusive due to decision priority; counts are not causal ablations.

### By reason

| value | rescued |
|---|---:|
| conditional_clause_auto_drop_disabled | 20 |
| gold_support_block | 4 |
| mnli_wrong_consensus_missing_wanli_anchor | 34 |
| sick_requires_unanimous_consensus | 5 |

### By dataset

| value | rescued |
|---|---:|
| mnlim | 30 |
| mnlimm | 21 |
| sick | 6 |
| snli | 6 |

### By mr_type

| value | rescued |
|---|---:|
| flip | 16 |
| inv | 9 |
| neutral | 38 |

### By mr_id

| value | rescued |
|---|---:|
| adding_contradiction | 10 |
| composite_flip | 1 |
| composite_neutral | 3 |
| conditional_clause | 20 |
| negation_flip | 5 |
| pronoun_substitution | 7 |
| synonym_replacement | 2 |
| uninformative | 15 |

### By gold_label

| value | rescued |
|---|---:|
| contradiction | 18 |
| entailment | 7 |
| neutral | 38 |

## Remaining removed samples

### By dataset

| value | removed |
|---|---:|
| mnlim | 38 |
| mnlimm | 32 |
| sick | 4 |
| snli | 53 |

### By mr_type

| value | removed |
|---|---:|
| flip | 49 |
| inv | 21 |
| neutral | 57 |

### By mr_id

| value | removed |
|---|---:|
| adding_contradiction | 15 |
| composite_flip | 7 |
| composite_inv | 3 |
| composite_neutral | 5 |
| negation_flip | 27 |
| pronoun_substitution | 6 |
| synonym_replacement | 11 |
| uninformative | 52 |
| voice_switch | 1 |

### By gold_label

| value | removed |
|---|---:|
| contradiction | 63 |
| entailment | 7 |
| neutral | 57 |

### By wrong_consensus_label

| value | removed |
|---|---:|
| contradiction | 17 |
| entailment | 90 |
| neutral | 20 |

### By consensus_size

| value | removed |
|---|---:|
| 3/4 | 50 |
| 4/4 | 77 |

### Supporting-vote confidence ranges

Ranges cover only auditors in the winning wrong-label consensus, not cross-model averaged probabilities.

| dataset | source min–max | augmented min–max | votes |
|---|---|---|---:|
| snli | 0.82382631–0.99992752 | 0.95379925–0.99991989 | 186 |
| mnlim | 0.8326748–0.99996591 | 0.95153165–0.99920398 | 137 |
| mnlimm | 0.88200569–0.99995637 | 0.95024115–0.9999069 | 119 |
| sick | 0.98576713–0.99994063 | 0.95711124–0.99821103 | 16 |

## Eligibility diagnostics

| dataset | eligible 0 | eligible 1 | eligible 2 | eligible 3 | eligible 4 | eligible <3 |
|---|---:|---:|---:|---:|---:|---:|
| snli | 400 (2.99%) | 418 (3.12%) | 802 (5.99%) | 1672 (12.50%) | 10089 (75.40%) | 1620 |
| mnlim | 466 (4.47%) | 405 (3.88%) | 619 (5.93%) | 1728 (16.56%) | 7216 (69.16%) | 1490 |
| mnlimm | 446 (4.28%) | 407 (3.90%) | 595 (5.71%) | 1466 (14.06%) | 7511 (72.05%) | 1448 |
| sick | 194 (3.22%) | 164 (2.72%) | 204 (3.39%) | 490 (8.14%) | 4967 (82.52%) | 562 |

KEEP does not mean the ensemble proved the label correct. This filter only deletes records with strong consistent contrary evidence. Confidence is uncalibrated model softmax confidence, not a probability that the label is wrong.

## Wrong-vote threshold sensitivity

Source threshold=0.80 and gold blocker=0.90 stay fixed; anchor, unanimity, and conditional protection stay enabled. Actual filtered files always use wrong threshold=0.95.

| wrong threshold | SNLI | MNLIM | MNLIMM | SICK | TOTAL |
|---|---:|---:|---:|---:|---:|
| 0.90 | 64 | 49 | 41 | 5 | 159 |
| 0.95 | 53 | 38 | 32 | 4 | 127 |
| 0.99 | 30 | 17 | 11 | 0 | 58 |

## Conditional-clause diagnostic (not applied)

| dataset | protected | would drop without protection | would drop with 4/4 at 0.99 |
|---|---:|---:|---:|
| snli | 2546 | 4 | 0 |
| mnlim | 2689 | 4 | 1 |
| mnlimm | 2710 | 1 | 0 |
| sick | 1216 | 0 | 0 |

## Integrity and offline execution

All 161 V1 files were checked before/after: file count, SHA-256, size, and mtime_ns unchanged. Original data files were checked the same way. Detailed receipts: `source_v1_integrity` and `original_data_integrity` in `summary_all_datasets.json`.

The refilter imports only Python standard-library modules; it never imports the evaluator, audit runner, torch, transformers, or any model loader. Individual model JSONL files are joined only if ensemble prediction fields are missing. Per-dataset summaries record any offline join model ids.

## Artifacts and reproduction

Each dataset contains `ensemble_predictions.jsonl`, `filtered.jsonl`, `removed.jsonl`, `rescued_from_v1.jsonl`, `ensemble_summary.json`, `policy_diff_from_v1.json`, and a provenance `model_manifest.json`. Raw fields and the complete V1 `ensemble_quality` remain unchanged; only `ensemble_quality_v1_1` is added. All subsets preserve original order.

Use `outputs/ensemble_filter_v1_1/{snli,mnlim,mnlimm,sick}/filtered.jsonl` preferentially for future experiments; V1 remains available for policy comparisons. Root `diff_v1_vs_v1_1.json` contains per-dataset and TOTAL transitions and rescued groups.

```bash
# Existing output roots are refused; use a fresh directory to reproduce.
python refilter_ensemble_v1_1.py --output-root outputs/ensemble_filter_v1_1_reproduced
python -m unittest discover -s tests -v
```
