# Merged MR quality audit

Input 78042; source 34377; augmented 43665.

| Dataset | Input | Kept | Conflict | Model V1.1 | Duplicate |
|---|---:|---:|---:|---:|---:|
| snli | 22548 | 22140 | 42 | 326 | 40 |
| mnlim | 22423 | 21686 | 23 | 298 | 416 |
| mnlimm | 22416 | 21690 | 2 | 310 | 414 |
| sick | 10655 | 10580 | 1 | 67 | 7 |

All source rows retained. Original fields and order retained. Conflict quarantine is a subset of removed rows.

Same-label duplicates retain every source; otherwise the earliest augmentation surviving model filtering is retained.

V1 and V1.1 evidence, historical/new prediction provenance, actual batch sizes, threshold sensitivity and integrity receipts are included.
