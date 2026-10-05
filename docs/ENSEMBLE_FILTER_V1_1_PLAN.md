# MTestQuality Ensemble Filter V1.1 增量修改方案

> 目标：在 **不重新运行 4 个 NLI 模型、不修改原始数据、不覆盖 V1 结果** 的前提下，基于现有 `outputs/ensemble_filter_v1/` 中已经保存的预测结果，对过滤策略进行一次保守升级，降低误删风险。  
> 适用数据集：**SNLI、MNLIM、MNLIMM、SICK**。  
> 本文档是对 `MULTI_DATASET_ENSEMBLE_FILTERING_PLAN.md` 的 **V1.1 增量规范**。未在本文档中明确修改的 V1 规则全部继续保持不变。

---

## 1. V1.1 的目标

V1 已经完成以下工作：

- 4 个 NLI auditor 已经全部运行；
- source prediction、follow-up prediction、confidence 已经保存；
- ensemble decision、removed、filtered、summary 已生成；
- 输入 hash、模型 revision、label mapping 等均已记录；
- V1 不允许被覆盖。

V1.1 不再做新的模型推理。

本次只做：

```text
读取 V1 已有预测结果
        ↓
应用 V1.1 新过滤策略
        ↓
重新生成 V1.1 decisions
        ↓
输出到新的 ensemble_filter_v1_1 目录
```

因此：

> **禁止重新下载、重新加载或重新运行任何 NLI 模型。**

---

# 2. V1.1 的核心变化

V1.1 相比 V1 有 4 个主要变化。

---

## Change 1：Gold-support blocker 从 0.95 降到 0.90

V1：

```text
high_conf_gold_support_threshold = 0.95
```

V1.1：

```text
GOLD_SUPPORT_BLOCK_THRESHOLD = 0.90
```

删除证据本身仍然要求：

```text
WRONG_VOTE_THRESHOLD = 0.95
```

也就是说：

- 要删除一条记录，需要多个 auditor 以 `>= 0.95` 高置信预测相同错误标签；
- 但只要存在一个 eligible auditor 以 `>= 0.90` 支持增强标签，就阻止自动删除。

这是一个非对称规则：

```text
DROP requires strong evidence.
KEEP only needs meaningful counter-evidence.
```

### 定义

```text
gold_support_block(m, a) =
    source_eligible(m)
    AND
    augmented_prediction(m) == augmented_gold_label
    AND
    augmented_confidence(m) >= 0.90
```

若：

```text
gold_support_block_count >= 1
```

则：

```text
decision = KEEP
reason = gold_support_block
```

---

# 3. Source gating 保持不变

V1.1 继续使用 V1 的 source gating：

```text
source_eligible(m) =
    source_prediction(m) == source_gold_label
    AND
    source_confidence(m) >= 0.80
```

参数保持：

```text
SOURCE_CONFIDENCE_THRESHOLD = 0.80
```

任何未通过 source gating 的模型：

```text
不得参与 wrong vote
不得参与 gold-support blocker
```

---

# 4. Wrong vote 定义保持不变

```text
high_conf_wrong_vote(m, a) =
    source_eligible(m)
    AND
    augmented_prediction(m) != augmented_gold_label
    AND
    augmented_confidence(m) >= 0.95
```

参数：

```text
WRONG_VOTE_THRESHOLD = 0.95
```

错误票必须按：

```text
predicted wrong label
```

分组。

不能把：

```text
neutral
contradiction
```

混为同一个“反对 gold”的投票。

---

# 5. SNLI 的 V1.1 规则

SNLI 保持最接近 V1。

对于：

```text
dataset == "snli"
```

自动 DROP 必须满足：

```text
1. is_source != true

2. mr_type / mr_id 不属于 conditional_clause

3. eligible_voter_count >= 3

4. 至少 3 个 eligible auditor
   以 >= 0.95 confidence
   预测同一个错误 label

5. wrong_consensus_count >= 3

6. gold_support_block_count == 0
   即：不存在 eligible auditor
   以 >= 0.90 confidence 支持增强 gold label
```

形式化：

```text
SNLI_DROP(a) iff

not source(a)

AND not conditional_clause(a)

AND eligible_count(a) >= 3

AND max_y!=gold count({
    m |
    source_eligible(m)
    AND pred_m(a) = y
    AND conf_m(a) >= 0.95
}) >= 3

AND count({
    m |
    source_eligible(m)
    AND pred_m(a) = gold
    AND conf_m(a) >= 0.90
}) = 0
```

---

# 6. MNLIM / MNLIMM 的 V1.1 规则

MNLI matched 和 mismatched 增加一个额外保护：

> **自动删除的错误共识中，必须包含 WANLI-only auditor。**

在 V1 的 4 个模型中，将：

```text
alisawuffles/roberta-large-wanli
```

定义为：

```text
OUT_OF_MNLI_ANCHOR
```

推荐配置别名：

```text
model_c
```

但实现中不要只依赖 `model_c` 字符串；应通过真实模型名称匹配：

```text
alisawuffles/roberta-large-wanli
```

---

## 6.1 MNLI DROP 条件

对于：

```text
dataset in {"mnlim", "mnlimm"}
```

必须同时满足：

```text
1. is_source != true

2. 不是 conditional_clause

3. eligible_voter_count >= 3

4. wrong_consensus_count >= 3

5. >=3 个 auditor 预测同一个错误 label

6. 所有 supporting wrong voters 的
   follow-up confidence >= 0.95

7. supporting wrong voters 中必须包含：
   alisawuffles/roberta-large-wanli

8. gold_support_block_count == 0
   即不存在 eligible auditor
   以 >=0.90 confidence 支持 gold
```

形式化：

```text
MNLI_DROP(a) iff

not source(a)

AND not conditional_clause(a)

AND eligible_count(a) >= 3

AND exists wrong_label y where
    wrong_supporters(y, a) >= 3

AND WANLI_ONLY_MODEL in wrong_supporters(y, a)

AND gold_support_block_count(a) == 0
```

---

## 6.2 为什么需要 WANLI anchor

以下 3 个 auditor 使用过 SNLI / MNLI 或明显依赖 MNLI fine-tuning：

```text
cross-encoder/nli-deberta-v3-large

MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli

facebook/bart-large-mnli
```

因此在 MNLI 数据上：

```text
A + B + D
```

形成的 3/4 共识不能视为完全独立的外部证据。

V1.1 要求：

```text
alisawuffles/roberta-large-wanli
```

必须参与错误共识，用来降低 shared-MNLI bias 的风险。

---

# 7. SICK 的 V1.1 规则

SICK 改为更保守的：

> **4/4 unanimous consensus**

对于：

```text
dataset == "sick"
```

自动 DROP 必须满足：

```text
1. is_source != true

2. 不是 conditional_clause

3. 四个 auditor 全部通过 source gating

4. 四个 auditor 全部预测同一个错误 label

5. 四个 auditor 的 follow-up confidence 都 >= 0.95

6. 不存在 gold-support blocker
```

由于第 3～5 条已经意味着 4 个模型都高置信支持同一个错误标签，因此 gold-support blocker 理论上不可能成立，但仍保留统一逻辑和日志字段。

形式化：

```text
SICK_DROP(a) iff

not source(a)

AND not conditional_clause(a)

AND eligible_count(a) == 4

AND exists wrong_label y such that

count({
    m |
    source_eligible(m)
    AND pred_m(a) = y
    AND conf_m(a) >= 0.95
}) == 4

AND gold_support_block_count(a) == 0
```

---

# 8. conditional_clause：V1.1 禁止自动删除

这是 V1.1 最明确的规则之一。

对于四个数据集中的任何：

```text
mr_type == "conditional_clause"
```

或项目实际字段中等价的：

```text
mr_id == "conditional_clause"
```

记录：

```text
decision = KEEP
reason = conditional_clause_auto_drop_disabled
```

即使：

```text
4/4 模型全部以 0.999 预测相同错误标签
```

也：

```text
KEEP
```

---

## 8.1 匹配字段要求

Codex 必须先检查项目实际增强记录如何保存 MR 名称。

优先匹配：

```text
mr_type
```

如果实际使用：

```text
mr_id
```

则使用 `mr_id`。

如果两者都存在：

```text
任一字段值 == "conditional_clause"
```

即可触发保护。

不要通过字符串模糊匹配，例如：

```text
"conditional" in mr_name
```

除非现有数据中确实没有精确字段值，并在代码注释中说明原因。

---

# 9. V1.1 决策优先级

为了避免不同保护条件冲突，必须按以下顺序决定。

```text
Priority 1:
is_source == true
→ KEEP
reason = source_never_filtered

Priority 2:
pair structure invalid / predictions incomplete
→ KEEP
reason = invalid_or_incomplete_evidence

Priority 3:
conditional_clause
→ KEEP
reason = conditional_clause_auto_drop_disabled

Priority 4:
gold_support_block_count >= 1
→ KEEP
reason = gold_support_block

Priority 5:
apply dataset-specific DROP rule
→ DROP or KEEP

Priority 6:
otherwise
→ KEEP
reason = insufficient_wrong_consensus
```

注意：

> gold-support blocker 必须在最终 DROP 判断之前执行。

---

# 10. Dataset-specific policy summary

最终规则汇总：

| Dataset | Wrong consensus | Anchor requirement | Gold blocker | Conditional clause |
|---|---:|---|---:|---|
| SNLI | >=3/4 | 无 | >=0.90 → KEEP | 永远 KEEP |
| MNLIM | >=3/4 | 必须包含 WANLI-only | >=0.90 → KEEP | 永远 KEEP |
| MNLIMM | >=3/4 | 必须包含 WANLI-only | >=0.90 → KEEP | 永远 KEEP |
| SICK | **4/4** | 自动满足 | >=0.90 → KEEP | 永远 KEEP |

统一参数：

```text
SOURCE_CONFIDENCE_THRESHOLD = 0.80
WRONG_VOTE_THRESHOLD = 0.95
GOLD_SUPPORT_BLOCK_THRESHOLD = 0.90
```

---

# 11. 不重新运行模型

V1.1 是 **post-hoc refilter**。

禁止执行：

```text
AutoModelForSequenceClassification.from_pretrained(...)
AutoTokenizer.from_pretrained(...)
model(...)
pipeline(...)
```

禁止调用任何会重新跑 inference 的代码路径。

必须直接读取 V1 已保存的：

```text
predictions
source predictions
augmented predictions
confidence
model identity
pair_id
MR metadata
```

如果当前 V1 输出已经有：

```text
ensemble_predictions.jsonl
```

且里面包含全部 per-model prediction + confidence + source_eligible 所需信息，则优先直接读取该文件。

只有在 `ensemble_predictions.jsonl` 缺少必要字段时，才允许读取：

```text
outputs/ensemble_filter_v1/<dataset>/model_a/...
outputs/ensemble_filter_v1/<dataset>/model_b/...
outputs/ensemble_filter_v1/<dataset>/model_c/...
outputs/ensemble_filter_v1/<dataset>/model_d/...
```

进行离线 join。

---

# 12. V1.1 输出目录

禁止覆盖：

```text
outputs/ensemble_filter_v1/
```

新建：

```text
outputs/ensemble_filter_v1_1/
```

目录：

```text
outputs/
  ensemble_filter_v1/
    ...

  ensemble_filter_v1_1/
    snli/
      ensemble_predictions.jsonl
      filtered.jsonl
      removed.jsonl
      ensemble_summary.json
      policy_diff_from_v1.json

    mnlim/
      ...

    mnlimm/
      ...

    sick/
      ...

    REPORT.md
    summary_all_datasets.json
    diff_v1_vs_v1_1.json
```

---

# 13. 不修改任何 V1 文件

运行前与运行后：

```text
outputs/ensemble_filter_v1/
```

中的：

```text
file count
file size
SHA-256
mtime（如容易保存）
```

不得因 V1.1 执行发生内容变化。

至少应对这些关键文件计算 SHA-256：

```text
ensemble_predictions.jsonl
filtered.jsonl
removed.jsonl
ensemble_summary.json
```

V1.1 summary 中保存：

```json
{
  "source_v1_integrity": {
    "verified": true,
    "files": {
      "...": {
        "sha256_before": "...",
        "sha256_after": "..."
      }
    }
  }
}
```

如果 hash 改变：

```text
V1.1 task fails
```

---

# 14. V1.1 ensemble_predictions 输出字段

在 V1 原有字段基础上增加：

```json
{
  "ensemble_quality_v1_1": {
    "policy": "conservative_consensus_v1_1",
    "decision": "keep",
    "reason": "gold_support_block",

    "dataset": "sick",

    "source_confidence_threshold": 0.80,
    "wrong_vote_threshold": 0.95,
    "gold_support_block_threshold": 0.90,

    "eligible_voter_count": 4,

    "wrong_consensus_label": "neutral",
    "wrong_consensus_count": 3,

    "gold_support_block_count": 1,

    "wrong_supporting_models": [
      "..."
    ],

    "gold_supporting_models": [
      "..."
    ],

    "wanli_anchor_required": false,
    "wanli_anchor_present_in_wrong_consensus": false,

    "conditional_clause_protected": false,

    "v1_decision": "drop",
    "v1_1_decision": "keep",
    "decision_changed": true
  }
}
```

注意：

```text
v1_decision
```

必须从 V1 结果读取，不要重新模拟。

---

# 15. V1.1 filtered.jsonl

包含：

```text
所有 source
+
所有 V1.1 decision == KEEP 的增强记录
```

必须：

- 保持原始输入顺序；
- 不修改原始字段；
- 可以附加 `ensemble_quality_v1_1`；
- 不允许丢掉 source。

---

# 16. V1.1 removed.jsonl

仅包含：

```text
V1.1 decision == DROP
```

必须带：

```text
ensemble_quality_v1_1
```

并确保：

```text
removed 中 source count == 0
```

---

# 17. policy_diff_from_v1.json

每个数据集生成：

```text
policy_diff_from_v1.json
```

至少统计：

```json
{
  "dataset": "sick",

  "v1_removed": 10,
  "v1_1_removed": 4,

  "drop_to_keep": 6,
  "keep_to_drop": 0,

  "unchanged_drop": 4,
  "unchanged_keep": 0,

  "changes_by_reason": {
    "gold_support_block": 2,
    "sick_requires_unanimous_4_of_4": 3,
    "conditional_clause_auto_drop_disabled": 1
  }
}
```

具体数字必须由程序计算，禁止写死。

---

# 18. diff_v1_vs_v1_1.json

跨数据集生成：

```text
outputs/ensemble_filter_v1_1/diff_v1_vs_v1_1.json
```

包含：

```text
SNLI
MNLIM
MNLIMM
SICK
TOTAL
```

至少统计：

```text
v1_removed
v1_1_removed
drop_to_keep
keep_to_drop
unchanged_drop
unchanged_keep
```

以及：

```text
按 protection reason
按 MR
按 gold label
```

统计被 V1.1 救回的样本。

---

# 19. V1.1 REPORT.md

必须新增：

```text
outputs/ensemble_filter_v1_1/REPORT.md
```

报告至少包含以下部分。

---

## 19.1 Overall

表格：

| dataset | augmented | V1 removed | V1.1 removed | rescued by V1.1 | V1.1 removal rate |
|---|---:|---:|---:|---:|---:|

---

## 19.2 Policy changes

明确说明：

```text
Gold blocker: 0.95 -> 0.90
SICK: 3/4 -> 4/4
MNLI: wrong consensus must include WANLI-only model
conditional_clause: auto-drop disabled
```

---

## 19.3 Rescued samples

统计：

```text
V1 DROP -> V1.1 KEEP
```

按原因：

```text
gold_support_block
conditional_clause protection
missing WANLI anchor
SICK not unanimous
```

---

## 19.4 Remaining removed samples

对 V1.1 最终仍被删除的样本统计：

```text
dataset
MR
gold label
wrong consensus label
3/4 or 4/4
confidence ranges
```

---

## 19.5 Eligibility diagnostics

每个数据集报告：

```text
eligible_count = 0
eligible_count = 1
eligible_count = 2
eligible_count = 3
eligible_count = 4
```

的增强记录数量和比例。

特别报告：

```text
eligible_count < 3
```

数量。

注意报告措辞：

> 被保留的样本并不等价于已经被 ensemble 证明标签正确。  
> 本过滤器仅在存在强一致错误证据时删除数据。

---

# 20. 阈值敏感性

V1.1 仍保留 sensitivity analysis。

但不要重新 inference。

在已有 probabilities 上计算。

至少报告：

```text
WRONG_VOTE_THRESHOLD
0.90
0.95
0.99
```

与此同时：

```text
GOLD_SUPPORT_BLOCK_THRESHOLD
```

正式 V1.1 固定为：

```text
0.90
```

如果实现方便，可额外统计：

```text
gold blocker = 0.90
gold blocker = 0.95
```

但：

> 实际 `filtered.jsonl` 必须固定使用 V1.1 正式参数，不得根据 sensitivity 动态改变。

---

# 21. Conditional-clause sensitivity

因为 V1.1 正式禁止删除 conditional clause，可额外统计：

```text
if conditional protection were disabled:
    how many would be dropped?
```

只作为 REPORT 诊断信息。

不要实际删除。

推荐字段：

```json
{
  "conditional_clause_diagnostic": {
    "protected_count": 0,
    "would_drop_under_standard_policy": 0,
    "would_drop_under_4_of_4_099": 0
  }
}
```

---

# 22. 可选：输出 rescued.jsonl

推荐额外生成：

```text
rescued_from_v1.jsonl
```

包含：

```text
V1 decision = DROP
V1.1 decision = KEEP
```

这样后续可以非常方便地人工 spot-check。

每条需要包含：

```text
dataset
pair_id
mr_type / mr_id
premise
hypothesis
gold label
V1 decision
V1.1 reason
per-model predictions
per-model confidence
```

此文件推荐实现，但不是强制验收项。

---

# 23. 自动测试

在 V1 已有测试基础上增加以下 V1.1 tests。

---

## Test V1.1-1：gold blocker 0.90

例如：

```text
gold = contradiction

A -> neutral       0.99
B -> neutral       0.98
C -> neutral       0.97
D -> contradiction 0.91
```

四个模型都通过 source gating。

结果必须：

```text
KEEP
reason = gold_support_block
```

---

## Test V1.1-2：0.89 不触发 gold blocker

```text
D -> gold label 0.89
```

不能触发 blocker。

若其他规则满足：

```text
DROP
```

---

## Test V1.1-3：SNLI 普通 3/4 仍可 DROP

```text
dataset=snli
not conditional_clause
A/B/C -> same wrong label >=0.95
D no high-conf gold support
```

必须：

```text
DROP
```

---

## Test V1.1-4：MNLIM A+B+D 共识但没有 WANLI model

例如：

```text
A -> neutral 0.99
B -> neutral 0.99
D -> neutral 0.99
C -> gold / low confidence / ineligible
```

即使形成 3/4：

```text
KEEP
reason = mnli_wrong_consensus_missing_wanli_anchor
```

---

## Test V1.1-5：MNLIM A+B+C 共识

其中 C 为：

```text
alisawuffles/roberta-large-wanli
```

且三者同一错误标签 >=0.95，没有 gold blocker：

```text
DROP
```

---

## Test V1.1-6：MNLIMM 同样要求 WANLI anchor

同 Test 4 / Test 5。

---

## Test V1.1-7：SICK 3/4 不删除

```text
dataset=sick
wrong_consensus_count=3
```

无论 confidence 多高：

```text
KEEP
reason = sick_requires_unanimous_consensus
```

---

## Test V1.1-8：SICK 4/4

四个 eligible auditor：

```text
same wrong label
all confidence >=0.95
```

则：

```text
DROP
```

---

## Test V1.1-9：conditional_clause 永远 KEEP

对所有 dataset：

```text
mr_type=conditional_clause
```

即使 4/4 高置信错误共识：

```text
KEEP
reason = conditional_clause_auto_drop_disabled
```

---

## Test V1.1-10：source 永不删除

保持 V1 测试。

---

## Test V1.1-11：V1 不被修改

运行 V1.1 前后关键 V1 文件 hash 完全一致。

---

## Test V1.1-12：不允许 inference

V1.1 主流程测试时 mock / monkeypatch 模型加载函数。

任何：

```text
from_pretrained
pipeline
model.forward
```

被调用都应导致测试失败。

如果当前代码结构不便 mock，可以通过 V1.1 独立脚本不 import inference module 的方式保证。

---

## Test V1.1-13：输出守恒

每个数据集：

```text
len(filtered_v1_1) + len(removed_v1_1)
==
len(original ensemble input)
```

---

## Test V1.1-14：decision diff 正确

必须：

```text
drop_to_keep
+
unchanged_drop
==
v1_removed
```

以及：

```text
keep_to_drop
+
unchanged_keep
==
v1_kept
```

---

# 24. 推荐实现文件

不要大改 V1。

建议新增：

```text
configs/
  ensemble_filter_v1_1.json

refilter_ensemble_v1_1.py

tests/
  test_ensemble_filter_v1_1.py
```

如果现有：

```text
filter_nli_ensemble.py
```

已经把 policy 与 IO 解耦，也可以复用核心函数，但：

> 不要为了 V1.1 重构大量已经稳定工作的 V1 代码。

优先选择最小增量实现。

---

# 25. 推荐配置

`configs/ensemble_filter_v1_1.json`

建议内容：

```json
{
  "policy_name": "conservative_consensus_v1_1",

  "source_predictions_root": "outputs/ensemble_filter_v1",
  "output_root": "outputs/ensemble_filter_v1_1",

  "datasets": [
    "snli",
    "mnlim",
    "mnlimm",
    "sick"
  ],

  "source_confidence_threshold": 0.80,
  "wrong_vote_threshold": 0.95,
  "gold_support_block_threshold": 0.90,

  "conditional_clause_auto_drop": false,

  "dataset_policies": {
    "snli": {
      "min_eligible_voters": 3,
      "min_wrong_consensus": 3,
      "require_wanli_anchor": false,
      "require_unanimous": false
    },

    "mnlim": {
      "min_eligible_voters": 3,
      "min_wrong_consensus": 3,
      "require_wanli_anchor": true,
      "wanli_anchor_model": "alisawuffles/roberta-large-wanli",
      "require_unanimous": false
    },

    "mnlimm": {
      "min_eligible_voters": 3,
      "min_wrong_consensus": 3,
      "require_wanli_anchor": true,
      "wanli_anchor_model": "alisawuffles/roberta-large-wanli",
      "require_unanimous": false
    },

    "sick": {
      "min_eligible_voters": 4,
      "min_wrong_consensus": 4,
      "require_wanli_anchor": false,
      "require_unanimous": true
    }
  }
}
```

---

# 26. Codex 实施顺序

请严格按下面顺序。

### Step 1

读取并理解 V1：

```text
outputs/ensemble_filter_v1/
filter_nli_ensemble.py
run_ensemble_audit.py
existing tests
```

确认哪些文件已经包含完整 per-model prediction 和 confidence。

---

### Step 2

实现 V1.1 config。

不要更改 V1 config。

---

### Step 3

实现：

```text
refilter_ensemble_v1_1.py
```

它必须只读取 V1 结果，不加载模型。

---

### Step 4

实现 dataset-specific policy：

```text
SNLI
MNLIM
MNLIMM
SICK
```

---

### Step 5

实现：

```text
conditional_clause protection
gold-support blocker 0.90
MNLI WANLI anchor
SICK unanimity
```

---

### Step 6

实现 V1 integrity hash check。

---

### Step 7

增加 V1.1 tests。

运行完整 test suite。

---

### Step 8

运行 V1.1 refilter：

```text
snli
mnlim
mnlimm
sick
```

禁止重新 inference。

---

### Step 9

生成：

```text
outputs/ensemble_filter_v1_1/REPORT.md
outputs/ensemble_filter_v1_1/summary_all_datasets.json
outputs/ensemble_filter_v1_1/diff_v1_vs_v1_1.json
```

推荐同时生成：

```text
rescued_from_v1.jsonl
```

---

### Step 10

更新根目录 README：

增加：

```text
Ensemble Filtering V1.1
```

说明：

- V1.1 是 post-hoc refilter；
- 不需要重新运行模型；
- V1 结果仍保留；
- 正式后续实验优先使用：

```text
outputs/ensemble_filter_v1_1/<dataset>/filtered.jsonl
```

---

# 27. 最终验收标准

只有全部满足，V1.1 才算完成。

- [ ] 没有重新运行任何 NLI 模型；
- [ ] V1 文件 SHA-256 前后不变；
- [ ] V1.1 输出在独立目录；
- [ ] SNLI 使用 >=3 同向错误共识；
- [ ] MNLI matched 使用 >=3 且包含 WANLI-only auditor；
- [ ] MNLI mismatched 使用 >=3 且包含 WANLI-only auditor；
- [ ] SICK 必须 4/4 unanimous；
- [ ] wrong vote threshold = 0.95；
- [ ] gold-support blocker threshold = 0.90；
- [ ] conditional_clause 删除数 = 0；
- [ ] source 删除数 = 0；
- [ ] 每条 DROP 均不存在 >=0.90 的 eligible gold-support voter；
- [ ] 每条 DROP 均能追溯到完整 per-model evidence；
- [ ] filtered + removed == input；
- [ ] V1 → V1.1 decision diff 已保存；
- [ ] REPORT.md 已生成；
- [ ] 所有新增及原有测试通过；
- [ ] README 已说明后续实验优先使用 V1.1 filtered data。

---

# 28. V1.1 不做的事情

本阶段明确不做：

```text
不新增 auditor
不重新 inference
不训练模型
不做人工标注
不做 temperature scaling
不做 probability calibration
不重新生成增强数据
不修改 MR
不修复 generator
不改变 source 数据
```

这些均超出当前小模块的投入目标。

---

# 29. 最终目标

Codex 最终需要实现的行为可以概括为：

> **在 V1 已有四模型预测基础上进行纯离线重新过滤：保持 source gating=0.80 和 wrong-vote threshold=0.95；将 gold-support blocker 降至 0.90；SNLI 继续采用至少 3 个模型同向高置信错误共识；MNLIM/MNLIMM 在此基础上要求 WANLI-only RoBERTa 必须参与错误共识；SICK 改为 4/4 全模型一致才删除；所有 conditional_clause 增强样本禁止自动删除。所有 V1 文件保持不变，V1.1 输出写入 `outputs/ensemble_filter_v1_1/`。**

---

# 30. Codex 可直接执行的简短指令

将本文件放入项目后，可以直接给 Codex：

```text
Read this V1.1 specification carefully and implement it as a post-hoc refilter over the existing outputs/ensemble_filter_v1 results.

Do NOT rerun or load any NLI model.
Do NOT modify or overwrite V1 outputs or original datasets.
Implement all dataset-specific V1.1 rules, tests, integrity checks, diff reports, and README updates.
Run the complete test suite and generate outputs/ensemble_filter_v1_1 for SNLI, MNLIM, MNLIMM, and SICK.
```
