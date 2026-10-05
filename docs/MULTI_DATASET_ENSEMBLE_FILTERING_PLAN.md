# MTestQuality 多数据集 NLI Ensemble 过滤实施方案

> 目标：对已经完成蜕变增强的 NLI 测试数据做**低成本、保守、可复现**的自动质量过滤。  
> 适用数据集：**SNLI、MNLIM（MNLI matched）、MNLIMM（MNLI mismatched）、SICK**。  
> 本方案只负责筛选，不重新生成数据，不引入人工逐条复核，也不做复杂的置信度校准。

---

## 1. 核心原则

本项目的目标不是找出所有潜在坏样本，而是：

> **只删除“多个独立 NLI 审计模型都高度确信标签错误”的增强样本。宁可漏删，不要误删。**

因此采用：

**4 个 NLI 模型独立推理 → source gating → 高置信度同向错误共识 → 删除**

必须遵守以下原则：

1. **只过滤增强记录**。`is_source=true` 的原始 source 永远不因 ensemble 规则被删除。
2. source 只用于判断某个 auditor 是否有资格评价其对应的增强样本。
3. 不平均不同模型的 softmax probability；只使用每个模型自己的 top-1 label 和 confidence。
4. 单模型不一致绝不直接删除。
5. 两模型不一致也默认保留。
6. 默认只有在 **至少 3 个合格模型对增强样本给出相同的错误标签，且都达到高置信度阈值** 时才删除。
7. 原始输入文件不可覆盖；所有过滤结果写入新的 output 目录。
8. `pair_id` 用于找到 source 和增强样本的对应关系，但**质量过滤按增强记录逐条进行**。一个增强变体被删，不应连带删除同一 source 的其他正确增强变体。
9. 后续划分训练/验证/测试集时仍应以 `pair_id` 为原子单位，防止 source 与其变体跨 split 泄漏。

---

## 2. 数据集范围与统一输入契约

支持以下 dataset id：

```text
snli
mnlim
mnlimm
sick
```

其中：

- `mnlim` = MultiNLI validation matched
- `mnlimm` = MultiNLI validation mismatched

每条 JSONL 至少包含：

```json
{
  "premise": "...",
  "hypothesis": "...",
  "label": 0,
  "pair_id": "...",
  "is_source": false
}
```

标签统一为：

```text
0 = entailment
1 = neutral
2 = contradiction
```

同时继续保留已有字段，例如：

```text
idx
pair_id
mr_id
mr_type
component_mrs
is_source
```

### 必须验证

对每个数据集，在正式推理前检查：

- JSONL 合法；
- `premise`、`hypothesis`、`label` 存在；
- label 可统一映射到 entailment / neutral / contradiction；
- 每条增强记录必须有 `pair_id`；
- 每个包含增强记录的 `pair_id` 应恰好找到 1 条 `is_source=true` 的 source；
- 不满足 pair 结构的记录**不得自动删除**，应保留并记录 `invalid_pair_structure`。

建议目录继续沿用：

```text
data/
  snli_v3_3/
    source/snli.jsonl
    augmented/snli.jsonl
  mnlim_v3_3/
    source/mnlim.jsonl
    augmented/mnlim.jsonl
  mnlimm_v3_3/
    source/mnlimm.jsonl
    augmented/mnlimm.jsonl
  sick_v3_3/
    source/sick.jsonl
    augmented/sick.jsonl
```

如果实际版本号或路径不同，不要硬编码版本号，应由配置文件指定。

---

## 3. Auditor 模型

默认使用 4 个三分类 NLI 模型。模型应**逐个加载、逐个推理、推理后释放显存**，不要求同时驻留 GPU。

### Model A — DeBERTa baseline

```text
cross-encoder/nli-deberta-v3-large
```

用途：

- 当前项目已经验证过；
- 强 NLI baseline；
- 训练数据包含 SNLI / MultiNLI；
- 继续作为主要 in-domain auditor。

已知 logits 顺序：

```text
contradiction, entailment, neutral
```

### Model B — Diverse-data DeBERTa

```text
MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
```

用途：

- 训练数据更丰富；
- 模型卡列出 MNLI、FEVER-NLI、ANLI、LingNLI、WANLI；
- 未列出 SNLI，因此对 SNLI 可提供与 Model A 不同的训练数据视角。

已知 logits 顺序：

```text
entailment, neutral, contradiction
```

### Model C — WANLI-only RoBERTa

```text
alisawuffles/roberta-large-wanli
```

用途：

- RoBERTa 架构；
- 只在 WANLI 上进行 NLI fine-tuning；
- 对 SNLI、MNLI、SICK 都可作为较独立的外部分布 auditor；
- 它是本 ensemble 中重要的“独立锚点”。

### Model D — BART-MNLI

```text
facebook/bart-large-mnli
```

用途：

- BART 架构，与 DeBERTa / RoBERTa 不同；
- 成熟、常用的三分类 NLI checkpoint；
- 用来增加模型架构多样性。

### 模型加载要求

不要假设所有模型的 label id 顺序一致。

优先使用：

```python
model.config.id2label
```

将 logits 映射为统一标签：

```text
entailment
neutral
contradiction
```

若 `id2label` 不可靠，则使用项目内显式 mapping。

如果无法安全确认三分类 logits 映射：

> **直接终止该模型运行，不允许猜测 label order。**

所有运行必须记录：

```text
model_name
requested_revision
resolved_revision
label_order
transformers_version
torch_version
device
fp16
batch_size
max_length
```

第一次成功运行后，应把实际 `resolved_revision` 写入 manifest，后续复现实验优先使用固定 commit。

---

## 4. 为什么四个数据集统一使用同一 ensemble

为了降低实现和维护成本，本方案**不针对 SNLI / MNLIM / MNLIMM / SICK 单独设计不同模型组合或不同阈值**。

需要注意：

- Model A、Model B、Model D 使用过 SNLI 或 MNLI 体系中的训练数据，因此在 SNLI/MNLI 上属于 in-domain 或 near-domain auditor；
- 这不是用 auditor 的 accuracy 作为论文 benchmark，而是用它们筛查增强标签，因此允许使用强 in-domain 模型；
- Model C（WANLI-only）用于降低所有 auditor 共享 SNLI/MNLI 数据偏差的风险；
- 对 SICK，这套模型整体更接近 out-of-domain 审计，因此 source gating 尤其重要。

不要把 ensemble auditor 在四个数据集上的 accuracy 当作模型性能实验结果；这里的 accuracy 只是**过滤系统诊断指标**。

---

## 5. Source Gating

这是本方案最重要的保护机制。

对每条增强记录 `a`，先找到同一 `pair_id` 的 source `s`。

对每个 auditor `m`：

```text
source_eligible(m) =
    source_prediction(m) == source_gold_label
    AND
    source_confidence(m) >= 0.80
```

只有满足 `source_eligible(m)` 的模型，才有资格评价该 pair 的增强记录。

默认：

```text
SOURCE_CONFIDENCE_THRESHOLD = 0.80
```

原因：

如果模型连原始 source 都不能正确、高置信地判断，那么它对该 source 的增强变体给出的“不一致”不能作为强删除证据。

---

## 6. 增强样本的高置信错误投票

对一个通过 source gating 的模型：

```text
high_conf_wrong_vote(m, a) =
    source_eligible(m)
    AND
    augmented_prediction(m) != augmented_gold_label
    AND
    augmented_confidence(m) >= 0.95
```

默认：

```text
FOLLOWUP_CONFIDENCE_THRESHOLD = 0.95
```

对错误票按模型预测出的错误 label 分组。

例如增强标签是：

```text
gold = entailment
```

四个模型结果：

```text
A -> neutral       0.998
B -> neutral       0.991
C -> neutral       0.973
D -> entailment    0.701
```

若 A/B/C 都通过 source gating，则出现 3 个相同的高置信错误票：

```text
neutral
```

这属于强删除候选。

---

## 7. 默认删除规则：Conservative Consensus V1

### 7.1 定义

对每条 `is_source != true` 的增强记录：

```text
eligible_count
    = 通过 source gating 的模型数

wrong_consensus_count
    = 对同一个错误 label 给出 high-confidence wrong vote 的最大模型数

high_conf_gold_support_count
    = 通过 source gating 且在增强记录上
      预测 gold label、confidence >= 0.95 的模型数
```

### 7.2 DROP 条件

只有同时满足：

```text
1. eligible_count >= 3

2. wrong_consensus_count >= 3

3. 这 >=3 个模型预测的是同一个错误 label

4. 每个支持删除的模型 augmented_confidence >= 0.95

5. high_conf_gold_support_count == 0
```

才执行：

```text
decision = DROP
reason = high_confidence_wrong_consensus
```

形式化表示：

```text
DROP(a) iff

max_y!=gold count({
    m |
    source_eligible(m)
    AND pred_m(a) = y
    AND conf_m(a) >= 0.95
}) >= 3

AND

count({
    m |
    source_eligible(m)
    AND pred_m(a) = gold
    AND conf_m(a) >= 0.95
}) = 0
```

### 7.3 其他情况全部 KEEP

包括：

- 只有 1 个模型高置信反对；
- 只有 2 个模型高置信反对；
- 3 个模型反对但预测成不同的错误 label；
- 模型之间高度分歧；
- 只有 1～2 个 auditor 通过 source gating；
- 存在至少一个 eligible auditor 高置信支持增强标签；
- pair 结构异常；
- source 缺失；
- source 多于 1 条；
- 模型结果不完整。

这些记录全部：

```text
decision = KEEP
```

理由分别记录，不需要人工处理。

---

## 8. 不使用简单多数票

禁止下面这种规则：

```text
3 个模型中 2 个预测 != gold
=> 删除
```

因为：

```text
A -> entailment 0.99
B -> neutral    0.99
C -> contradiction 0.99
```

这种情况说明样本可能很难或存在歧义，但并不能说明某个具体错误标签得到一致支持。

本项目只删：

> **多个模型对“同一个替代标签”形成高置信共识的样本。**

---

## 9. 不对概率做跨模型平均

禁止：

```text
mean_probability(entailment)
mean_probability(neutral)
mean_probability(contradiction)
```

作为最终删除标准。

原因是不同模型的 softmax calibration 不一致，`0.95` 在不同 checkpoint 上不代表完全相同的真实概率。

本方案只把 softmax confidence 当作**模型内部的强度门槛**，最终依赖离散 label 共识。

---

## 10. 实现结构

尽量复用现有：

```text
evaluate_nli_quality.py
```

不要重写已经验证过的单模型推理逻辑。

建议新增：

```text
configs/
  ensemble_filter_v1.json

run_ensemble_audit.py
filter_nli_ensemble.py

docs/
  MULTI_DATASET_ENSEMBLE_FILTERING_PLAN.md

tests/
  test_ensemble_filter.py
```

### `configs/ensemble_filter_v1.json`

至少包含：

```json
{
  "policy_name": "conservative_consensus_v1",
  "datasets": ["snli", "mnlim", "mnlimm", "sick"],
  "source_confidence_threshold": 0.80,
  "followup_confidence_threshold": 0.95,
  "min_eligible_voters": 3,
  "min_wrong_consensus": 3,
  "block_on_high_conf_gold_support": true,
  "models": [
    "cross-encoder/nli-deberta-v3-large",
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    "alisawuffles/roberta-large-wanli",
    "facebook/bart-large-mnli"
  ]
}
```

数据路径不要写死在 Python 中，可在 config 中为每个 dataset 指定。

---

## 11. 推理流程

### Stage A：对四个数据集分别运行四个 auditor

逻辑：

```text
for dataset in datasets:
    for model in models:
        run evaluate_nli_quality.py
        write independent model output
        free GPU memory
```

建议输出：

```text
outputs/
  ensemble_filter_v1/
    snli/
      model_a/
      model_b/
      model_c/
      model_d/
    mnlim/
      ...
    mnlimm/
      ...
    sick/
      ...
```

每个模型仍保留原有：

```text
predictions.jsonl
disagreements.jsonl
pair_quality.jsonl
summary.json
```

### Stage B：合并 auditor 结果

合并前必须验证四次单模型运行：

- input SHA-256 相同；
- record count 相同；
- row positions 完全一致；
- label 完全一致；
- `pair_id` / `is_source` 完全一致。

跨模型结果建议使用：

```text
row_position
```

进行 join。

不要只用 `idx`，因为 `idx` 不一定在所有数据集中全局唯一。

### Stage C：执行 Conservative Consensus V1

逐条生成 ensemble decision。

---

## 12. Ensemble 输出

每个数据集生成：

```text
ensemble_predictions.jsonl
filtered.jsonl
removed.jsonl
ensemble_summary.json
model_manifest.json
```

### `ensemble_predictions.jsonl`

保留原始记录，并增加：

```json
{
  "ensemble_quality": {
    "policy": "conservative_consensus_v1",
    "decision": "keep",
    "reason": "insufficient_wrong_consensus",
    "eligible_voter_count": 4,
    "wrong_consensus_label": "neutral",
    "wrong_consensus_count": 2,
    "high_conf_gold_support_count": 1,
    "supporting_models": [],
    "per_model": {
      "model_a": {
        "source_eligible": true,
        "source_prediction": "entailment",
        "source_confidence": 0.998,
        "augmented_prediction": "neutral",
        "augmented_confidence": 0.991,
        "high_conf_wrong_vote": true
      }
    }
  }
}
```

### `filtered.jsonl`

包含：

```text
所有 source
+
所有 decision=KEEP 的增强记录
```

必须保持原始输入顺序。

### `removed.jsonl`

只包含：

```text
decision=DROP 的增强记录
```

每条都必须带完整 `ensemble_quality`，便于之后审计。

### `ensemble_summary.json`

至少统计：

```text
input_count
source_count
augmented_count
kept_augmented_count
removed_augmented_count
removal_rate
```

以及按以下维度统计：

```text
label
mr_type
mr_id
dataset
wrong_consensus_label
decision_reason
eligible_voter_count
```

还要记录四个模型在 source 上的：

```text
accuracy
high_confidence_accuracy
label-wise accuracy
```

这些只是 auditor 健康检查，不作为论文模型性能结果。

---

## 13. 阈值敏感性统计

正式过滤固定使用：

```text
source threshold = 0.80
follow-up threshold = 0.95
min consensus = 3
```

不需要重新跑模型。

在已有 probabilities 上额外统计：

```text
follow-up threshold = 0.90
follow-up threshold = 0.95
follow-up threshold = 0.99
```

分别会删除多少条。

只写入：

```text
ensemble_summary.json
```

实际 `filtered.jsonl` 始终采用 `0.95`。

这样后续论文如果需要，可以非常低成本报告过滤结果对阈值不敏感/敏感。

---

## 14. 数据集级安全检查

每个 auditor × dataset 在进入 ensemble 之前，执行简单 sanity check。

如果出现以下情况，停止该数据集 ensemble，不生成 filtered 文件：

```text
模型加载失败
label mapping 无法安全确认
预测记录数与输入不一致
input SHA-256 不一致
source accuracy < 0.40
输出包含 NaN / Inf probability
概率和严重偏离 1
```

`source accuracy < 0.40` 主要用于捕获：

- label mapping 写反；
- premise/hypothesis 顺序错误；
- checkpoint 使用错误；
- 数据格式错误。

不要因为某个模型 source accuracy 比其他模型低就自动更换模型；只要通过 sanity check，具体样本仍由 source gating 决定其是否具有投票资格。

---

## 15. 对四个数据集的处理方式

四个数据集使用**完全相同的过滤规则**：

| Dataset | Filtering policy |
|---|---|
| SNLI | Conservative Consensus V1 |
| MNLIM | Conservative Consensus V1 |
| MNLIMM | Conservative Consensus V1 |
| SICK | Conservative Consensus V1 |

不要为某个数据集单独降低阈值以“多删一点”。

如果 SICK 因 domain shift 导致大量模型无法通过 source gating，那么结果应自然表现为：

```text
过滤数量更少
```

而不是人为降低门槛。

这正是 source gating 的作用。

---

## 16. 与当前项目代码的兼容要求

当前 `evaluate_nli_quality.py` 已经支持：

- 三分类标签归一化；
- model `id2label` / 显式 label order；
- confidence；
- margin；
- entropy；
- gold probability；
- source / augmented 分组；
- `pair_id` 分组；
- input SHA-256；
- resolved model revision。

Codex 应优先复用这些能力。

### 不要做

- 不覆盖现有 SNLI 已生成结果；
- 不删除现有单模型输出；
- 不修改原始 JSONL；
- 不把 ensemble 逻辑硬塞进现有单模型 evaluator，导致脚本职责混乱；
- 不引入 LLM API；
- 不增加人工标注流程；
- 不增加复杂 calibration；
- 不训练新的模型。

---

## 17. 自动测试要求

至少实现以下测试。

### Test 1：source 永不删除

任何：

```text
is_source=true
```

必须：

```text
decision=KEEP
```

### Test 2：3 个相同高置信错误票

若 4 个模型中 3 个：

```text
source_eligible=true
aug_pred=neutral
gold=entailment
aug_conf >= 0.95
```

且不存在 high-confidence gold support：

```text
decision=DROP
```

### Test 3：只有 2 个错误票

```text
decision=KEEP
```

### Test 4：3 个模型都反对，但错误标签不一致

例如：

```text
neutral
neutral
contradiction
```

则：

```text
decision=KEEP
```

因为最大同向错误共识只有 2。

### Test 5：存在高置信 gold blocker

例如：

```text
A -> neutral     0.99
B -> neutral     0.98
C -> neutral     0.97
D -> entailment  0.99
```

四个模型都通过 source gating。

由于 D 高置信支持 gold：

```text
decision=KEEP
```

### Test 6：模型 source 判断失败

模型对 source：

```text
prediction != source_gold
```

即使它对 augmentation 给出：

```text
wrong label, confidence=0.999
```

也不能计入删除票。

### Test 7：pair 缺 source

```text
decision=KEEP
reason=invalid_pair_structure
```

### Test 8：输出守恒

必须满足：

```text
len(filtered) + len(removed) == len(input)
```

其中 source 全在 filtered。

### Test 9：顺序稳定

`filtered.jsonl` 中保留记录的相对顺序必须与输入一致。

### Test 10：确定性

相同：

```text
input
model predictions
config
```

必须生成完全相同的 decisions 和 summary。

---

## 18. Codex 实施顺序

请严格按以下顺序实现。

### Step 1

检查当前仓库与四个数据集实际路径，建立统一 dataset config。

不要复制或重新生成数据。

### Step 2

扩展模型 label mapping 支持，并验证 4 个 checkpoint 均可进行三分类 NLI 推理。

### Step 3

实现 `run_ensemble_audit.py`：

- 顺序运行 4 个模型；
- 复用 `evaluate_nli_quality.py`；
- 每个模型独立输出；
- 每次完成后释放模型与 CUDA cache；
- 失败即停止，不生成 ensemble filtered result。

### Step 4

实现 `filter_nli_ensemble.py`：

- 验证 4 份 audit 输入哈希和行号一致；
- 建立 `pair_id -> source`；
- 执行 source gating；
- 执行 Conservative Consensus V1；
- 写出 ensemble outputs。

### Step 5

实现 unit tests。

### Step 6

先在 SNLI 已有数据上跑完整流程。

检查：

```text
四个 auditor 均完成
summary 合理
source 永不删除
removed 每条都能解释
```

### Step 7

运行：

```text
SNLI
MNLIM
MNLIMM
SICK
```

### Step 8

生成一个跨数据集总表：

```text
outputs/ensemble_filter_v1/summary_all_datasets.json
```

至少包含：

| dataset | source | augmented | kept | removed | removal_rate |
|---|---:|---:|---:|---:|---:|

并额外给出：

```text
按 MR 的 removal rate
按 gold label 的 removal rate
0.90 / 0.95 / 0.99 sensitivity
```

---

## 19. 最终验收标准

只有全部满足以下条件，任务才算完成：

- [ ] SNLI、MNLIM、MNLIMM、SICK 全部接入；
- [ ] 四个 auditor 全部使用固定模型 revision；
- [ ] 四个模型 label mapping 均经过验证；
- [ ] 原始输入文件没有被修改；
- [ ] source 删除数为 0；
- [ ] DROP 只发生在增强记录；
- [ ] 每条 DROP 至少有 3 个通过 source gating 的模型支持；
- [ ] 每条 DROP 的 >=3 个模型预测相同错误 label；
- [ ] 每个支持删除的模型 follow-up confidence >= 0.95；
- [ ] 每条 DROP 不存在 eligible auditor 的 high-confidence gold support；
- [ ] `filtered + removed == input`；
- [ ] 所有单模型与 ensemble 运行都记录 input hash 和 model revision；
- [ ] 有按 dataset / MR / label 的过滤统计；
- [ ] 有 0.90 / 0.95 / 0.99 阈值敏感性统计；
- [ ] unit tests 全部通过；
- [ ] README 增加 ensemble filtering 的运行方法。

---

## 20. 默认参数汇总

实现时默认值固定为：

```text
AUDITOR_COUNT = 4

SOURCE_CONFIDENCE_THRESHOLD = 0.80
FOLLOWUP_CONFIDENCE_THRESHOLD = 0.95

MIN_ELIGIBLE_VOTERS = 3
MIN_WRONG_CONSENSUS = 3

REQUIRE_SAME_WRONG_LABEL = true
BLOCK_ON_HIGH_CONF_GOLD_SUPPORT = true

FILTER_SOURCE_ROWS = false
FILTER_AUGMENTED_ROWS = true
```

这套默认规则的目标是：

> **高 precision 删除明显错误增强样本，而不是追求最高 bad-sample recall。**

后续若需要更激进的过滤，只调整 config，不修改核心算法。

---

## 21. 参考模型页面

- `cross-encoder/nli-deberta-v3-large`  
  https://huggingface.co/cross-encoder/nli-deberta-v3-large

- `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`  
  https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli

- `alisawuffles/roberta-large-wanli`  
  https://huggingface.co/alisawuffles/roberta-large-wanli

- `facebook/bart-large-mnli`  
  https://huggingface.co/facebook/bart-large-mnli

---

## 22. 一句话实现目标

Codex 最终应实现：

> **对 SNLI、MNLIM、MNLIMM、SICK 的增强数据分别运行 4 个预训练 NLI auditor，只让能正确且较高置信判断原始 source 的模型参与投票；若至少 3 个合格 auditor 对某条增强样本以 >=0.95 的置信度预测出同一个、且不同于增强标签的 NLI label，同时没有合格 auditor 高置信支持增强标签，则删除该增强样本；其他情况全部保留。**
