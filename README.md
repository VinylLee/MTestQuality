# MTestQuality

用于审计 NLI 增强测试数据质量的独立工具。当前版本使用训练好的三分类 NLI cross-encoder，对每条 `premise` / `hypothesis` 重新推理，并记录模型预测、三类概率、置信度、margin、熵、与数据标签是否一致，以及按 source、label、MR 和 `pair_id` 聚合的统计信息。

当前默认模型是 [`cross-encoder/nli-deberta-v3-large`](https://huggingface.co/cross-encoder/nli-deberta-v3-large)。它在 SNLI 和 MultiNLI 上训练，适合对 SNLI 体系的增强数据做 in-domain 自动筛查。自动审计结果是候选筛选信号，不应直接视为人工金标。

## 安装

建议使用 Python 3.9 或更新版本，并先安装与本机 CUDA 匹配的 PyTorch：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

项目首次运行时会从 Hugging Face 下载模型。若要把缓存放在项目目录：

```bash
export HF_HOME="$PWD/nli_model_cache"
```

## 输入格式

输入必须是 JSONL，每行至少包含：

```json
{"premise":"A person is walking.","hypothesis":"Someone is moving.","label":0}
```

标签支持名称或数字。数字映射固定为：

- `0`：entailment
- `1`：neutral
- `2`：contradiction

若存在 `idx`、`pair_id`、`mr_id`、`mr_type`、`is_source`、`component_mrs` 等字段，脚本会原样保留，并用于生成相应分组统计。详细契约见 [`docs/OUTPUT_SCHEMA.md`](docs/OUTPUT_SCHEMA.md)。

## 使用

```bash
python evaluate_nli_quality.py /path/to/dataset.jsonl \
  --output-dir outputs/my_audit \
  --device cuda:0 \
  --batch-size 128 \
  --max-length 256 \
  --confidence-threshold 0.90
```

CPU 推理：

```bash
python evaluate_nli_quality.py /path/to/dataset.jsonl \
  --output-dir outputs/my_audit \
  --device cpu
```

如果换用其他三分类 NLI 模型，而模型配置未提供可靠的 `id2label`，必须显式指定 logits 顺序：

```bash
python evaluate_nli_quality.py data.jsonl \
  --output-dir outputs/audit \
  --model some/model \
  --model-label-order contradiction,entailment,neutral
```

## 输出

每次运行生成：

- `predictions.jsonl`：完整逐条结果，保留输入字段并新增 `nli_quality`。
- `disagreements.jsonl`：模型预测与数据标签不一致的记录。
- `high_confidence_agreements.jsonl`：预测一致且达到指定置信度阈值的候选记录。
- `pair_quality.jsonl`：按 `pair_id` 汇总，便于整组筛选并防止数据切分泄漏。
- `summary.json`：总体和分组指标、混淆矩阵、不同阈值表现、模型 revision、输入哈希及运行环境。

仓库不追踪完整数据集、模型权重和逐条 JSONL 结果。已有 SNLI v3.3 审计的轻量报告位于 [`reports/snli_v3_3_cross_encoder_deberta_v3_large/`](reports/snli_v3_3_cross_encoder_deberta_v3_large/)。

## 结果解释

- `confidence` 是模型最大 softmax 概率，不是经过校准的真实正确概率。
- 高置信不一致可能是增强标签错误，也可能是审计模型错误。
- 构建测试集时应按 `pair_id` 整组切分，避免 source 和增强变体跨集合泄漏。
- 建议对高置信一致、高置信不一致和低置信三组分别做人审抽样。

## 后续计划

当前项目刻意只保留已经运行验证过的单模型审计能力。计划中的工作记录在 [`docs/ROADMAP.md`](docs/ROADMAP.md)：

1. 接入其他测试数据集的增强数据并统一审计。
2. 增加第二类模型的交叉审计。
3. 在多模型结果基础上增加可配置筛选功能。

## 已验证环境

首次 SNLI v3.3 完整审计使用：Python 3.9.21、PyTorch 1.12.1+cu113、Transformers 4.29.2、RTX 3090、fp16。23,205 条记录均成功推理，且无记录触发 256-token 截断上限。
