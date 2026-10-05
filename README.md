# MTestQuality

用于审计和保守过滤 NLI 增强测试数据质量的独立工具。单模型 evaluator 对每条 `premise` / `hypothesis` 重新推理，记录预测、概率、置信度、margin、熵和分组统计；ensemble 流程复用该 evaluator，对 SNLI、MNLIM、MNLIMM、SICK 使用四个固定版本 auditor，通过 source gating 和同向高置信错误共识逐条过滤增强记录。

当前默认模型是 [`cross-encoder/nli-deberta-v3-large`](https://huggingface.co/cross-encoder/nli-deberta-v3-large)。它在 SNLI 和 MultiNLI 上训练，适合对 SNLI 体系的增强数据做 in-domain 自动筛查。自动审计结果是候选筛选信号，不应直接视为人工金标。

## 安装

建议使用 Python 3.9 或更新版本，并先安装与本机 CUDA 匹配的 PyTorch：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

项目已包含 SNLI、MNLI-matched、MNLI-mismatched 和 SICK 四套 v3.3 数据，四套均已完成四模型 ensemble 审计和保守过滤。结果与验收记录见 [`outputs/ensemble_filter_v1/REPORT.md`](outputs/ensemble_filter_v1/REPORT.md)。首次推理时仍会从 Hugging Face 下载模型权重；若要把模型缓存放在项目目录：

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

## 仓库内置数据

当前仓库已经包含四套可直接运行和复核的 v3.3 数据：

- [`data/snli_v3_3/source/snli.jsonl`](data/snli_v3_3/source/snli.jsonl)：增强前的 source 数据，9,824 条。
- [`data/snli_v3_3/augmented/snli.jsonl`](data/snli_v3_3/augmented/snli.jsonl)：source 与增强记录合并后的完整审计输入，23,205 条。
- [`data/snli_v3_3/augmented/snli.report.json`](data/snli_v3_3/augmented/snli.report.json)：增强数据生成报告。
- [`data/mnlimm_v3_3/source/mnlimm.jsonl`](data/mnlimm_v3_3/source/mnlimm.jsonl)：MNLI validation mismatched source 数据，9,832 条。
- [`data/mnlimm_v3_3/augmented/mnlimm.jsonl`](data/mnlimm_v3_3/augmented/mnlimm.jsonl)：MNLI-mismatched 完整增强数据，20,257 条。
- [`data/mnlimm_v3_3/augmented/mnlimm.report.json`](data/mnlimm_v3_3/augmented/mnlimm.report.json)：MNLI-mismatched 增强数据生成报告。
- [`data/mnlim_v3_3/source/mnlim.jsonl`](data/mnlim_v3_3/source/mnlim.jsonl)：MNLI validation matched source 数据，9,815 条。
- [`data/mnlim_v3_3/augmented/mnlim.jsonl`](data/mnlim_v3_3/augmented/mnlim.jsonl)：MNLI-matched 完整增强数据，20,249 条。
- [`data/mnlim_v3_3/augmented/mnlim.report.json`](data/mnlim_v3_3/augmented/mnlim.report.json)：MNLI-matched 增强数据生成报告。
- [`data/sick_v3_3/source/sick.jsonl`](data/sick_v3_3/source/sick.jsonl)：SICK source 数据，4,906 条。
- [`data/sick_v3_3/augmented/sick.jsonl`](data/sick_v3_3/augmented/sick.jsonl)：SICK 完整增强数据，10,925 条。
- [`data/sick_v3_3/augmented/sick.report.json`](data/sick_v3_3/augmented/sick.report.json)：SICK 增强数据生成报告。
- [`outputs/snli_v3_3_cross_encoder_deberta_v3_large/`](outputs/snli_v3_3_cross_encoder_deberta_v3_large/)：SNLI 输入的完整已生成审计结果。

文件说明、行数和 SHA-256 见 [`data/README.md`](data/README.md)。从 GitHub 克隆本仓库后，无需再从 MTrain 或其他数据目录复制输入。

内置数据分别衍生自 SNLI、MultiNLI 与 SICK；数据来源、修改说明、许可和论文引用见 [`DATA_LICENSE.md`](DATA_LICENSE.md)。

## 使用

```bash
python evaluate_nli_quality.py data/snli_v3_3/augmented/snli.jsonl \
  --output-dir outputs/snli_v3_3_rerun \
  --device cuda:0 \
  --batch-size 128 \
  --max-length 256 \
  --confidence-threshold 0.90
```

MNLI-mismatched 增强数据可直接运行：

```bash
python evaluate_nli_quality.py data/mnlimm_v3_3/augmented/mnlimm.jsonl \
  --output-dir outputs/mnlimm_v3_3_cross_encoder_deberta_v3_large \
  --device cuda:0 \
  --batch-size 128 \
  --max-length 256 \
  --confidence-threshold 0.90
```

MNLI-matched 与 SICK 增强数据可分别运行：

```bash
python evaluate_nli_quality.py data/mnlim_v3_3/augmented/mnlim.jsonl \
  --output-dir outputs/mnlim_v3_3_cross_encoder_deberta_v3_large \
  --device cuda:0 --batch-size 128 --max-length 256 --confidence-threshold 0.90

python evaluate_nli_quality.py data/sick_v3_3/augmented/sick.jsonl \
  --output-dir outputs/sick_v3_3_cross_encoder_deberta_v3_large \
  --device cuda:0 --batch-size 128 --max-length 256 --confidence-threshold 0.90
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

仓库会追踪当前审计使用的数据集和逐条 JSONL 结果，因此可以直接复核现有结果或重新运行。已有 SNLI v3.3 完整审计位于 [`outputs/snli_v3_3_cross_encoder_deberta_v3_large/`](outputs/snli_v3_3_cross_encoder_deberta_v3_large/)。下载的模型权重仍由 Hugging Face cache 管理，不纳入 Git。

## 结果解释

- `confidence` 是模型最大 softmax 概率，不是经过校准的真实正确概率。
- 高置信不一致可能是增强标签错误，也可能是审计模型错误。
- 构建测试集时应按 `pair_id` 整组切分，避免 source 和增强变体跨集合泄漏。
- 建议对高置信一致、高置信不一致和低置信三组分别做人审抽样。

## 四模型 ensemble 过滤

完整规则见 [`docs/MULTI_DATASET_ENSEMBLE_FILTERING_PLAN.md`](docs/MULTI_DATASET_ENSEMBLE_FILTERING_PLAN.md)。默认配置为 [`configs/ensemble_filter_v1.json`](configs/ensemble_filter_v1.json)，输入路径和输出根目录相对于项目根目录解析，可在配置中修改，不在 Python 中写死版本号。

默认四个 auditor 分别是 DeBERTa baseline、diverse-data DeBERTa、WANLI-only RoBERTa 和 BART-MNLI。配置固定了完整 commit SHA、经官方 `config.json` 核对的 logits 顺序及来源链接。模型按数据集、按 auditor 顺序在独立子进程中运行；子进程退出会释放该模型的 CUDA 分配和缓存。下载的权重保存在 `nli_model_cache/hub/`，不纳入 Git。

```bash
# 本机两张 GPU 中默认使用 cuda:1；其他机器请修改 config.runtime.device。
# CPU 运行还应将 fp16 设置为 false。
python run_ensemble_audit.py --smoke-only --output-root outputs/new_smoke_check

# 先完成 SNLI 的真实全流程
python run_ensemble_audit.py --datasets snli --resume

# 验证已完成结果后继续其余数据集，并生成四数据集总表
python run_ensemble_audit.py --resume

# 只重新执行过滤，无需重新跑模型；写入新的目录
python filter_nli_ensemble.py --dataset snli \
  --audit-root outputs/ensemble_filter_v1/snli \
  --output-dir outputs/snli_ensemble_refilter

# 自动测试无需下载模型或运行 GPU 推理
python -m unittest discover -s tests -v
```

正式规则：source 预测正确且 confidence ≥ 0.80 的 auditor 才有资格投票；至少 3 个合格 auditor 对增强记录以 confidence ≥ 0.95 预测出同一个错误标签，且没有合格 auditor 高置信支持 gold，才删除这条增强记录。source 永远保留，同一 pair 的其他增强变体独立决策，不平均跨模型概率。

每个 `outputs/ensemble_filter_v1/<dataset>/` 包含四份独立单模型输出，以及：

- `ensemble_predictions.jsonl`：全部原始字段及逐模型证据、决策和理由。
- `filtered.jsonl`：全部 source 和保留的增强记录，保持输入相对顺序，附带 `ensemble_quality`。
- `removed.jsonl`：仅删除的增强记录，附带完整删除证据。
- `ensemble_summary.json`：按 dataset、gold label、MR、决策理由、共识标签、eligible voter 数的统计，source auditor 健康检查和阈值敏感性。
- `model_manifest.json`：输入哈希、固定模型 revision、映射、运行环境及产物哈希。

四个数据集全部成功后生成 `summary_all_datasets.json`；只选择部分数据集时生成 `summary_selected_datasets.json`。removal rate 的分母是增强记录数，source 不计入该分母。0.90 / 0.95 / 0.99 敏感性统计分别重算错误票和 gold blocker，不改变正式过滤文件。

无 source、多个 source、缺失 pair id 或不合法的 source 标记会保留该增强记录，并记录 `invalid_pair_structure`。位置已对齐但单条预测字段不完整时保留，并记录 `incomplete_model_results`。整份审计缺失、输入哈希/数量/行号/原始字段不一致、无法确认标签映射、NaN/Inf、概率和偏离 1 超过 1e-5，或 source accuracy < 0.40，都会停止该数据集，不发布过滤文件。

已有输出不会被默认覆盖。`--resume` 会重新核验已完成单模型与 ensemble 的输入、配置、版本和产物哈希。失败的单模型日志保留在该数据集目录的 `.model_*-*/` 中；重新运行会使用新的临时目录。已有 SNLI baseline 只有通过固定 revision、输入、运行参数和完整逐行核验后才复用；历史请求 revision 仍记录为 `main`，实际使用的 resolved revision 已固定。原 SNLI 结果目录与输入文件保持不变。

## 后续计划

多数据集接入、四模型交叉审计及保守筛选已实现。历史路线见 [`docs/ROADMAP.md`](docs/ROADMAP.md)，本次实现与验收依据见 ensemble 实施方案。

## 已验证环境

首次 SNLI v3.3 完整审计使用：Python 3.9.21、PyTorch 1.12.1+cu113、Transformers 4.29.2、RTX 3090、fp16。23,205 条记录均成功推理，且无记录触发 256-token 截断上限。
