# 四数据集 Conservative Consensus V1 过滤结果

四套数据均已完成四个固定版本 auditor 的真实推理、source gating、保守过滤和独立验收。总输入 74,636 条，其中 source 34,377 条、增强 40,259 条；删除增强 190 条，保留增强 40,069 条。过滤文件合计 74,446 条，source 删除数为 0。

## 正式过滤结果

| 数据集 | source 全部保留 | 原始增强 | 保留增强 | 删除增强 | 增强删除率 |
|---|---:|---:|---:|---:|---:|
| SNLI | 9,824 | 13,381 | 13,322 | 59 | 0.4409% |
| MNLIM | 9,815 | 10,434 | 10,366 | 68 | 0.6517% |
| MNLIMM | 9,832 | 10,425 | 10,372 | 53 | 0.5084% |
| SICK | 4,906 | 6,019 | 6,009 | 10 | 0.1661% |
| 合计 | 34,377 | 40,259 | 40,069 | 190 | 0.4719% |

正式阈值固定为 source 0.80、follow-up 0.95、至少 3 个 eligible auditor 对同一个错误标签的共识，并阻止存在 eligible high-confidence gold support 的删除。所有 source 都保留；过滤按增强记录逐条进行，输入顺序和原始字段保持不变。

## 产物

- 每套数据：`<dataset>/model_a` 至 `model_d` 中保留完整单模型输出和运行元数据。
- 可使用的新输入：[SNLI](snli/filtered.jsonl)、[MNLIM](mnlim/filtered.jsonl)、[MNLIMM](mnlimm/filtered.jsonl)、[SICK](sick/filtered.jsonl)。这些文件包含全部 source 和保留增强，新增 `ensemble_quality`。
- 删除证据：各数据集的 `removed.jsonl`；完整逐条证据：`ensemble_predictions.jsonl`。
- 分组统计与运行版本：各数据集的 `ensemble_summary.json` 与 `model_manifest.json`。
- 跨数据集总表、MR/gold label 删除率和敏感性：[summary_all_datasets.json](summary_all_datasets.json)。
- 原文件保护哈希：[input_integrity_before.json](input_integrity_before.json)。最终验收记录：[validation_receipt.json](validation_receipt.json)。

## Source auditor 健康检查

以下是 source 标签一致率，仅用于过滤诊断，不是模型性能 benchmark：

| 数据集 | A DeBERTa baseline | B diverse-data DeBERTa | C WANLI RoBERTa | D BART-MNLI |
|---|---:|---:|---:|---:|
| SNLI | 92.20% | 88.98% | 73.96% | 88.97% |
| MNLIM | 90.20% | 91.22% | 80.38% | 90.18% |
| MNLIMM | 90.49% | 90.80% | 80.94% | 90.01% |
| SICK | 58.85% | 64.59% | 78.56% | 57.83% |

四模型在各数据集上均通过 0.40 的 source accuracy 下限。SICK 的模型诊断差异较大，仍使用相同 source gating 和删除阈值。KEEP 表示没有满足强删除证据，不表示已认证为无标签错误的高质量样本。

## 阈值敏感性

以下统计复用现有预测，分别重算 follow-up 错误票和 gold blocker；正式输出仍采用 0.95：

| follow-up 阈值 | SNLI 删除 | MNLIM 删除 | MNLIMM 删除 | SICK 删除 | 合计 |
|---|---:|---:|---:|---:|---:|
| 0.90 | 68 | 86 | 71 | 10 | 235 |
| 0.95 | 59 | 68 | 53 | 10 | 190 |
| 0.99 | 32 | 25 | 23 | 1 | 81 |

按 gold label 删除：entailment 14、neutral 95、contradiction 81。按 MR 的删除数最多的是 `uninformative` 67、`negation_flip` 32 和 `adding_contradiction` 25；完整分组分母和删除率保存在总表中。

## 验收

`python -m unittest discover -s tests -v`：37 项测试全部通过，无跳过。覆盖策略边界、异常输入、顺序、确定性、失败传播、缓存复用，以及真实四数据集的独立验收。

验收逐条核对每条 DROP 的真实审计预测：至少 3 个 source eligible 模型以 ≥0.95 置信度预测同一个非 gold 标签，且没有 eligible 模型高置信支持 gold。所有过滤文件与 ensemble 决策逐条一致，`filtered + removed == input`，输入顺序稳定。12 个原始输入/生成报告和 6 个原有 SNLI 审计文件的 SHA-256 均未改变。

四个模型的 revision、映射来源和默认运行参数固定在 [配置](../../configs/ensemble_filter_v1.json)。真实运行使用 CUDA:1、fp16、max_length=256，A batch_size=128，B/C/D batch_size=64。SNLI model_a 复用了已核验的历史预测，保留旧请求 revision `main`，resolved revision 与配置固定 commit 一致。其余 15 个 dataset × auditor 组合均重新以固定 commit 运行。

首次 WANLI smoke 因网络 TLS 中断失败，诊断记录保留在 `smoke/model_c/run.log`；模型缓存齐备后，在 `validation/smoke/` 完成四模型离线真实推理验证。随后所有全量推理均在离线模式下完成。
