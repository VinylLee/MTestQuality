# 数据来源与许可

## 来源

本仓库 `data/snli_v3_3/` 中的数据衍生自 Stanford Natural Language Inference（SNLI）Corpus 1.0：

- 官方项目页：https://nlp.stanford.edu/projects/snli/
- 数据维护者：The Stanford NLP Group
- 原始论文：Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. “A Large Annotated Corpus for Learning Natural Language Inference.” EMNLP 2015.

SNLI 官方项目页声明该语料采用 [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)（CC BY-SA 4.0）许可。仓库内包含 SNLI 内容及其衍生文本的数据文件按同一许可提供。

## 修改说明

相对原始 SNLI，本仓库中的数据经过了以下处理：

- 选择并转换为本项目使用的 JSONL 字段结构；
- 使用 MTrain 的 metamorphic relations 生成 hypothesis 变体；
- 增加 `idx`、`pair_id`、`mr_id`、`mr_type`、`is_source` 和部分 `component_mrs` 元数据；
- 使用预训练 NLI 模型生成预测概率、置信度、一致性、pair 级统计和筛选视图。

这些修改与审计结果不代表 Stanford NLP Group 的认可。完整文件哈希和记录数见 `data/README.md`，生成配置见 `data/snli_v3_3/augmented/snli.report.json`。

## 引用

使用这些数据开展研究时，请引用 SNLI 原始论文：

```bibtex
@inproceedings{bowman-etal-2015-large,
  title     = {A Large Annotated Corpus for Learning Natural Language Inference},
  author    = {Bowman, Samuel R. and Angeli, Gabor and Potts, Christopher and Manning, Christopher D.},
  booktitle = {Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing},
  year      = {2015},
  pages     = {632--642},
  doi       = {10.18653/v1/D15-1075},
  url       = {https://aclanthology.org/D15-1075}
}
```

代码本身未因本文件自动获得 CC BY-SA 许可；本说明仅覆盖 `data/` 与由这些数据直接生成的 `outputs/` 数据制品。
