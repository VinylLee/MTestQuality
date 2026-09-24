# 数据来源与许可

## SNLI 来源

本仓库 `data/snli_v3_3/` 中的数据衍生自 Stanford Natural Language Inference（SNLI）Corpus 1.0：

- 官方项目页：https://nlp.stanford.edu/projects/snli/
- 数据维护者：The Stanford NLP Group
- 原始论文：Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. “A Large Annotated Corpus for Learning Natural Language Inference.” EMNLP 2015.

SNLI 官方项目页声明该语料采用 [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)（CC BY-SA 4.0）许可。仓库内包含 SNLI 内容及其衍生文本的数据文件按同一许可提供。

## MultiNLI 来源

本仓库 `data/mnlimm_v3_3/` 中的数据衍生自 Multi-Genre Natural Language Inference（MultiNLI）Corpus 1.0 的 `validation_mismatched` split：

- NYU 官方归档：https://archive.nyu.edu/handle/2451/41736
- 原始项目页：https://www.nyu.edu/projects/bowman/multinli/
- 作者：Adina Williams, Nikita Nangia, and Samuel R. Bowman

NYU 官方归档将 MultiNLI 标记为允许分发，但许可随语料区段而异，并要求参照随语料发布的论文了解详情。MultiNLI 论文说明大部分非 FICTION 内容使用 Open American National Corpus 的许可，FICTION 内容使用若干其他宽松许可。使用或再分发 `mnlimm_v3_3` 时应保留本归属说明，并同时遵守 MultiNLI 分发包中适用于具体来源区段的许可条款。

## 修改说明

相对原始 SNLI 与 MultiNLI，本仓库中的数据经过了以下处理：

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

MultiNLI 数据还应引用：

```bibtex
@inproceedings{williams-etal-2018-broad,
  title     = {A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference},
  author    = {Williams, Adina and Nangia, Nikita and Bowman, Samuel R.},
  booktitle = {Proceedings of NAACL-HLT, Volume 1 (Long Papers)},
  year      = {2018},
  pages     = {1112--1122},
  doi       = {10.18653/v1/N18-1101},
  url       = {https://aclanthology.org/N18-1101}
}
```

代码本身未因本文件自动获得数据集许可；本说明仅覆盖 `data/` 与由这些数据直接生成的 `outputs/` 数据制品。
