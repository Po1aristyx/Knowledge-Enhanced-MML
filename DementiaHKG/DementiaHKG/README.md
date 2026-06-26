# DementiaHKG Graph-Guided Multimodal AI Diagnostic Model

## 项目简介 (Project Overview)

本项目包含用于阿尔茨海默症及相关痴呆症诊断的 **DementiaHKG 异构知识图谱引导的多模态人工智能模型**的完整复现代码。

本研究的核心贡献在于构建并引入了针对不同多模态数据集定制的异构知识图谱 **DementiaHKG**。为了严谨地验证外部知识库的引入以及异构精炼过程的有效性，我们在下游多模态分类任务中设计了以下对比基线：
* **CustomKG**：仅基于内部数据集诊断记录表提取构建的知识图谱，旨在作为内部知识库的基线。
* **PrimeKG (未精炼)**：直接引入外部 PrimeKG 知识库，但未针对目标任务进行异构精炼，旨在证明异构精炼机制的必要性。
* **DementiaHKG**：结合了内部诊断记录与外部 PrimeKG 知识，并经过异构精炼，旨在提供最精准的图谱先验引导。

## 📁 目录结构 (Repository Structure)

```text
DementiaHKG/
├── model1/                                 # 下游分类代码依赖的共享模型库 (包含各数据集对应的 model1.py)
├── 图谱构建/                               # DementiaHKG 图谱构建代码 (Step 1 to Step 3)
├── 诊断记录表/                             # 各数据集的原始诊断记录表 (用于构建 CustomKG 内部知识)
├── ADNI三模态+DementiaHKG.ipynb            # ADNI 数据集下游多模态分类与评估
├── AIBL三模态+DementiaHKG.ipynb            # AIBL 数据集下游多模态分类与评估
├── NACC三模态+DementiaHKG.ipynb            # NACC 数据集下游多模态分类与评估
├── PPMI三模态+DementiaHKG.ipynb            # PPMI 数据集下游多模态分类与评估
├── kg.csv                                  # 原始 PrimeKG 外部知识图谱文件
└── README.md                               # 项目说明文档
```

## 💾 数据集获取 (Datasets)

本研究所使用的多模态特征数据（涵盖影像、Bio 特征及临床评估等）已开源托管于 Zenodo 平台。
* **下载链接**: [https://zenodo.org/records/19646995](https://zenodo.org/records/19646995)
* **基准模型链接**: [https://zenodo.org/records/19652263](https://zenodo.org/records/19652263)
* 请将下载后的数据文件放置于项目根目录或在代码中正确配置读取路径。

## 🚀 运行指南 (Getting Started)

### 第一步：DementiaHKG 图谱构建
在运行下游分类任务前，需要先为每个数据集生成对应的 DementiaHKG 嵌入与实体文件。
1.  进入 `图谱构建/` 目录。
2.  按顺序运行针对目标数据集的 3 个构建步骤 (Step 1-3) 脚本（step1为检索代码，可以跳过）。
3.  运行完毕后，程序将为对应数据集输出以下三个核心文件：
    * `[Dataset]-DementiaHKG-Embeddings.npy`
    * `[Dataset]-DementiaHKG-Entities.json`
    * `[Dataset]-DementiaHKG-Entity2ID.json`

### 第二步：下游多模态分类任务
图谱构建完成后，即可运行根目录下的 Jupyter Notebooks 进行模型训练与评估。
1.  打开对应数据集的 Notebook（例如：`ADNI三模态+DementiaHKG.ipynb`）。
2.  从`model1/` 目录下找到对应数据集的代码。

