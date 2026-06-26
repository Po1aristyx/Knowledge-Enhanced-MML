# ConvKAMT: 知识增强的多模态阿尔茨海默症诊断模型

**作者：童逸轩**  
**学号：2023337621312**  

本项目为 KG-2026 知识图谱大作业（路线 A）的完整实现代码。提出了 ConvKAMT 框架，将 ConvE 和 ConvR 知识图谱嵌入技术应用于医学领域图谱（DemRKG），并利用跨模态注意力机制将知识动态注入多模态（EHR, MRI, BIO）数据中，实现对脑疾病的高精度分类诊断。

## 环境配置
```bash
conda create -n audiokeshe python=3.9
conda activate audiokeshe
pip install torch torchvision torchaudio numpy pandas scikit-learn matplotlib
```

## 核心结构说明

* `train_conve_convr.py`: 用于在 DemRKG 上训练 ConvE 和 ConvR 知识图谱嵌入的核心脚本。支持断点续训。
* `DementiaHKG/DementiaHKG/图谱构建/`: 存放各数据集子图构建逻辑和数据。
* `DementiaHKG/DementiaHKG/*_ConvE.py`: 各数据集基于 ConvE 嵌入的下游多模态诊断脚本。
* `DementiaHKG/DementiaHKG/*_ConvR.py`: 各数据集基于 ConvR 嵌入的下游多模态诊断脚本。
* `generate_figures.py`: 可视化脚本，用于渲染所有消融实验及横向对比高清图表。
* `checkpoints/`: 存储模型长时间训练的权重，用于断电恢复机制。
* `figures/`: 产出的实验图表（SOTA 对比，消融雷达图等）。

## 运行指南

1. **图谱嵌入训练**
```bash
python train_conve_convr.py
```
> 将在 `DementiaHKG/DementiaHKG/` 目录下生成各数据集的 `-Embeddings.npy` 向量。

2. **下游分类实验**
```bash
# 推荐使用批量自动化脚本，会自动监控进度并支持中断恢复
python run_kamt_downstream.py
```

3. **结果提取与图表生成**
```bash
python extract_ablation_results.py
python generate_figures.py
```

## 实验结果与可视化
模型生成的日志位于 `DementiaHKG/DementiaHKG/log_*.txt`。提取出的性能对比表汇总至 `ablation_embedding_method.csv`。使用 `generate_figures.py` 可以重现最终报告中使用的所有高清图表。

## 鲁棒性设计（断电保护）
本项目在进行 KGE 以及多模态 Transformer 训练时，引入了自动 Checkpointing 机制：每隔 10 epoch 会自动在 `checkpoints/` 目录下保存模型。若遭遇意外终端或人为停止，再次运行原有脚本即可全自动从断点继续执行，不会造成重复训练。

## 相关数据集与模型资产下载
本项目所需的数据集和模型压缩包可以在 Zenodo下载，可直接点击以下链接进行下载：
*   **多模态数据集**: [https://zenodo.org/records/19646995](https://zenodo.org/records/19646995) (总结神经影像 MRI、电子健康记录 EHR、生物标志物 BioMarker 三种数据模态，用于进行联合学习以支持辅助诊断任务)
*   **Baseline常规型方案**: [https://zenodo.org/records/19652263](https://zenodo.org/records/19652263)
*   **集成外部知识库的DemRKG方法**: [https://zenodo.org/records/19652297](https://zenodo.org/records/19652297) (试图在私域知识提取基础上，通过领域知识蒸馏和精炼的方式，进一步提升有效性)
