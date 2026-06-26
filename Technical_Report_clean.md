---
title: KG大作业实验报告 (ConvKAMT)
author: 童逸轩 (2023337621312)
---

# 知识增强的多模态阿尔茨海默症诊断模型：ConvKAMT 的设计与实现

**作者：童逸轩**  
**学号：2023337621312**  
**课程：知识图谱 (KG-2026)**  

---

## 摘要

阿尔茨海默症（Alzheimer's Disease, AD）的早期诊断对延缓病情进展具有重要意义。随着医学信息学的发展，结合电子病历（EHR）、医学影像（MRI）及生物标志物（Biomarkers）的多模态融合方法逐渐成为研究热点。然而，现有多模态模型往往缺乏医学背景知识的指导，难以捕获不同模态间潜在的病理关联。为了解决这一问题，本文提出了一种基于知识增强的多模态 Transformer 诊断模型——**ConvKAMT**。该模型引入了专注于痴呆症的领域知识图谱（DemRKG），并首次在多模态融合中引入了 **ConvE** 和 **ConvR** 知识嵌入策略。通过卷积操作对实体与关系之间的高阶交互特征进行提取，模型能产生更具表征能力的知识序列。在知识注入层，我们利用交叉注意力机制将知识嵌入动态融入各个模态特征中，从而实现先验医学知识对多模态数据的自适应引导。
我们在四个公开脑疾病队列数据集（ADNI, AIBL, NACC, PPMI）上进行了广泛的实验。结果表明，相比于单模态和传统的多模态基线模型，以及基于 TransE 的已有方法，ConvKAMT 表现出显著的优势。特别是使用 ConvR 嵌入的变体，在所有数据集上的诊断 F1 Score 和 AUC-ROC 均达到了 SOTA (State-of-The-Art) 水平。

---

## 1. 问题定义与研究动机

### 1.1 脑疾病多模态诊断的挑战
阿尔茨海默症的演变过程高度复杂。临床上，医生通常需要综合患者的人口统计学信息、认知量表评分（如 MMSE）、脑部核磁共振成像（MRI）以及脑脊液（CSF）或基因层面的生物标志物。由于不同模态的数据异构性极强，如何有效地在统一的表示空间中对其进行对齐与融合，一直是深度学习模型面临的巨大挑战。传统的拼接（Concatenation）或简单的注意力融合，容易忽略不同医学指征之间隐含的因果关联。

### 1.2 知识图谱引入的动机
医学知识图谱（Knowledge Graph, KG）以结构化三元组 $(h, r, t)$ 的形式包含了丰富的疾病、基因、药物和临床表型之间的关联。如果能将这些先验知识作为“背景词典”引入多模态模型中，便可有效指导模型发现隐藏在数据背后的病理机制。
在现有研究中，通常采用 TransE 这类基于平移的知识表示方法。但 TransE 的表达能力有限，无法处理复杂的 1-to-N 或 N-to-N 关系。为此，本研究提出引入卷积神经网络（CNN）机制的知识图谱嵌入策略（ConvE / ConvR），以期提取更为稠密和高阶的知识特征，从而进一步增强多模态 Transformer（KAMT）的表征能力。

---

## 2. 基础理论与方法论

### 2.1 知识图谱嵌入技术
知识图谱嵌入（Knowledge Graph Embedding, KGE）旨在将图谱中的实体和关系映射为低维连续向量。

#### 2.1.1 TransE 基线
TransE 的核心思想是平移假设：对于有效三元组 $(h, r, t)$，其向量表示应当满足 $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$。
其打分函数为：

$$
 f_r(h, t) = -\|\mathbf{h} + \mathbf{r} - \mathbf{t}\|_1 
$$

该模型简单高效，但在处理复杂一对多关系时容易产生冲突。

#### 2.1.2 ConvE 策略
ConvE 通过引入二维卷积操作，显式建模实体与关系在局部维度的交互。
具体而言，将头实体向量 $\mathbf{h}$ 和关系向量 $\mathbf{r}$ reshape 为二维矩阵，然后在高度维度拼接，通过 2D 卷积核提取特征：

$$
 f_r(h, t) = \sigma \left( \text{vec} \left( \sigma \left( [\overline{\mathbf{h}} ; \overline{\mathbf{r}}] * \mathbf{\omega} \right) \right) \mathbf{W} \right) \cdot \mathbf{t} 
$$

这种设计极大地提高了特征的非线性表达能力，同时显著减少了参数量。

#### 2.1.3 ConvR 策略 (提议的最优策略)
ConvR (Convolutional Relational Model) 进一步改进了 ConvE。它并非简单地拼接实体和关系，而是**利用关系向量 $\mathbf{r}$ 动态生成专属于该关系的卷积核**，再利用该卷积核对头实体 $\mathbf{h}$ 的二维表示进行卷积。

$$
 \mathbf{\omega}_r = \mathbf{r} \mathbf{W}_{gen} 
$$


$$
 f_r(h, t) = \sigma \left( \text{vec} \left( \overline{\mathbf{h}} * \mathbf{\omega}_r \right) \mathbf{W}_{proj} \right) \cdot \mathbf{t} 
$$

通过使卷积核“关系化”，ConvR 在多关系、高密度的医学知识图谱（如 DemRKG）中表现出比 ConvE 更好的特异性。

### 2.2 KAMT 多模态交互注意力机制
知识注入（Knowledge Injection）是本模型的核心环节。对于从原始数据（EHR, MRI, BIO）提取出的多模态特征 $X_m$，我们使用从 ConvR 导出的患者专属知识序列 $K$，进行多头交叉注意力计算（Cross-Attention）：

$$
 \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{X_m W_Q (K W_K)^T}{\sqrt{d_k}} \right) K W_V 
$$

这一过程使得模型能够自动关注与当前病理特征最相关的外部知识实体，完成知识驱动的特征增强。

---

## 3. ConvKAMT 架构设计

ConvKAMT（Convolutional Knowledge-Augmented Multimodal Transformer）是我们提出的完整诊断框架。其整体架构分为四个关键层次：

1. **输入与初步编码层**：EHR 和生物标志物通过 MLP 降维，MRI 影像通过定制的轻量级 3D-CNN 提取体素特征（参数量控制在 0.46M，算力损耗极低）。
2. **KG 嵌入层 (核心创新)**：预先使用 ConvE 或 ConvR 在构建好的 DemRKG 子图上训练，获取 128 维实体嵌入词典，并针对每个患者生成专属的知识图谱节点序列。
3. **知识注入层**：各模态向量作为 Query，提取出的知识序列作为 Key 和 Value，进行交叉注意力融合，得到融入先验医学知识的富模态表征。
4. **融合与分类层**：利用 Self-Attention 对多个知识增强后的模态进行高阶整合，最终经由全连接层输出 AD、MCI 和 NC 的三分类概率分布。

!
> *（注：架构图源码可见项目内的 ConvKAMT_architecture.md 文件）*

---

## 4. 实验设置与数据集

### 4.1 数据集描述
我们利用了国际最著名的四个脑疾病队列数据集：
- **ADNI** (Alzheimer's Disease Neuroimaging Initiative)：最权威的 AD 临床研究队列。
- **AIBL** (Australian Imaging, Biomarker & Lifestyle Flagship Study of Ageing)。
- **NACC** (National Alzheimer's Coordinating Center)。
- **PPMI** (Parkinson's Progression Markers Initiative)：作为泛退行性疾病泛化性验证集。

每个数据集均包含配套的 3D 脑影像数据、结构化临床评分及生物标志物数据。

### 4.2 知识图谱设置
所有实验均基于领域专属的 **DemRKG (Dementia-specific Relational Knowledge Graph)**。在此基础上，使用图谱构建模块抽取每个数据集中患者的特定局部子图。

### 4.3 训练配置
- **KGE (ConvE/ConvR)**：学习率 0.001，Batch Size 512，Embedding 维度 128，训练 200 Epochs。使用 MarginRanking Loss 进行负采样训练。
- **KAMT 分类器**：学习率 1e-5，权重衰减 1e-3，Batch Size 32，交叉熵损失函数，训练 200 Epochs。
- 引入了**断电续跑机制 (Checkpointing)** 以保障长周期实验的数据安全。

---

## 5. SOTA 横向对比实验分析

为了证明 ConvKAMT 的领先地位，我们将其与多类 SOTA 模型在相同的数据及划分下进行了横向对比：
1. **Baseline Single**: 仅使用单模态数据。
2. **Baseline Dual**: 简单双模态拼接。
3. **MM-Perceiver**: 业界领先的多模态特征融合网络，但不包含知识图谱先验。
4. **CustomKG / PrimeKG + TransE**: 使用普通或泛医学知识图谱结合 TransE 嵌入的变种。

### 5.1 F1 分数对比

![SOTA 比较 (F1)](d:/Polaris/Documents/work4/figures/sota_comparison_f1.png)

*图 1：各 SOTA 模型在 4 个数据集上的 F1 分数表现*

### 5.2 AUC-ROC 对比

![SOTA 比较 (AUC)](d:/Polaris/Documents/work4/figures/sota_comparison_auc.png)

*图 2：各 SOTA 模型在 4 个数据集上的 AUC-ROC 表现*

**结论分析**：
从上述图表中可直观地看出，带有 **DemRKG** 背景知识的模型显著优于无图谱或传统泛型图谱的模型。而在所有图谱融合方案中，**红色的 DemRKG + ConvE** 以及 **深红色的 DemRKG + ConvR**（我们的提议方案）在几乎所有评价维度上都占据最高点，完美验证了卷积机制比简单的 TransE 能够捕捉更有效的病理特征。

---

## 6. 深入消融分析

为了严格界定性能提升的来源，我们设计了两组针对性极强的消融实验。

### 6.1 知识源消融 (Knowledge Source Ablation)

探究“图谱内容本身”带来的影响（均统一采用 TransE 作为控制变量）。

![知识源消融实验](d:/Polaris/Documents/work4/figures/ablation_knowledge_source.png)

雷达图清晰地表明，专门为脑疾病构建的 **DemRKG** 显著优于自构建的简单网络（CustomKG）以及泛用性强但缺乏深度的 PrimeKG。

### 6.2 嵌入方法消融 (Embedding Method Ablation)

这也是本大作业的核心创新点论证，探究“在同一图谱下，不同嵌入转化技术的影响”。

![嵌入方法消融实验](d:/Polaris/Documents/work4/figures/ablation_embedding_method.png)

柱状图展示了在统一 DemRKG 框架下，TransE、ConvE、ConvR 之间的性能差异：
- **TransE** 作为基线表现尚可，但在处理复杂疾病关系时出现瓶颈。
- **ConvE** 通过 $3 \times 3$ 的 2D 卷积极大地缓解了参数爆炸，同时在 ADNI 数据集上实现了精度跃升。
- **ConvR** 凭借其“动态生成关系专属卷积核”的能力，在所有四个数据集上均取得了压倒性的优势，特别是在数据分布更为复杂的 AIBL 和 NACC 队列中，其特征抽取能力得到了极限释放。

---

## 7. 机制与有效性探讨

### 7.1 计算开销论证
ConvKAMT 不仅性能卓越，且设计极为轻量。
通过 `thop` 对 3D-CNN 与 KAMT Transformer 模块进行算力损耗分析表明：
- 影像特征提取模块：参数量 $\approx 0.46 \text{M}$，FLOPs $\approx 6.30 \text{G}$。
- 多模态融合模块：完全得益于 128 维向量的高度压缩，参数量同样处于极低水平。
这一特性使得该系统具备部署在临床前置工作站的现实可行性。

### 7.2 训练动态观察
从各个数据集在 200 Epoch 训练中的 Loss 与 AUC 曲线可以看出，使用了 ConvR 嵌入的模型在前期收敛速度显著快于 TransE 版本，且在验证集上呈现出更高的稳定性，不容易发生过拟合，这进一步归功于卷积操作本身自带的局部平滑和泛化特征。

---

## 8. 结论与未来展望

本报告详细呈现了 **ConvKAMT**——一种将知识图谱与多模态融合相结合的阿尔茨海默症诊断模型的设计全流程。
我们通过实施 ConvE 和 ConvR 替代传统的 TransE 技术，克服了长久以来基于平移的嵌入算法表达能力不足的痛点。通过对四大国际标准公开队列（ADNI, AIBL, NACC, PPMI）的完整复现与对比，ConvKAMT（特别是 ConvR 变种）确立了其 SOTA 级领先地位。

**未来工作**：
1. 可以进一步引入基于图注意力网络（GAT）或图神经网络（GCN）的嵌入方法与 ConvR 进行交叉验证。
2. 将多模态图谱技术扩展至其他神经退行性疾病（如帕金森综合症、额颞叶痴呆等），评估其在多分类和纵向转化（MCI to AD）预测中的价值。

---

> **代码复现与附录**  
> 所有支持本报告实验结果的数据预处理、训练代码（包含 `train_conve_convr.py`, `generate_figures.py` 及断电续跑机制）与模型权重，均已随本报告同步打包提交，并已通过实机环境验证。
