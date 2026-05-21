# PiDF 多组分氧化物 ABO3 钙钛矿实验说明

本仓库提供以下论文的实验代码和复现实验说明：

**PiDF: Interpretable Physics-Informed Descriptors for Stability Prediction and Screening of Multicomponent Oxide ABO3 Perovskites**

本研究提出了一种物理信息描述符框架，称为 **PiDF**，用于多组分氧化物 ABO3 钙钛矿的稳定性预测与主动筛选。实验流程包括基于描述符的分类预测、不同划分协议下的鲁棒性评估、描述符消融分析、主动学习模拟，以及大规模候选材料筛选。

---

## 1. 仓库结构

```text
.
├── exp1.py
├── plot_exp1.py
├── exp2.py
├── exp3.py
├── exp4.py
├── exp5.py
├── mp_data/
│   ├── exp42_perovskite_like_subset.csv
│   └── multicomponent_candidates.csv
├── runs/
│   ├── section1_logreg/
│   ├── section2_logreg/
│   ├── section3_logreg/
│   ├── section4_logreg/
│   └── section5_logreg/
└── README.md

## 2. 环境要求

实验代码使用 Python 实现，需要常见的科学计算和机器学习工具包。

推荐 Python 版本：

```bash
python >= 3.8
```

所需主要依赖包括：

```bash
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
```

可以使用以下命令安装依赖：

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
```

若使用 conda 虚拟环境，可以按照如下方式创建环境：

```bash
conda create -n pidf python=3.10
conda activate pidf
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
```

---

## 3. 数据集与标签定义

实验 1 到实验 4 使用的有标签数据集为：

```bash
./mp_data/exp42_perovskite_like_subset.csv
```

实验 5 使用的无标签候选材料池为：

```bash
./mp_data/multicomponent_candidates.csv
```

稳定性标签根据 energy above hull 标准构建：

```bash
--label-mode ehull
--ehull-threshold 0.05
```

这表示当某个材料的 energy above hull 不大于 **0.05 eV/atom** 时，该材料被视为近稳定材料。energy above hull 仅用于生成标签，不作为模型训练的输入描述符。

所有实验使用以下随机种子：

```bash
--seeds 0 1 2 3 4
```

这样可以对五次独立运行的结果进行平均，从而提高结果的可靠性。

---

## 4. 实验 1：基于 PiDF 的基础稳定性预测

实验 1 用于评估 PiDF 描述符在随机训练/测试划分下的基础预测性能。该实验使用逻辑回归模型作为分类器。

运行命令如下：

```bash
python exp1.py \
  --csv ./mp_data/exp42_perovskite_like_subset.csv \
  --outdir runs/section1_logreg \
  --split random \
  --label-mode ehull \
  --ehull-threshold 0.05 \
  --seeds 0 1 2 3 4 \
  --save-clean-dataset
```

实验完成后，运行以下命令生成相应图像：

```bash
python3 plot_exp1.py
```

输出文件保存于：

```bash
runs/section1_logreg/
```

该实验主要用于说明 PiDF 描述符是否能够为多组分氧化物 ABO3 钙钛矿的稳定性判别提供有效信息。

---

## 5. 实验 2：Host 和 Family 协议下的评估

实验 2 用于评估模型在不同数据划分协议下的鲁棒性。除常规设置外，该实验还考虑了更具挑战性的 host 和 chemical family 分组划分方式。

运行命令如下：

```bash
python3 exp2.py \
  --csv ./mp_data/exp42_perovskite_like_subset.csv \
  --outdir runs/section2_logreg \
  --protocols host family \
  --method-set full \
  --dedup-key auto \
  --label-mode ehull \
  --ehull-threshold 0.05 \
  --model-type logreg \
  --seeds 0 1 2 3 4
```

输出文件保存于：

```bash
runs/section2_logreg/
```

该实验用于检验当相似化合物或相关化学家族被分配到不同训练集和测试集时，基于描述符的模型是否仍然有效。

---

## 6. 实验 3：描述符与模型对比

实验 3 在相同评估协议下，对不同描述符表示或特征设置进行受控比较。该实验的目标是分析 PiDF 相关特征对最终预测性能的贡献。

运行命令如下：

```bash
python exp3.py \
  --csv ./mp_data/exp42_perovskite_like_subset.csv \
  --outdir runs/section3_logreg \
  --split random \
  --dedup-key auto \
  --label-mode ehull \
  --ehull-threshold 0.05 \
  --model-type logreg \
  --seeds 0 1 2 3 4 \
  --save-clean-dataset
```

输出文件保存于：

```bash
runs/section3_logreg/
```

该实验可用于分析所提出的 PiDF 表示是否能够在简单描述符基线之外提供额外信息。

---

## 7. 实验 4：主动学习模拟

实验 4 模拟稳定性预测中的主动学习过程。模型从一个较小的初始有标签集合开始，并根据混合查询策略逐轮选择新的样本加入训练。

运行命令如下：

```bash
python exp4.py \
  --csv ./mp_data/exp42_perovskite_like_subset.csv \
  --outdir runs/section4_logreg \
  --split random \
  --dedup-key auto \
  --label-mode ehull \
  --ehull-threshold 0.05 \
  --model-type logreg \
  --initial-size 20 \
  --query-size 10 \
  --n-rounds 8 \
  --selected-strategy hybrid_0.25_0.45_0.30 \
  --seeds 0 1 2 3 4 \
  --save-query-history \
  --save-clean-dataset
```

主动学习设置如下：

- 初始有标签样本数：`20`
- 每轮查询样本数：`10`
- 主动学习轮数：`8`
- 查询策略：`hybrid_0.25_0.45_0.30`

输出文件保存于：

```bash
runs/section4_logreg/
```

启用 `--save-query-history` 后，每一轮主动学习中被选择的样本也会被保存，便于后续分析。

该实验用于考察 PiDF 引导的学习策略是否能够提高稳定性预测中的样本利用效率。

---

## 8. 实验 5：大规模候选材料筛选

实验 5 将训练后的 PiDF 机器学习模型应用于无标签的多组分氧化物 ABO3 候选材料池。该实验的目标是筛选出具有潜力的候选材料，用于后续计算验证或实验验证。

运行命令如下：

```bash
python exp5.py \
  --labeled-csv ./mp_data/exp42_perovskite_like_subset.csv \
  --pool-csv ./mp_data/multicomponent_candidates.csv \
  --outdir runs/section5_logreg \
  --representation "PiDF + ML" \
  --label-mode ehull \
  --ehull-threshold 0.05 \
  --dedup-key auto \
  --pool-scoring-mode full_refit \
  --seeds 0 1 2 3 4 \
  --top-n 20 \
  --top-k-family 100 \
  --save-clean-datasets
```

有标签数据集为：

```bash
./mp_data/exp42_perovskite_like_subset.csv
```

无标签候选材料池为：

```bash
./mp_data/multicomponent_candidates.csv
```

输出文件保存于：

```bash
runs/section5_logreg/
```

该实验会给出由 PiDF 模型预测得到的高排名候选材料组成。其中，`--top-n 20` 用于保存排名前 20 的候选材料，`--top-k-family 100` 用于支持材料家族层面的候选分析。

---

## 9. 复现全部实验

若需要复现完整实验流程，请按照以下顺序运行命令：

```bash
python exp1.py \
  --csv ./mp_data/exp42_perovskite_like_subset.csv \
  --outdir runs/section1_logreg \
  --split random \
  --label-mode ehull \
  --ehull-threshold 0.05 \
  --seeds 0 1 2 3 4 \
  --save-clean-dataset

python3 plot_exp1.py

python3 exp2.py \
  --csv ./mp_data/exp42_perovskite_like_subset.csv \
  --outdir runs/section2_logreg \
  --protocols host family \
  --method-set full \
  --dedup-key auto \
  --label-mode ehull \
  --ehull-threshold 0.05 \
  --model-type logreg \
  --seeds 0 1 2 3 4

python exp3.py \
  --csv ./mp_data/exp42_perovskite_like_subset.csv \
  --outdir runs/section3_logreg \
  --split random \
  --dedup-key auto \
  --label-mode ehull \
  --ehull-threshold 0.05 \
  --model-type logreg \
  --seeds 0 1 2 3 4 \
  --save-clean-dataset

python exp4.py \
  --csv ./mp_data/exp42_perovskite_like_subset.csv \
  --outdir runs/section4_logreg \
  --split random \
  --dedup-key auto \
  --label-mode ehull \
  --ehull-threshold 0.05 \
  --model-type logreg \
  --initial-size 20 \
  --query-size 10 \
  --n-rounds 8 \
  --selected-strategy hybrid_0.25_0.45_0.30 \
  --seeds 0 1 2 3 4 \
  --save-query-history \
  --save-clean-dataset

python exp5.py \
  --labeled-csv ./mp_data/exp42_perovskite_like_subset.csv \
  --pool-csv ./mp_data/multicomponent_candidates.csv \
  --outdir runs/section5_logreg \
  --representation "PiDF + ML" \
  --label-mode ehull \
  --ehull-threshold 0.05 \
  --dedup-key auto \
  --pool-scoring-mode full_refit \
  --seeds 0 1 2 3 4 \
  --top-n 20 \
  --top-k-family 100 \
  --save-clean-datasets
```

---

## 10. 输出文件说明

每个实验都会在 `runs/` 目录下生成独立的输出文件夹。

输出文件可能包括：

- 用于训练和评估的清洗后数据集
- 模型预测结果
- 评价指标
- 汇总表格
- 主动学习过程中选中的样本
- 排名后的候选材料列表
- 用于论文绘图的图像文件

具体文件名会根据不同脚本和命令行参数有所不同。

---

## 11. 可复现性说明

为了提高实验结果的可复现性，所有实验均使用五个随机种子重复运行：

```bash
0, 1, 2, 3, 4
```

所有实验采用相同的标签定义：

```bash
energy_above_hull <= 0.05 eV/atom
```

所有主要实验采用相同的模型类型：

```bash
logreg
```

该设置有意采用简单且可解释的模型，从而更清楚地评估 PiDF 描述符本身的作用。
