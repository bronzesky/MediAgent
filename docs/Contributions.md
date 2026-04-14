# MediAgent Contributions

> 目标期刊：Medical Image Analysis (MIA) 或 IEEE TMI  
> 任务：乳腺癌 WSI 诊断（亚型分类 + 报告生成）  
> 数据集：BRACS + TCGA-BRCA（训练/验证），WSI-Bench + MultiPathQA（Agent能力评估）

---

## 竞品 Novelty 核查

| 已有工作 | 做了什么 | 没做什么 |
|---------|---------|---------|
| HACT-Net (MedIA 2022) | 层次细胞-组织图，固定KNN/RAG连边 | 无超图，无语言对齐，无Agent |
| DyHG (2501.16787) | 动态超图+Gumbel-Softmax，MIL分类 | 纯数据驱动，无知识本体，无语言，无Agent |
| WSI-HGMamba (2505.17457) | 超图+Mamba，纯分类 | 无语言对齐，无知识本体，无Agent |
| MLLM-HWSI (2603.23067) | 四粒度层次对比对齐（cell/patch/region/WSI）+报告生成 | 无图结构，对齐锚点是报告文本非知识本体，无Agent |
| Sigmma (2511.15464) | 多尺度图+HE-转录组对比对齐 | 无病理知识本体，无Agent，无报告生成 |
| PathFinder (2502.08916) | 多Agent串行导航WSI | 无图结构，无拓扑约束 |
| WSI-Agents (2507.14680) | 多Agent协作+LLM一致性验证（软约束） | 无图拓扑硬约束，验证依赖LLM自身 |
| SurvAgent (2511.16635) | CoT病例库+向量RAG | 无图结构，向量检索非图语义 |
| LAMMI-Pathology (2602.18773) | 工具调用Agent+AEN轨迹 | 无图结构约束，无视觉-语言图对齐 |
| PathoGraph (Nat. Sci. Data 2025) | 病理知识本体/Schema定义 | 无自动实例化，无可训练模块，无Agent |

---

## 三个主要 Contribution

### Contribution 1（主）：PathoGraph语义驱动的可学习超图构建

**覆盖原C1+C2，是核心方法创新。**

**动机**：现有图方法（HACT、DyHG）的超边/连边是空间距离或数据驱动的，无病理语义。PathoGraph定义了multi-cell phenotype是多个实体的组合，天然对应超边语义，但从未被自动实例化为可训练的图结构。

**方法**：
- 以PathoGraph的Phenotype Graph为schema，定义超边语义（超边 = 一个multi-cell phenotype实例）
- 细胞检测（HoVer-Net）→ 细胞节点；组织分割（SLIC）→ 组织节点
- **可学习超边生成**：cross-attention动态生成incidence matrix，决定哪些细胞归属同一超边
- 监督信号：BRACS ROI标注 + PathoGraph表型文本弱监督（见C2）
- 区别于DyHG的Gumbel-Softmax：DyHG纯数据驱动，本工作以病理知识本体为约束

**Novelty确认**：PathoGraph从未被实例化为可训练图结构；DyHG/WSI-HGMamba的超图无知识本体驱动。

---

### Contribution 2（主）：PathoGraph锚定的多粒度视觉-语言对齐

**覆盖原C3，是方法创新，区别于MLLM-HWSI和Sigmma。**

**动机**：
- 细胞节点（形态数值特征）、组织节点（CONCH patch特征）、超边（聚合特征）三个粒度的特征空间不统一，无法与文本语义直接比较
- MLLM-HWSI做了多粒度对齐，但锚点是报告中的自然语言（数据驱动），不是结构化知识
- PathoGraph为每个表型提供了精确的形态学文本定义，是更可靠的对齐锚点

**方法**：

```
特征初始化（三粒度保留）：
  细胞节点：HoVer-Net形态特征 + 细胞patch CNN特征 → MLP投影到d维
  组织节点：CONCH patch特征 → 线性投影到d维
  超边：成员细胞节点attention聚合 → d维 → 投影头映射到CONCH文本空间

对齐目标（InfoNCE，仅在超边层次）：
  超边表示 ↔ PathoGraph对应表型的文本描述（CONCH text encoder编码）
  正样本：超边 ↔ 其对应的PathoGraph表型文本
  负样本：超边 ↔ 其他表型文本
```

**与MLLM-HWSI的关键区别**：
- MLLM-HWSI：patch/region → 报告句子（数据驱动，锚点质量依赖报告标注）
- 本工作：超边（病理表型）→ PathoGraph结构化定义（知识驱动，锚点语义精确）
- 对齐后超边表示可直接用于C3的验证器，无需额外转换

**Novelty确认**：无工作在超图超边层次做PathoGraph知识本体锚定的对齐。

---

### Contribution 3（主）：PathoGraph诊断流程引导的可验证Agent推理

**覆盖原C4+C5，是系统创新。**

**动机**：
- 现有Agent（PathFinder、WSI-Agents）推理是自由的或靠LLM软约束，无结构化诊断流程
- WSI-Agents的验证依赖LLM内部一致性，仍可产生幻觉
- PathoGraph的Diagnosis Graph定义了病理诊断的阶段性流程，是天然的Agent推理骨架

**方法**：

```
三阶段结构化推理（PathoGraph Diagnosis Graph）：

[阶段1 Preliminary Diagnosis]
  输入：组织节点全局特征 + Graph-RAG检索（WL kernel匹配历史病例超图）
  LLM输出：① 鉴别诊断列表  ② 需进一步查询的表型列表  ③ 推理依据（引用超边ID）

[阶段2 Further Diagnosis]
  输入：阶段1指定的目标超边细节 + PathoGraph量化参数
  LLM输出：① 缩小的诊断可能性  ② 关键特征量化描述  ③ 推理依据

[阶段3 Final Diagnosis]
  输入：阶段1+2全部中间输出
  LLM输出：① 最终亚型分类  ② 结构化报告  ③ 完整推理链（可追溯到超边）

图拓扑硬约束验证器（每阶段输出后执行）：
  报告描述 → CONCH text encoder → 在Hypergraph中检索最近邻超边
  相似度 < τ → 判定幻觉，返回重生成 + 反馈（"图中未找到X表型的证据"）
  验证的数学基础：C2保证超边表示与文本在同一CONCH空间
```

**与WSI-Agents的关键区别**：
- WSI-Agents：LLM内部一致性检查（软约束，仍依赖LLM判断）
- 本工作：图拓扑硬约束（外部结构验证，不依赖LLM自身）

**Novelty确认**：无工作将PathoGraph诊断流程作为Agent骨架；无工作用图拓扑做LLM输出硬约束。

---

## 架构总览

```
WSI 输入
  ↓
[C1] PathoGraph语义驱动的可学习超图构建
     细胞节点（形态+CNN特征）+ 组织节点（CONCH特征）
     可学习超边生成（attention，PathoGraph表型弱监督）
  ↓
[C2] PathoGraph锚定的多粒度视觉-语言对齐
     超边聚合表示 → 投影到CONCH空间
     InfoNCE: 超边 ↔ PathoGraph表型文本
  ↓
[C3] PathoGraph诊断流程引导的可验证Agent推理
     Gemini 2.5 Pro + 三阶段结构化推理
     + Graph-RAG（WL kernel历史病例检索）
     + 图拓扑硬约束验证器（每阶段）
  ↓
输出：亚型分类 + 结构化诊断报告 + 完整推理链
```

---

## 对比实验设计

### 分类任务（BRACS / TCGA-BRCA）
| 类别 | 方法 |
|------|------|
| 传统MIL | ABMIL, TransMIL, HIPT, CONCH+ABMIL, UNI+ABMIL |
| 病理MLLM | Quilt-LLaVA, WSI-LLaVA, SlideChat, MLLM-HWSI |
| 病理Agent | PathFinder, CPathAgent, WSI-Agents, GIANT, SlideSeek |
| 通用大模型 | GPT-4o+GIANT框架, Gemini 2.5 Pro（无图约束版） |

### Agent能力评估（WSI-Bench / MultiPathQA）
- 指标：诊断准确率、BLEU/ROUGE/METEOR（报告质量）、幻觉率

### 消融实验
| 消融项 | 验证的Contribution |
|--------|-------------------|
| 语义超边 → 随机超边 | C1：PathoGraph语义的必要性 |
| 可学习超边 → 固定超边 | C1：可学习生成的必要性 |
| PathoGraph对齐 → 无对齐 | C2：知识锚定对齐的必要性 |
| PathoGraph对齐 → MLLM-HWSI式对齐 | C2：知识本体锚点 vs 报告文本锚点 |
| 三阶段推理 → 自由生成 | C3：诊断流程结构的必要性 |
| 硬约束验证器 → 软约束（LLM自检） | C3：图拓扑硬约束 vs 软约束 |
| w/o Graph-RAG | C3：历史病例检索的贡献 |

---

## 技术风险

| 风险 | 等级 | 应对 |
|------|------|------|
| C1弱监督标签质量 | 中 | 预实验：BRACS ROI标注能否生成足够超边监督信号 |
| C3验证器阈值τ敏感性 | 低 | 验证集调参，报告τ的敏感性分析 |
| CONCH访问权限 | 中 | 申请CONCH；备选：PLIP（完全开源）或UNI |
| Gemini API成本 | 低 | 评估阶段用API，训练阶段图模块本地跑 |
