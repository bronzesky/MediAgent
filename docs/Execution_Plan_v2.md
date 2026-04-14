# MediAgent 执行计划（V2，面向可发表证据）

> 目标：在不牺牲工程进度的前提下，把“创新点”转化为“可复现证据”，确保每个 claim 都有对应实验与验收门槛。
> 适配你现有三阶段路线（Phase1/2/3），但增加闸门和回退策略。

---

## 0A. 多Agent进度看板（2026-04-15）

> 状态定义：`✅ Done` / `🔄 In Progress` / `⏳ Pending` / `⚠️ Blocked`

| 模块 | 当前状态 | 当前产出 | 下一步（单一动作） | Owner |
|------|---------|---------|--------------------|-------|
| 数据下载-BRACS ROI | ✅ Done | `data/BRACS/train`, `data/BRACS/test` 已落地 | 校验文件完整性并固化 `manifest` | Agent-Data |
| PathoGraph知识库 | ✅ Done | `data/pathograph/pathograph_phenotypes.json` + schema | 补充同义词与排斥关系字段（如需） | Agent-KG |
| 模型权重准备 | ✅ Done | `models/conch`, `models/conchv1_5`, `models/TITAN` 本地化完成 | 写入统一权重路径配置 | Agent-Infra |
| 编码器统一加载 | ✅ Done | `PathoHGA/core/preprocessing/encoder_loader.py` + `scripts/5_probe_encoder.py` | 接入 patch feature 提取脚本 | Agent-Model |
| Graph Builder（Phase1入口） | ⏳ Pending | 仅有计划，无实现文件 | 新建 `core/preprocessing/graph_builder.py` 最小可跑版 | Agent-Graph |
| Baseline训练（Phase1闸门） | ⏳ Pending | 无训练日志/指标 | 完成一次端到端 smoke train（1 epoch） | Agent-Train |
| C1超图语义模块 | ⏳ Pending | 仅设计，无实现 | 新建 `core/models/hypergraph.py` 骨架 | Agent-C1 |
| C2对齐模块 | ⏳ Pending | 仅设计，无实现 | 新建 `core/models/alignment.py` 骨架 | Agent-C2 |
| C3 Agent推理 | ⏳ Pending | 仅设计，无实现 | 搭建 `graph_rag.py` 最小检索流程 | Agent-C3 |

### 当前主线所处步骤

- 当前位于：`Phase 1` 的“环境与依赖就绪”后半段，进入“数据构图与baseline复现”前置阶段。
- 阶段百分比（按可执行工作包粗估）：`~25%`（下载/权重/编码器完成；训练与评测尚未开始）。
- 最近完成里程碑：
  - BRACS ROI 落地
  - PathoGraph JSON + Schema
  - CONCH / CONCHv1.5 / TITAN 统一加载并 dry-run 通过

### 并行协作建议（避免冲突）

- `Agent-Graph` 与 `Agent-Train` 先定义统一 I/O 协议（`.pt` 字段名、label键、split键），再各自开发。
- `Agent-C1/C2` 暂不改动 dataloader，先写独立可单测模块，避免与 Phase1 冲突。
- 统一实验记录追加到 `results/registry.csv`（每次实验一行，包含 commit/seed/config/metrics）。

---

## 0. 先决原则（避免后期返工）

1. **单变量推进**：任何阶段只新增一个核心变量，保证增益可归因。
2. **先可跑，再最优**：先拿稳定 baseline，再做性能冲刺。
3. **患者级隔离**：所有 split、RAG 索引、调参过程严格 patient-level，避免信息泄漏。
4. **冻结优先**：CONCH/PLIP 优先冻结，只训练新增模块（超边、对齐头、分类头）。
5. **每周有硬输出**：每周必须产出“表格/图/日志”之一，能直接放进论文草稿。

---

## 1. 阶段闸门（Go / No-Go）

### Phase 1：HACT Baseline（1-2天）

**目标**
- 在 BRACS ROI 上跑通完整数据流：预处理 -> 训练 -> 推理 -> 指标。

**必须完成（Go）**
- 预处理产物完整：`cell_graphs / tissue_graphs / assignment_matrices`。
- baseline 可重复跑通（不同随机种子至少 3 次）。
- Weighted F1 达到 `>= 0.58`（你的 Demo 下限）。
- 训练日志可复现（固定 seed、固定 config、固定数据版本）。

**No-Go 条件**
- 跑不通或结果严重波动（std 过大），不得进入 Phase 2。

**输出物**
- `table_baseline_bracs.csv`
- `fig_baseline_confusion_matrix.png`
- `runbook_phase1.md`

---

### Phase 2A：C1（仅超图语义构建）

**目标**
- 验证“PathoGraph 语义超边”是否优于固定/随机超边。

**必须完成（Go）**
- C1 相比 baseline 有稳定提升（主要看 Weighted F1、Macro F1）。
- 关键消融完成：
  - 语义超边 vs 随机超边
  - 可学习超边 vs 固定超边
- 超边可视化可解释（至少 20 个 case 的人工核验样例）。

**No-Go 条件**
- C1 对比固定超边无显著增益。

**回退方案**
- 降低超边生成自由度：加入稀疏/熵正则、限制每节点可连接超边数。
- 从 cross-attention 简化到 prototype assignment，先保稳定。

---

### Phase 2B：C2（仅对齐模块）

**目标**
- 验证“超边-PathoGraph文本对齐”是否带来额外增益和可检索性提升。

**必须完成（Go）**
- 在 C1 基础上加入 C2 后，分类指标进一步提升或至少不下降。
- 检索评估：文本查询命中正确表型超边的 Recall@K 提升。
- 对齐消融完成：
  - 无对齐
  - 报告文本锚点对齐
  - PathoGraph 文本锚点对齐（你的主张）

**No-Go 条件**
- 对齐引入后训练不稳定或指标持续下降。

**回退方案**
- 先固定文本编码器，仅训练投影头。
- 调低对齐损失权重 λ，采用 warmup（前 N epoch 仅分类）。

---

### Phase 3：C3 Agent（1-2周）

**目标**
- 验证“结构化三阶段推理 + 图硬约束验证器”确实降低幻觉并提升诊断质量。

**必须完成（Go）**
- 评估集上对比：
  - 无约束 Agent
  - LLM 自检软约束
  - 图拓扑硬约束（你的方法）
- 幻觉率显著下降，同时诊断准确率不下降。
- 报告可追溯：每个关键结论能映射到超边 ID 或证据 patch。

**No-Go 条件**
- 幻觉率下降但准确率显著劣化（过度保守）。

**回退方案**
- 调整 verifier 阈值 τ，做 τ-性能曲线，选择 Pareto 点。
- verifier 从“硬拒绝”改“置信重加权 + 二次检索”。

---

## 2. Claim-Experiment 对照矩阵（论文主线）

| Claim | 必做实验 | 最低通过标准 |
|------|---------|------------|
| C1：语义驱动超边有效 | C1 vs 随机/固定超边 | 主指标稳定优于对照 |
| C2：知识锚定对齐有效 | PathoGraph锚点 vs 报告锚点 vs 无对齐 | 检索与分类至少一项显著提升 |
| C3：可验证推理降低幻觉 | 硬约束 vs 软约束 vs 无约束 | 幻觉率下降且准确率不降 |
| 系统整体有效 | C1+C2+C3 vs 现有SOTA | 主任务达到可发表竞争力 |
| 泛化能力 | BRACS -> TCGA-BRCA（可加跨癌种） | 保持相对优势，不崩溃 |

---

## 3. 最小可发表实验包（优先级）

### P0（必须）
- BRACS：baseline、C1、C1+C2、C1+C2+C3 全链路。
- 全部关键消融（Contributions 文档里列出的核心项）。
- 3 seed 重复实验 + 方差报告。

### P1（强烈建议）
- TCGA-BRCA 外部验证。
- Agent 指标：准确率 + 幻觉率 + 报告质量（BLEU/ROUGE/METEOR）。

### P2（加分项）
- 跨癌种（LUAD/LUSC）泛化。
- 成本-延迟-性能曲线（API 成本透明化）。

---

## 4. 工程实施顺序（代码层面）

1. 锁定数据与配置版本（`configs/` + `data manifest`）。
2. 先完成 `core/preprocessing/graph_builder.py` 的稳定产出。
3. 基于 `core/train.py` 跑通 baseline。
4. 新增 `core/models/hypergraph.py`，仅接 C1。
5. 新增 `core/models/alignment.py`，接入 C2（带 λ warmup）。
6. 最后接入 `core/agent/graph_rag.py` 与 `core/agent/reasoning.py`。
7. 每步都产出独立可复现实验记录，不跨阶段混改。

---

## 5. 6周时间线（可压缩）

- Week 1：Phase 1 全通过 + baseline 图表产出。
- Week 2-3：C1 完整实现与消融。
- Week 4：C2 对齐实现与消融。
- Week 5：C3 Agent 与 verifier、阈值分析。
- Week 6：外部验证 + 论文主表/主图定稿。

---

## 6. 风险与提前止损

### 风险1：环境不稳定（CUDA 12.8 与依赖版本）
- 动作：首日完成 `1 batch smoke test` + `1 epoch dry run`。
- 止损：无法稳定则先降版本到已验证组合（PyTorch/PyG 对齐）。

### 风险2：C1 提升不显著
- 动作：先强化监督信号质量（表型 JSON、标签清洗）。
- 止损：保留 C1 作为可解释模块，主增益由 C2/C3 提供。

### 风险3：C3 增益被质疑 prompt trick
- 动作：固定 LLM、固定提示模板，仅切换“约束机制”。
- 止损：重点突出“外部结构验证”而非生成技巧。

---

## 7. 本周执行清单（你现在就能做）

1. 完成 `pathograph_phenotypes.json` 初版（先覆盖 BRACS 相关高频表型）。
2. 跑通 Phase 1，拿到第一版 baseline 指标和混淆矩阵。
3. 预留 C1 接口：在 dataloader 中加 `phenotype_json` 读取与映射。
4. 确定 C1 的 2 个必做消融配置文件（随机超边、固定超边）。
5. 建一个 `results/registry.csv`，统一记录每次实验（config、seed、指标、commit）。

---

## 8. 通过标准（是否进入“投稿准备”）

满足以下 4 条即可进入投稿准备：
- 三个 claim 都有独立实验证据。
- 至少一个外部数据集验证通过。
- 关键结果具备 3 seed 稳定性。
- 失败案例和局限性分析完整可写进 Discussion。
