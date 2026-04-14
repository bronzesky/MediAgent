# PathoHGA 实现计划

> 版本：v1.0 | 日期：2026-04-08  
> 服务器：4× RTX 4090，CUDA 12.8  
> 基础代码：hact-net（DGL）→ 迁移至 PyG

---

## 0A. 多Agent进度看板（2026-04-15）

> 状态定义：`✅ Done` / `🔄 In Progress` / `⏳ Pending` / `⚠️ Blocked`

| 工作包 | 状态 | 当前结果 | 下一步 | Owner |
|---|---|---|---|---|
| BRACS ROI 数据准备 | ✅ Done | `data/BRACS/train`、`data/BRACS/test` 已到位 | 输出文件清单与样本计数 | Agent-Data |
| PathoGraph 表型库 | ✅ Done | `data/pathograph/pathograph_phenotypes.json` + schema 完成 | 增补同义词/互斥约束（按需） | Agent-KG |
| 编码器与权重接入 | ✅ Done | CONCH/CONCHv1.5/TITAN 本地权重可加载 | 接入批量特征提取脚本 | Agent-Model |
| 统一加载器验证 | ✅ Done | `core/preprocessing/encoder_loader.py` + `scripts/5_probe_encoder.py` dry-run 全通过 | 增加最小前向样例与错误处理 | Agent-Model |
| Phase1 构图脚本 | ⏳ Pending | `graph_builder.py` 尚未实现 | 先做最小可跑版（ResNet+坐标+PyG保存） | Agent-Graph |
| Phase1 Baseline 训练 | ⏳ Pending | 尚无训练日志与F1 | 先跑 1 epoch smoke test | Agent-Train |
| Phase2 C1 超图模块 | ⏳ Pending | 仅设计，无代码 | 建 `hypergraph.py` 骨架并可单测 | Agent-C1 |
| Phase2 C2 对齐模块 | ⏳ Pending | 仅设计，无代码 | 建 `alignment.py` 骨架并可单测 | Agent-C2 |
| Phase3 C3 Agent 推理 | ⏳ Pending | 仅设计，无代码 | 先做 `graph_rag.py` 检索最小闭环 | Agent-C3 |

### 当前阶段定位

- 当前处于 `Phase 1`：前置资源准备已完成，正进入“构图 + baseline复现”。
- 进度粗估：约 `25%`（数据/权重/加载器完成；训练评测未开始）。

### 协作约束（多Agent并行）

- 统一 I/O 协议：`HeteroData` 字段名、label 键、split 键先冻结再开发。
- `Agent-C1/C2` 暂不改 `dataloader.py`，先独立模块化实现，避免阻塞 Phase1。
- 所有实验必须登记到 `results/registry.csv`（config、seed、metrics、commit）。

---

## 0. 总体策略

三阶段推进，每阶段有明确的可验证里程碑：

```
Phase 1（1-2天）   ：hact-net baseline PyG复现 → Weighted F1 ≥ 0.58
Phase 2（2-3周）   ：PathoHGA核心模型 → C1+C2，F1超过baseline
Phase 3（1-2周）   ：Agent推理模块 → C3，在WSI-Bench上评估
```

---

## 1. 目录结构

```
PathoHGA/
├── environment.yml
├── IMPLEMENTATION_PLAN.md          ← 本文件
├── data/
│   ├── pathograph_phenotypes.json  ← 手动整理（约20-30条乳腺癌表型）
│   └── target.png                  ← 染色归一化目标图（从hact-net复制）
├── configs/
│   ├── bracs_baseline.yml          ← Phase 1：HACT baseline复现配置
│   ├── bracs_pathoHGA.yml          ← Phase 2：完整模型配置
│   └── agent.yml                   ← Phase 3：Agent推理配置
├── core/
│   ├── preprocessing/
│   │   ├── graph_builder.py        ← 主预处理脚本（改自generate_hact_graphs.py）
│   │   └── conch_extractor.py      ← CONCH/PLIP特征提取封装
│   ├── models/
│   │   ├── hypergraph.py           ← C1核心：可学习超边生成模块
│   │   ├── alignment.py            ← C2核心：InfoNCE超图-语言对齐
│   │   └── pathoHGA.py             ← 完整模型（整合C1+C2）
│   ├── agent/
│   │   ├── graph_rag.py            ← C3：WL kernel病例库构建与检索
│   │   └── reasoning.py            ← C3：LangGraph三阶段推理+验证器
│   ├── dataloader.py               ← 数据加载（PyG格式）
│   └── train.py                    ← 训练脚本（DDP，消融开关）
├── scripts/
│   ├── 0_setup_env.sh
│   ├── 1_preprocess_bracs.sh
│   ├── 2_train_baseline.sh
│   ├── 3_train_pathoHGA.sh
│   └── 4_eval_agent.sh
├── checkpoints/
└── logs/
```

---

## 2. 数据流总览

```
BRACS ROI .png
    │
    ▼ graph_builder.py
    ├── 细胞节点特征：HoVer-Net形态(32d) + ResNet18 patch(512d) → MLP → 256d
    ├── 组织节点特征：CONCH/PLIP 256×256 patch → 512d → Linear → 256d
    ├── 细胞kNN图（k=5, thresh=50px）
    ├── 组织RAG图（邻接超像素）
    └── assignment matrix（细胞→组织归属）
    │
    ▼ 保存为 PyG .pt 格式（非DGL .bin）
    │
    ▼ dataloader.py
    │
    ▼ pathoHGA.py（训练）
    ├── C1: hypergraph.py → incidence matrix B（N_cell × K）
    ├── HypergraphConv消息传递（PyG）
    ├── C2: alignment.py → L_align（超边 ↔ PathoGraph文本）
    └── 分类头 → L_cls
    │
    L = L_cls + λ * L_align
    │
    ▼ 训练完成后 → graph_rag.py
    ├── 训练集超图 → WL kernel索引
    └── 训练集推理轨迹（Gemini生成CoT）
    │
    ▼ reasoning.py（推理）
    └── 三阶段LangGraph + 验证器 → 诊断报告
```

---

## 3. Phase 1：Baseline 复现

### 3.1 目标
DGL → PyG 格式迁移，验证 HACT 的 GNN 逻辑在新框架下复现，F1 ≥ 0.58。

### 3.2 graph_builder.py 的改动

**原始（hact-net）**：
- 细胞特征：ResNet34 (512d) + 归一化坐标 (2d) = 514d
- 组织特征：ResNet34 (512d) + 归一化坐标 (2d) = 514d
- 保存格式：DGL `.bin` + h5

**Phase 1 改动（最小改动）**：
- 特征不变（ResNet34 + 坐标）
- **仅修改保存格式**：`torch_geometric.data.HeteroData` 保存为 `.pt`
- 图结构不变（kNN + RAG + assignment_matrix）

```python
# 核心数据结构（Phase 1）
data = HeteroData()
data['cell'].x         # (N_cell, 514)   细胞节点特征
data['tissue'].x       # (N_tissue, 514) 组织节点特征
data['cell', 'knn', 'cell'].edge_index      # (2, E_cg)
data['tissue', 'rag', 'tissue'].edge_index  # (2, E_tg)
data['cell', 'belongs', 'tissue'].edge_index # assignment matrix稀疏化
data.y                 # scalar label
```

**为什么选 PyG 而非继续 DGL**：
- PyG `HypergraphConv` 是 Phase 2 的核心算子，DGL 无直接对应实现
- PyG `HeteroData` 对异构图的支持更简洁
- torch_geometric 生态（scatter, MessagePassing）与 C1 的 cross-attention 融合更自然

### 3.3 dataloader.py

```python
class BRACSDataset(Dataset):
    # 加载 .pt 文件，返回 HeteroData
    # 支持：cell-only / tissue-only / HACT 三种模式
    pass

def collate_hetero(batch):
    # PyG 的 Batch.from_data_list(batch)
    pass
```

### 3.4 pathoHGA.py（Phase 1：baseline 部分）

Phase 1 只实现 HACT backbone，不含 C1/C2：

```python
class HACTBaseline(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_classes):
        # cg_gnn: PNA (复用config参数)
        # tg_gnn: PNA
        # cross-attention fusion (cell→tissue via assignment)
        # classifier
        pass

    def forward(self, data: HeteroData):
        # 1. cell graph → cell embeddings
        # 2. tissue graph → tissue embeddings
        # 3. assignment-based fusion
        # 4. global readout → classify
        pass
```

**关键实现决策**：用 PyG 的 `SAGEConv` 或 `PNAConv` 替代 histocartography 封装的 PNA 层，直接复用 hact-net 的 config 参数（layer_type, aggregators, scalers）。

---

## 4. Phase 2：PathoHGA 核心模型

### 4.1 graph_builder.py 的增量改动（Phase 2 特定）

**在 Phase 1 基础上替换特征提取**：

| 节点类型 | Phase 1 | Phase 2 |
|---------|---------|---------|
| 细胞节点 | ResNet34 512d + 坐标 2d | HoVer-Net形态 32d + ResNet18 512d + 坐标 2d → MLP → 256d |
| 组织节点 | ResNet34 512d + 坐标 2d | CONCH/PLIP 512d + 坐标 2d → Linear → 256d |

**HoVer-Net 形态特征（32维）**：
```
细胞核：面积(1) + 周长(1) + 长轴(1) + 短轴(1) + 圆度(1) + 偏心率(1) + 
        实度(1) + 凸包面积比(1) + 长短轴比(1) + 核质比(近似)(1) +
        染色强度均值(1) + 染色强度标准差(1) + 核型one-hot(8) +
        梯度幅值均值(1) + 细胞类型概率分布(8) = 32d
```
→ HoVer-Net 已输出 `inst_type`（细胞类型）和 `inst_centroid`，形态特征通过 OpenCV/skimage regionprops 计算

**CONCH 特征提取**：
```python
# conch_extractor.py
class CONCHExtractor:
    # 优先：CONCH (MahmoodLab/CONCH)
    # 备选：PLIP (vinid/plip)
    # 接口统一，调用方无需区分

    def extract_tissue_features(self, image, superpixels) -> torch.Tensor:
        # 256×256 patch → CONCH image encoder → 512d
        pass

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        # PathoGraph表型文本 → CONCH text encoder → 512d
        pass
```

**设计决策**：
- CONCH/PLIP 权重完全冻结，只做特征提取（不微调）
- 用统一的 `VLPExtractor` 接口封装，方便消融时切换 PLIP/UNI

### 4.2 hypergraph.py（C1 核心）

**完整实现计划**：

```python
class LearnableHyperedgeGenerator(nn.Module):
    """
    输入：细胞节点特征矩阵 H_c (N × d)
    输出：incidence matrix B (N × K), B[i,k] ∈ [0,1]
    
    原理：PathoGraph的K个表型 → 每个表型对应一个可学习的"表型查询向量" q_k
         cross-attention(q_k, H_c) → B[:, k]（每个细胞属于表型k的概率）
    
    训练信号来源：
    1. C2 InfoNCE loss（语义约束，主要信号）
    2. 分类 loss 反向传播
    3. 可选：BRACS ROI区域弱监督（如果标注可用）
    """
    def __init__(self, node_dim: int, num_phenotypes: int, temp: float = 1.0):
        # phenotype_queries: (K, d) 可学习参数，初始化为PathoGraph文本embedding
        # cross_attn: MultiheadAttention(d, num_heads=4)
        # gumbel_temp: 温度参数（训练时退火）
        pass

    def forward(self, h_c: Tensor, hard: bool = False) -> Tensor:
        # Q: phenotype_queries (K, d)
        # K,V: h_c (N, d)
        # attn_weights: (K, N) → transpose → (N, K)
        # Gumbel-Softmax(attn_weights, dim=1) → B (N, K)
        # 返回 B: 每列是一个超边的成员概率分布
        pass
```

**初始化策略**：`phenotype_queries` 用 PathoGraph 表型文本的 CONCH 文本 embedding 初始化（然后可训练），这确保超边从一开始就有病理语义。

**超边数量 K**：从 `pathograph_phenotypes.json` 读取，乳腺癌相关表型约 20-30 个。

**与 DyHG 的实现区别**：
- DyHG：学习 K 个可训练向量（纯随机初始化），纯数据驱动
- 本工作：K 由 PathoGraph 固定，查询向量用 PathoGraph 文本初始化（知识驱动）

```python
class HypergraphConvBlock(nn.Module):
    """
    用 PyG 的 HypergraphConv 实现层次消息传递。
    消息传递顺序：
    1. 细胞 → 超边（聚合各超边的成员细胞）
    2. 超边 → 细胞（更新细胞特征）
    3. 超边 → 组织（通过assignment matrix的超边级别汇总）
    """
    def __init__(self, in_channels, out_channels, num_hyperedges):
        # PyG HypergraphConv(in_channels, out_channels, use_attention=True)
        # LayerNorm, dropout, residual
        pass

    def forward(self, x_cell, hyperedge_index, hyperedge_attr, x_tissue, assign_mat):
        # hyperedge_index: (2, nnz) - 从incidence matrix B提取的稀疏表示
        # hyperedge_attr: (K, d) - 超边聚合特征（由B加权求和细胞特征得到）
        pass
```

**incidence matrix → hyperedge_index 的转换**：
```python
# B: (N, K) dense → 稀疏化
# threshold: 取 top-k cells per hyperedge，或阈值截断
def b_to_hyperedge_index(B: Tensor, threshold: float = 0.1):
    row, col = (B > threshold).nonzero(as_tuple=True)
    # hyperedge_index[0] = row (node indices)
    # hyperedge_index[1] = col (hyperedge indices)
    return torch.stack([row, col])
```

### 4.3 alignment.py（C2 核心）

```python
class PhenotypicAlignment(nn.Module):
    """
    超边表示 ↔ PathoGraph表型文本 的 InfoNCE 对比对齐。
    
    核心假设：
    - 超边 e_k 对应 PathoGraph 表型 p_k
    - 训练后 proj(h_e_k) 与 CONCH_text(p_k) 在同一空间中相近
    - 这是 C3 验证器的数学基础：图中超边 ↔ 报告文本 可相互检索
    
    与 MLLM-HWSI 的区别（需要在 Paper 中明确量化）：
    - MLLM-HWSI 对齐的是 patch/region → 报告句子（数据驱动，K=报告句数）
    - 本工作对齐的是 超边 → PathoGraph表型定义（知识驱动，K=表型数目≈20-30）
    - 更少的锚点，更精确的语义
    """
    def __init__(self, hyperedge_dim: int, text_dim: int, temperature: float = 0.07):
        # proj: Linear(hyperedge_dim, text_dim) + L2Norm
        # text_embeddings: (K, text_dim) 预计算并冻结
        # temperature: learnable or fixed
        pass

    def forward(self, h_e: Tensor, text_embeddings: Tensor) -> Tensor:
        # h_e: (K, d) 超边聚合表示
        # text_embeddings: (K, d_text) PathoGraph文本encoding（预计算）
        # proj_e = self.proj(h_e)  # (K, d_text)
        # InfoNCE: 超边k的正样本 = 第k个文本
        # L_align = -mean(log(exp(sim(proj_e[k], t[k])/τ) / Σ_j exp(sim(proj_e[k], t[j])/τ)))
        pass
```

**文本 embedding 的预计算**：
```python
# 在训练开始前一次性计算，固定不变（CONCH text encoder冻结）
text_embeddings = conch.encode_text(phenotype_texts)  # (K, 512)
# 保存到 data/pathograph_phenotypes_embeddings.pt
```

### 4.4 pathoHGA.py（整体模型）

```python
class PathoHGA(nn.Module):
    """
    完整模型，整合 C1 + C2。
    
    前向流程：
    1. 输入：HeteroData（细胞图 + 组织图 + assignment）
    2. 细胞特征 → LearnableHyperedgeGenerator → incidence matrix B
    3. B → 超边聚合特征 H_e
    4. HypergraphConvBlock × L → 更新 H_c, H_e
    5. H_e 同时送入 alignment.py 计算 L_align
    6. H_t（组织特征）通过 B 和 assignment 融合
    7. 全局 readout → 分类头 → L_cls
    输出：logits, L_align（训练时），或 logits（推理时）
    """
    def __init__(self, config: dict):
        # cell_proj: MLP(cell_feat_dim → hidden_dim)
        # tissue_proj: Linear(tissue_feat_dim → hidden_dim)
        # hyperedge_gen: LearnableHyperedgeGenerator
        # hgconv_layers: nn.ModuleList of HypergraphConvBlock
        # alignment: PhenotypicAlignment
        # classifier: MLP(hidden_dim → num_classes)
        pass

    def forward(self, data: HeteroData, text_embeddings: Tensor = None):
        pass
```

**训练时 loss**：
```python
L = L_cls + args.lambda_align * L_align
# lambda_align：从配置文件读取，推荐初始值 0.1，消融时设为 0
```

**消融开关（在 config 文件中控制）**：
```yaml
# bracs_pathoHGA.yml
model:
  use_semantic_init: true      # False → 随机初始化 phenotype_queries（消融C1语义性）
  use_learnable_hyperedge: true # False → 固定 KNN 超边（消融C1可学习性）
  use_alignment: true          # False → lambda_align=0（消融C2）
  alignment_anchor: "pathograph"  # "report" → MLLM-HWSI式消融
lambda_align: 0.1
gumbel_temp_init: 2.0
gumbel_temp_final: 0.5
gumbel_anneal_epochs: 30
```

---

## 5. Phase 3：Agent 推理模块

### 5.1 graph_rag.py（病例库）

```python
class CaseBank:
    """
    训练完成后构建，不参与模型训练。
    
    内容：
    - 训练集每个样本的超图结构（incidence matrix B + 节点特征）
    - 对应的 Gemini 生成的 CoT 推理轨迹
    - 真实诊断标签
    
    检索方式：WL graph kernel + FAISS
    """

    def build(self, train_loader, model, gemini_client):
        """
        对训练集所有样本：
        1. 前向传播提取超图结构
        2. 离散化节点特征（k-means聚类 → 整数标签）
        3. 计算 WL kernel 特征向量（grakel）
        4. 调用 Gemini 生成 CoT 推理轨迹
        5. 保存到 data/case_bank/
        """
        pass

    def retrieve(self, query_hypergraph, k: int = 3):
        """
        输入：查询样本的超图
        输出：top-k 相似历史病例（含 CoT 轨迹）
        
        步骤：
        1. 查询超图 → WL 特征向量
        2. FAISS cosine 搜索 → top-k 候选
        3. 返回：[(诊断标签, CoT轨迹, 相似度), ...]
        """
        pass
```

**WL kernel 实现细节**：
- 节点标签离散化：将 256d 节点特征用 k-means（k=50）量化为整数标签
- grakel `WeisfeilerLehman(n_iter=3, base_graph_kernel=VertexHistogram)` 计算图核矩阵
- WL 特征向量缓存，FAISS IndexFlatIP 构建检索索引

**与 SurvAgent 的区别**：
- SurvAgent：FAISS 向量相似度（特征空间距离）
- 本工作：WL kernel（图拓扑结构相似度），捕捉细胞微环境模式

### 5.2 reasoning.py（三阶段 LangGraph）

**State 定义**：
```python
class DiagnosisState(TypedDict):
    # 输入
    hypergraph: HeteroData
    retrieved_cases: List[dict]
    text_embeddings: Tensor   # 预计算的PathoGraph文本embeddings
    
    # 阶段1输出
    differential_diagnosis: List[str]
    target_phenotypes: List[str]   # 需要进一步查询的表型ID列表
    stage1_reasoning: str
    
    # 阶段2输出
    narrowed_diagnosis: List[str]
    quantitative_features: dict   # {表型ID: 量化描述}
    stage2_reasoning: str
    
    # 阶段3输出
    final_diagnosis: str
    report: str
    reasoning_chain: str   # 完整推理链，引用超边ID
    
    # 验证器状态
    verification_log: List[dict]   # 每阶段的验证结果
    regeneration_count: int
```

**LangGraph 节点定义**：
```python
def stage1_preliminary(state: DiagnosisState) -> DiagnosisState:
    """
    输入：组织节点全局特征（CONCH空间） + RAG检索的top-3历史病例CoT
    Prompt结构：
      - System：PathoGraph诊断流程约束
      - Context：top-k历史病例 + 其CoT轨迹
      - Query：当前样本的组织特征描述
    输出：鉴别诊断列表 + 需进一步查询的表型ID
    """
    pass

def verifier(state: DiagnosisState, stage: int) -> DiagnosisState:
    """
    图拓扑硬约束验证器（每阶段输出后执行）。
    
    算法：
    1. 提取当前阶段输出中的形态学描述短语（正则 + LLM提取）
    2. 短语 → CONCH text encoder → query embedding q
    3. 在超图中检索：sim_k = cosine(q, proj(h_e_k)) for k in K
    4. max_sim = max(sim_k)
    5. 若 max_sim < τ（默认0.3）：
       → 判定幻觉（报告描述的表型在当前WSI的超图中找不到视觉证据）
       → 将 "图中未找到 X 表型的视觉证据（最近邻超边为 Y，相似度{max_sim:.2f}）" 
         加入反馈，标记需要重生成
    6. 若 max_sim ≥ τ：通过，记录 "描述 X 对应超边 e_k"（可溯源）
    
    验证的数学基础：
    - C2 保证了 proj(h_e) 和 CONCH_text(phenotype) 在同一空间
    - 因此文本描述可以直接检索超边，无需额外转换
    """
    pass

def stage2_further(state: DiagnosisState) -> DiagnosisState:
    # 根据 stage1 指定的 target_phenotypes，查询对应超边的量化特征
    pass

def stage3_final(state: DiagnosisState) -> DiagnosisState:
    # 综合 stage1+2，生成最终分类 + 结构化报告 + 完整推理链
    pass

# LangGraph 图构建
def build_diagnosis_graph():
    workflow = StateGraph(DiagnosisState)
    workflow.add_node("stage1", stage1_preliminary)
    workflow.add_node("verify1", lambda s: verifier(s, stage=1))
    workflow.add_node("stage2", stage2_further)
    workflow.add_node("verify2", lambda s: verifier(s, stage=2))
    workflow.add_node("stage3", stage3_final)
    workflow.add_node("verify3", lambda s: verifier(s, stage=3))
    
    # 条件边：验证失败 → 重生成（最多3次）
    workflow.add_conditional_edges("verify1", route_after_verify, 
                                   {"pass": "stage2", "fail": "stage1"})
    # ...
    return workflow.compile()
```

---

## 6. 数据加载（dataloader.py）

```python
class BRACSDataset(Dataset):
    """
    统一接口，支持 Phase1（无表型文本）和 Phase2（含表型文本embedding）。
    """
    def __init__(self, graph_dir, phenotype_emb_path=None, split='train'):
        # graph_dir: 包含 .pt 文件的目录
        # phenotype_emb_path: 预计算的 PathoGraph 文本 embedding（Phase2用）
        pass

    def __getitem__(self, idx) -> Tuple[HeteroData, Tensor, int]:
        # 返回：(HeteroData, text_embeddings, label)
        # text_embeddings：所有样本共用，从文件加载一次即可（不需要per-sample存储）
        pass

def collate_fn(batch):
    # PyG: Batch.from_data_list([item[0] for item in batch])
    # labels: torch.stack([item[2] for item in batch])
    # text_embeddings: batch[0][1]（所有样本共用同一组表型文本embedding）
    pass
```

---

## 7. 训练脚本（train.py）

```python
def train(args, config):
    # 1. 加载 PathoGraph 文本 embedding（预计算，Phase2才需要）
    # 2. 创建 dataloader（train/val/test）
    # 3. 初始化模型（HACTBaseline 或 PathoHGA，由 config 决定）
    # 4. 初始化优化器（AdamW，不同子模块不同学习率）
    #    - backbone：lr=1e-4（较小，特征提取层）
    #    - hyperedge_gen：lr=5e-4（需要快速收敛到语义表型）
    #    - classifier：lr=5e-4
    # 5. 多GPU DDP（DistributedDataParallel，4× RTX 4090）
    # 6. 训练循环：
    #    - Gumbel温度退火（epoch 0→30：2.0→0.5）
    #    - 记录 L_cls, L_align 分别的值（用于监控）
    #    - val 上 early stopping（patience=10，监控 Weighted F1）
    #    - 保存 best model（val F1）
    # 7. 测试集评估：Accuracy, Weighted F1, AUC, Kappa
    pass
```

**多GPU策略**：
```bash
# 4张卡 DDP
torchrun --nproc_per_node=4 core/train.py --config configs/bracs_pathoHGA.yml
```

---

## 8. 关键设计决策汇总

| 决策点 | 选择 | 原因 |
|--------|------|------|
| 图框架 | PyG（非DGL） | HypergraphConv是PyG原生算子；与C1 cross-attention融合更自然 |
| 组织节点特征 | CONCH（备选PLIP） | CONCH已在病理VLP空间；组织节点特征与文本在同一空间，是C2的基础 |
| 细胞节点特征 | HoVer-Net形态 + ResNet18 | 形态特征对细胞类型区分最有效；ResNet18比34更快，诊断任务够用 |
| 超边数量K | 由PathoGraph固定（~20-30） | 每个超边对应一个生物学定义的表型；保证可解释性 |
| 超边初始化 | PathoGraph文本CONCH embedding | 确保超边从训练开始就有病理语义；而非随机初始化的纯数据驱动 |
| 对齐温度τ | 0.07（InfoNCE标准值） | K较小（20-30），标准温度即可 |
| Graph-RAG检索 | WL kernel + FAISS | WL kernel捕捉图拓扑结构相似性；FAISS保证检索效率 |
| LLM | Gemini 2.5 Pro | 超长上下文（支持完整WSI描述）；最新版医学推理能力强 |
| 验证器阈值τ | 0.3（初始值，需调参） | 在val集上搜索[0.1, 0.5]，找到幻觉召回率最优点 |

---

## 9. Phase 1 立即可执行的步骤

**前置条件**：
- [x] hact-net 代码已存在
- [x] 依赖库已通过 git clone 准备
- [ ] BRACS ROI 数据已下载（P0优先级）
- [ ] CONCH 权限已申请（或改用 PLIP）

**Step 1：环境配置**
```bash
conda env create -f environment.yml
conda activate pathoHGA
# 验证 PyG HypergraphConv 可用
python -c "from torch_geometric.nn import HypergraphConv; print('OK')"
```

**Step 2：复制 target.png（染色归一化用）**
```bash
cp /home/hanz/MediAgent/hact-net/data/target.png /home/hanz/MediAgent/PathoHGA/data/
```

**Step 3：整理 PathoGraph 表型 JSON**（人工，约2小时）
- 打开 PathoGraph 论文（doi:10.1038/s41597-025-04906-z）
- 找附录中的乳腺癌表型列表（phenotype definitions）
- 整理为 `data/pathograph_phenotypes.json`

格式：
```json
[
  {
    "id": "P001",
    "name": "nuclear_pleomorphism",
    "zh_name": "核多形性",
    "definition": "Marked variation in nuclear size and shape...",
    "entities": ["nucleus", "chromatin"],
    "relevant_subtypes": ["ADH", "DCIS", "IC"]
  },
  ...
]
```

**Step 4：写 graph_builder.py（Phase 1版本）**
- 逻辑复用 `generate_hact_graphs.py`
- 只改输出格式：DGL → PyG HeteroData

**Step 5：写 dataloader.py**

**Step 6：写 pathoHGA.py（HACTBaseline部分）**

**Step 7：验证 Phase 1**
```bash
python core/train.py --config configs/bracs_baseline.yml --cg_path /data/BRACS/graphs/cell_graphs ...
# 期望：val Weighted F1 ≥ 0.58
```

---

## 10. 风险点与应对

| 风险 | 应对 |
|------|------|
| CONCH 权限未获批 | 用 PLIP（完全开源，`vinid/plip`）；接口已统一，只改一行 |
| HoVer-Net 推理太慢 | 用 histocartography 的 NucleiExtractor（底层也是深度检测，但有缓存） |
| Gumbel-Softmax 训练不稳定 | 先用 softmax（soft assignment），收敛后再切到 Gumbel；监控 incidence matrix 的熵 |
| WL kernel 对大图慢 | 离散化时适当减少 k-means 中心数；对超大图做子图采样 |
| Gemini API 调用成本 | Case bank 构建时批量生成 CoT（训练集约4000样本），一次性生成，不重复调用 |
| Phase 1 F1 不达标 | 检查 PyG PNA 的 avg_d 参数是否与 DGL 版一致（DGL 和 PyG 的实现细节不同） |

---

*本文件是实现计划，代码写之前先与导师确认整体路线。*
