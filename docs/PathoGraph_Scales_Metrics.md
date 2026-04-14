# PathoGraph 尺度与指标整理（面向 PathoHGA）

> 目标：回答“先做哪些尺度、哪些指标”，并直接服务 C1/C2/C3 实现。
> 依据：`PathoML/schema.md`、`Representation_Examples/*.xml`、HER2 use case。

---

## 1. 推荐的尺度划分（先做 4 层）

1. 细胞/核尺度（Cell/Nucleus）
2. 组织区域尺度（Tissue/Region）
3. 表型超边尺度（Phenotype/Hyperedge）
4. 诊断流程尺度（Diagnosis Process）

> 当前阶段不建议先上 WSI 全局复杂指标，先把这 4 层打通。

---

## 2. 每个尺度的指标清单

## 2.1 细胞/核尺度（P0 必须）

### 几何与形态（来自 PathoML 现有可计算字段）
- `area`
- `roundness`
- `shape_factor`
- `bbox`
- `centroid`
- `contour`

### 类型与归属
- `cell_type`
- `tissue_id`（核归属到组织）
- `nucleus_id`

### 建议派生统计（用于 C1）
- 每组织内核密度（count / area）
- 核面积分布统计（mean/std/quantile）
- 核形态异质性（roundness/shape_factor 的方差）

---

## 2.2 组织区域尺度（P0 必须）

### 区域基础属性
- `tissue_id`
- `tissue_type`（如 tumor/stroma）
- `bbox`
- `contour`
- `nucleus_ids`

### 建议派生统计（用于 C1 + Phase1解释）
- tumor/stroma 面积占比
- 组织边界复杂度（perimeter-area 比）
- 区域内核数量与密度

---

## 2.3 表型超边尺度（P0 必须，P1 增强）

### 本体可直接承载的语义
- `Histopathological_Phenotype`
- `Cytopathological_Phenotype`
- `Immunophenotype`
- `hasSupportEvidence` / `has_OpposeEvidence`（或同义属性）

### 对应到你的超边定义（C1/C2）
- 超边 = 一个 phenotype 实例
- 超边成员 = 多个 cell/nucleus/tissue 实体
- 超边特征 = 成员节点聚合 + 组织上下文

### 建议定量指标
- `hasValue`（例如比例/百分比）
- 表型覆盖率（该表型超边覆盖的 tumor 区域比例）
- 表型强度分数（成员证据置信度聚合）

---

## 2.4 诊断流程尺度（P1 必须，做 C3）

### 流程结构字段（PathoML 示例中已出现）
- `DiagnosisProcess`
- `DiagnosisStage`
- `diagnosisOrder`
- `nextStep`
- `Diagnosis` / `Final_Diagnosis` / `Differential_Diagnosis`

### 证据字段（C3 验证器核心）
- `has_Morphologic_Evidence`
- `has_Quantitative_Metric_Evidence`
- `hasSupportEvidence`
- `hasContradictEvidence`（示例里有该命名）

### 建议评估指标（Agent）
- 每阶段证据覆盖率
- 支持/反证一致性比
- 幻觉率（无法映射到超边证据的陈述比例）

---

## 3. 与 PathoHGA 三个贡献的映射

- C1（可学习超边）：重点依赖 `细胞/核 + 组织 + 表型超边` 三层。
- C2（超边-语言对齐）：重点依赖 `表型超边` 的标准文本定义和 `hasValue` 类定量描述。
- C3（可验证推理）：重点依赖 `诊断流程` 与 `支持/反证证据` 关系。

---

## 4. 你现在应优先整理的最小字段集（建议直接落 JSON）

每个 phenotype 建议字段：

```json
{
  "id": 0,
  "name": "...",
  "level": "cyto|histo|immuno",
  "text": "...",
  "positive_evidence": ["..."],
  "negative_evidence": ["..."],
  "quantitative_metrics": [
    {"name": "...", "unit": "...", "operator": ">=", "threshold": 0.4}
  ],
  "source": "PathoML/PathoGraph",
  "organ": "breast"
}
```

> 先把 BRACS 高频相关的 20-40 个 phenotype 做出来就足够启动 C1/C2。

---

## 5. P0 / P1 执行优先级

### P0（本周必须）
- 细胞/核尺度：`area, roundness, shape_factor, cell_type, tissue_id`
- 组织尺度：`tissue_type, nucleus_ids, area占比`
- 表型超边：`name, text, level, positive_evidence`

### P1（下一阶段）
- 反证字段：`negative_evidence`
- 定量字段：`hasValue/threshold/unit`
- 诊断流程字段：`DiagnosisStage/nextStep`

---

## 6. 备注（来自当前仓库检查）

- 已有本体文件：`PathoML/PathoML.owl`
- `schema.md` 已给出 nucleus/tissue 可计算特征（可直接工程化）
- `Representation_Examples` 已给出诊断流程与证据关系模板（可直接映射 C3）
- 当前未见可直接用于训练的 `pathograph_phenotypes.json`，需要你整理生成

