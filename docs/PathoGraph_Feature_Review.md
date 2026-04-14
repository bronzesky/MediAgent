# PathoGraph 特征 Review（是否合理、是否足够）

## 1. 结论

- 当前特征设计**合理**，可直接支撑 `Phase 1 -> C1 -> C2`。
- 对于“完整 C3 可验证推理 + 投稿级证据”，当前还需补充少量关键字段。

---

## 2. 当前方案为何合理

你当前的四层结构是对的：

1. 细胞/核尺度（形态基础）
2. 组织尺度（上下文）
3. 表型超边尺度（知识语义）
4. 诊断流程尺度（Agent阶段推理）

该结构和 PathoML 的对象关系（`Phenotype`、`DiagnosisProcess`、`has_Morphologic_Evidence` 等）一致，具备可解释链路基础。

---

## 3. 是否足够（按阶段）

### 对 C1/C2（当前开工）
- **足够**：
  - 细胞/核：`area`, `roundness`, `shape_factor`, `cell_type`, `tissue_id`
  - 组织：`tissue_type`, `nucleus_ids`, area-based 统计
  - 表型：`name`, `text`, `level`, `positive_evidence`

### 对 C3（验证器与结构化推理）
- **尚不完全足够**，建议补以下最小增量：
  - `negative_evidence`（反证）
  - `quantitative_metrics`（阈值化指标）
  - `operator/unit/threshold` 标准字段
  - `verifier_keywords`（文本-证据对齐触发词）

---

## 4. 必补字段（最小集合）

每个 phenotype 至少应有：

- `id`
- `name`
- `level` (`cyto|histo|immuno`)
- `text`
- `positive_evidence`（列表）
- `negative_evidence`（列表，可空）
- `quantitative_metrics`（列表，可空）
- `organ`（固定 `breast`）
- `bracs_relevance`（关联 BRACS 类别）

---

## 5. 通过标准（你现在即可执行）

满足以下条件即可进入 C1/C2 主训练：

1. `pathograph_phenotypes.json` 覆盖 20+ 个高频乳腺表型。
2. 每个表型都有 `text + positive_evidence`。
3. 至少 8 个关键表型有 `quantitative_metrics`。
4. 全部字段可被 dataloader 稳定读取（JSON schema 一致）。

---

## 6. 本次已落地

- 已生成可直接用的初版：
  - `data/pathograph/pathograph_phenotypes.json`
- 该版本优先覆盖 BRACS 高频相关表型，可直接用于 C1/C2。

