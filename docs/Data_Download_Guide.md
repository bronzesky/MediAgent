# MediAgent 数据集下载指南

> 适用项目：`/media/share/HDD_16T_1/AIFFPE/MediAgent`
> 目标：按 Phase 1 -> 2 -> 3 的顺序完成数据准备，避免返工。

---

## 1. 下载优先级

### P0（立刻需要）
- BRACS（主训练/验证数据，Demo 必需）
- PathoGraph 乳腺表型文本 JSON（手工整理）

### P1（Phase 2/3 前）
- TCGA-BRCA WSI（外部验证）
- TCGA-BRCA 临床标签（PAM50）
- CONCH 权重访问（或备选 PLIP）

### P2（Agent 评估）
- WSI-Bench（QA 数据）
- MultiPathQA（QA 数据）

---

## 2. 推荐目录结构

先创建目录：

```bash
mkdir -p /media/share/HDD_16T_1/AIFFPE/MediAgent/data/{BRACS,TCGA-BRCA,WSI-Bench,MultiPathQA,pathograph}
mkdir -p /media/share/HDD_16T_1/AIFFPE/MediAgent/data/TCGA-BRCA/{raw_wsi,clinical}
```

目标结构示例：

```text
data/
├── BRACS/
│   ├── train/
│   │   ├── 0_N/ 1_PB/ 2_UDH/ 3_FEA/ 4_ADH/ 5_DCIS/ 6_IC/
│   ├── val/
│   └── test/
├── pathograph/
│   └── pathograph_phenotypes.json
├── TCGA-BRCA/
│   ├── raw_wsi/
│   └── clinical/
├── WSI-Bench/
└── MultiPathQA/
```

---

## 3. BRACS 下载说明（P0）

### 3.1 下载内容
- 下载 BRACS 的 ROI 图像（`train/val/test`）
- 当前阶段不需要 WSI 原图（可后续补）

### 3.2 放置路径
- 放入：`/media/share/HDD_16T_1/AIFFPE/MediAgent/data/BRACS/`

### 3.3 校验项
- `train/val/test` 三个 split 都存在
- 每个 split 下 7 个类别目录都存在（`0_N ... 6_IC`）

---

## 4. PathoGraph 表型 JSON（P0）

### 4.1 文件路径
- ` /media/share/HDD_16T_1/AIFFPE/MediAgent/data/pathograph/pathograph_phenotypes.json`

### 4.2 建议格式

```json
[
  {
    "id": 0,
    "name": "normal_gland",
    "text": "Normal glandular structures with regular tubular arrangement..."
  }
]
```

### 4.3 要求
- 至少先覆盖 BRACS 高频相关表型（可迭代扩充）
- `id` 唯一、`name` 稳定、`text` 尽量规范医学术语

---

## 5. TCGA-BRCA 下载（P1）

> 来源：GDC Portal

### 5.1 准备
- 在 GDC Portal 中筛选 `TCGA-BRCA` + `Slide Image` 并导出 manifest
- 保存为：`gdc_manifest_TCGA-BRCA.txt`

### 5.2 下载命令

```bash
cd /media/share/HDD_16T_1/AIFFPE/MediAgent/data/TCGA-BRCA
wget https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip
unzip gdc-client_v1.6.1_Ubuntu_x64.zip
./gdc-client download -m gdc_manifest_TCGA-BRCA.txt -d ./raw_wsi
```

### 5.3 临床标签
- 下载 TCGA-BRCA clinical 数据
- 提取并保存 PAM50 相关字段到：`./clinical/`

---

## 6. WSI-Bench / MultiPathQA（P2）

```bash
cd /media/share/HDD_16T_1/AIFFPE/MediAgent/data
git clone https://github.com/cpystan/WSI-LLaVA WSI-Bench_src
git clone https://github.com/manrai/GIANT MultiPathQA_src
```

后续从两个仓库中提取 QA 文件并整理到：
- `data/WSI-Bench/`
- `data/MultiPathQA/`

---

## 7. 一次性快速校验命令

```bash
# 1) 检查目录
find /media/share/HDD_16T_1/AIFFPE/MediAgent/data -maxdepth 2 -type d | sort

# 2) 检查 BRACS 目录层级
find /media/share/HDD_16T_1/AIFFPE/MediAgent/data/BRACS -maxdepth 2 -type d | sort

# 3) 检查 PathoGraph JSON
python - <<'PY'
import json
p = '/media/share/HDD_16T_1/AIFFPE/MediAgent/data/pathograph/pathograph_phenotypes.json'
with open(p, 'r', encoding='utf-8') as f:
    data = json.load(f)
print('entries:', len(data))
print('first_keys:', list(data[0].keys()) if data else [])
PY
```

---

## 8. 常见问题

- **Q: 现在必须下载 TCGA 吗？**
  - A: 不是。先完成 BRACS + PathoGraph JSON，跑通 Phase 1 再下 TCGA。

- **Q: CONCH 权限没批下来怎么办？**
  - A: 先用 PLIP 或其他开源病理视觉编码器做替代，先验证流程可跑。

- **Q: 如何避免数据泄漏？**
  - A: 所有 split 和 RAG 检索库都按 patient-level 隔离，不允许 test 样本进入索引库。


---

## 9. BRACS 实测可用命令（服务器验证通过）

> 注意：官网页面里的命令有时会把 `--no-parent` 显示成中文破折号，需手动改成两个半角减号。

### 9.1 先验证 FTP 可连通

```bash
wget --spider ftp://histoimage.na.icar.cnr.it/
```

### 9.2 仅下载 BRACS ROI（推荐，当前阶段只需这个）

```bash
mkdir -p /media/share/HDD_16T_1/AIFFPE/MediAgent/data/BRACS
cd /media/share/HDD_16T_1/AIFFPE/MediAgent/data/BRACS

# 只拉 BRACS_RoI/latest_version，下到本地后直接得到 train/val/test
wget -m -np -nH --cut-dirs=2 -R "index.html*" \
  ftp://histoimage.na.icar.cnr.it/BRACS_RoI/latest_version/
```

### 9.3 下载 BRACS 汇总文件（可选）

```bash
cd /media/share/HDD_16T_1/AIFFPE/MediAgent/data/BRACS
wget -O BRACS.xlsx ftp://histoimage.na.icar.cnr.it/BRACS.xlsx
```

### 9.4 下载 WSI 与标注（后续阶段可选）

```bash
cd /media/share/HDD_16T_1/AIFFPE/MediAgent/data
wget -m -np -nH --cut-dirs=1 -R "index.html*" ftp://histoimage.na.icar.cnr.it/BRACS_WSI/
wget -m -np -nH --cut-dirs=1 -R "index.html*" ftp://histoimage.na.icar.cnr.it/BRACS_WSI_Annotations/
```

---

## 10. PathoGraph / PathoML 说明

- 你当前仓库里已有：`/media/share/HDD_16T_1/AIFFPE/MediAgent/PathoML/PathoML.owl`
- 目前未见现成 `pathograph_phenotypes.json`，建议从 PathoML/论文附录抽取并整理为：
  - `/media/share/HDD_16T_1/AIFFPE/MediAgent/data/pathograph/pathograph_phenotypes.json`

建议字段：

```json
{
  "id": 0,
  "name": "normal_gland",
  "text": "...",
  "source": "PathoGraph/PathoML",
  "organ": "breast"
}
```
