# 实体图构建方案（Cell/Tissue Entity Graph）

更新时间：2026-04-15

## 目标
从“采样点/区域代理特征”切换到“实体级节点特征”：
- 细胞节点 = 细胞实例（nucleus/cell instance）
- 组织节点 = 组织实例（tissue region instance）
- 特征来自实体掩膜区域本身，而不是固定 patch 采样

## 实体输入约定
给定每张图 `xxx.png`，在 annotation 根目录下提供同名 sidecar：
- `cells/xxx.cells.json`
- `tissues/xxx.tissues.json`

### cells JSON
```json
{
  "cells": [
    {"id": 1, "centroid": [x, y], "bbox": [x0, y0, x1, y1], "type": 2}
  ]
}
```

### tissues JSON
```json
{
  "tissues": [
    {"id": 10, "centroid": [x, y], "bbox": [x0, y0, x1, y1], "region_type": 1}
  ]
}
```

可选字段：`polygon`（点序列）可替代 `bbox`。

## 特征定义（实体级）

### 细胞实体特征
在 cell mask 内统计：
- RGB mean/std（6）
- 灰度 mean/std + P10/P50/P90（5）
- 梯度 mean/std + P50/P90（4）
- 面积、宽高比、紧致度（3）
- 细胞类型 one-hot（可选扩展）

### 组织实体特征
在 tissue mask 内统计：
- RGB mean/std（6）
- 灰度 mean/std + P10/P50/P90（5）
- 梯度 mean/std + P50/P90（4）
- 区域面积占比、宽高比（2）
- 区域类型 one-hot（可选扩展）

统一通过投影/截断映射到 `feature_dim`（当前默认 64）。

## 图结构
- Cell-Cell：kNN（`k_cell=5`）
- Tissue-Tissue：kNN（`k_tissue=3`）
- Cell->Tissue：优先实体包含关系（点在 bbox/polygon 内），否则退化到最近 tissue centroid

## 兼容策略
`graph_builder.py` 提供两种模式：
1. `entity`（默认）：必须读取实体注释 sidecar
2. `proxy`（仅过渡）：无注释时允许代理特征（需 `--allow_proxy_features`）

> 结论：后续正式训练/论文实验必须使用 `entity` 模式，`proxy` 只用于管线联调。

## 当前实现状态
- [x] 文档定义实体图输入/输出与特征规则
- [x] `graph_builder.py` 支持 entity 模式与 sidecar 读取
- [x] `graph_builder.py` 保留 proxy 过渡开关（显式开启）
- [ ] 接入真实 HoVer-Net/组织分割输出目录并跑首轮构图
- [ ] 下游训练切换为 strict entity-only
