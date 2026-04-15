"""
C2: PathoGraph锚定的多粒度视觉-语言对齐模块

核心功能:
1. 超边表示 ↔ PathoGraph表型文本对齐 (InfoNCE)
2. 对齐后超边表示可用于C3的验证器
3. 区别于MLLM-HWSI: 锚点是结构化知识本体，非报告文本
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PathoGraphAligner(nn.Module):
    """
    PathoGraph知识本体锚定的视觉-语言对齐模块。
    
    对齐策略:
    - 超边聚合表示 → 投影到CONCH文本空间
    - InfoNCE loss: 超边 ↔ 对应PathoGraph表型文本
    
    与MLLM-HWSI的区别:
    - MLLM-HWSI: patch/region → 报告句子 (数据驱动)
    - 本工作: 超边 → PathoGraph定义 (知识驱动)
    """
    
    def __init__(
        self,
        hyperedge_dim: int,
        text_dim: int = 512,  # CONCH text encoder output dim
        projection_dim: int = 256,
        temperature: float = 0.07,
        dropout: float = 0.1,
    ):
        """
        Args:
            hyperedge_dim: 超边特征维度
            text_dim: 文本编码器输出维度 (CONCH: 512)
            projection_dim: 对齐空间维度
            temperature: InfoNCE温度参数
            dropout: Dropout率
        """
        super().__init__()
        self.hyperedge_dim = hyperedge_dim
        self.text_dim = text_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        
        # 超边表示投影到对齐空间
        self.hyperedge_proj = nn.Sequential(
            nn.Linear(hyperedge_dim, hyperedge_dim),
            nn.LayerNorm(hyperedge_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hyperedge_dim, projection_dim),
        )
        
        # 文本表示投影到对齐空间
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim, projection_dim),
        )
        
        # 可学习的温度参数 (可选)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / temperature)))
    
    def forward(
        self,
        hyperedge_features: Tensor,
        text_features: Tensor,
        return_similarity: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        计算InfoNCE对齐损失。
        
        Args:
            hyperedge_features: (B, K, d_h) 超边特征
                B: batch size
                K: 每个样本的超边数
                d_h: 超边特征维度
            text_features: (K, d_t) PathoGraph表型文本特征
                K: 表型数量 (与超边数对应)
                d_t: 文本特征维度
            return_similarity: 是否返回相似度矩阵
        
        Returns:
            loss: InfoNCE对齐损失
            similarity: (B*K, K) 相似度矩阵 (如果return_similarity=True)
        """
        B, K, d_h = hyperedge_features.shape
        
        # Flatten batch dimension: (B*K, d_h)
        hyperedge_flat = hyperedge_features.reshape(B * K, d_h)
        
        # 投影到对齐空间
        hyperedge_proj = self.hyperedge_proj(hyperedge_flat)  # (B*K, d_proj)
        text_proj = self.text_proj(text_features)  # (K, d_proj)
        
        # L2归一化
        hyperedge_proj = F.normalize(hyperedge_proj, dim=-1)
        text_proj = F.normalize(text_proj, dim=-1)
        
        # 计算相似度矩阵: (B*K, K)
        logit_scale = self.logit_scale.exp()
        similarity = logit_scale * hyperedge_proj @ text_proj.t()
        
        # InfoNCE loss
        # 正样本: 每个超边对应其表型文本 (对角线)
        # 负样本: 其他表型文本
        
        # 创建标签: 每个超边对应的表型索引
        # 假设超边k对应表型k (需要在数据加载时保证)
        labels = torch.arange(K, device=hyperedge_features.device).repeat(B)  # (B*K,)
        
        # Cross-entropy loss (等价于InfoNCE)
        loss = F.cross_entropy(similarity, labels)
        
        if return_similarity:
            return loss, similarity
        return loss, None
    
    def get_aligned_features(
        self,
        hyperedge_features: Tensor,
    ) -> Tensor:
        """
        获取对齐后的超边特征 (用于C3验证器)。
        
        Args:
            hyperedge_features: (B, K, d_h) 或 (K, d_h)
        
        Returns:
            aligned_features: (B, K, d_proj) 或 (K, d_proj)
        """
        original_shape = hyperedge_features.shape
        
        if hyperedge_features.dim() == 3:
            B, K, d_h = hyperedge_features.shape
            hyperedge_flat = hyperedge_features.reshape(B * K, d_h)
        else:
            K, d_h = hyperedge_features.shape
            hyperedge_flat = hyperedge_features
        
        # 投影并归一化
        aligned = self.hyperedge_proj(hyperedge_flat)
        aligned = F.normalize(aligned, dim=-1)
        
        if hyperedge_features.dim() == 3:
            aligned = aligned.reshape(B, K, -1)
        
        return aligned
    
    def compute_text_similarity(
        self,
        hyperedge_features: Tensor,
        text_features: Tensor,
    ) -> Tensor:
        """
        计算超边与文本的相似度 (用于C3验证器)。
        
        Args:
            hyperedge_features: (K, d_h) 超边特征
            text_features: (M, d_t) 文本特征 (M个候选文本)
        
        Returns:
            similarity: (K, M) 相似度矩阵
        """
        # 投影并归一化
        hyperedge_proj = self.hyperedge_proj(hyperedge_features)
        hyperedge_proj = F.normalize(hyperedge_proj, dim=-1)
        
        text_proj = self.text_proj(text_features)
        text_proj = F.normalize(text_proj, dim=-1)
        
        # 计算相似度
        logit_scale = self.logit_scale.exp()
        similarity = logit_scale * hyperedge_proj @ text_proj.t()
        
        return similarity


class MultiGranularityAligner(nn.Module):
    """
    多粒度对齐模块 (可选扩展)。
    
    对齐三个粒度:
    1. 细胞节点 ↔ 细胞类型文本
    2. 组织节点 ↔ 组织结构文本
    3. 超边 ↔ PathoGraph表型文本 (主要)
    """
    
    def __init__(
        self,
        cell_dim: int,
        tissue_dim: int,
        hyperedge_dim: int,
        text_dim: int = 512,
        projection_dim: int = 256,
        temperature: float = 0.07,
        dropout: float = 0.1,
        align_cell: bool = False,
        align_tissue: bool = False,
        align_hyperedge: bool = True,
    ):
        """
        Args:
            cell_dim: 细胞节点特征维度
            tissue_dim: 组织节点特征维度
            hyperedge_dim: 超边特征维度
            text_dim: 文本编码器输出维度
            projection_dim: 对齐空间维度
            temperature: InfoNCE温度参数
            dropout: Dropout率
            align_cell: 是否对齐细胞粒度
            align_tissue: 是否对齐组织粒度
            align_hyperedge: 是否对齐超边粒度
        """
        super().__init__()
        self.align_cell = align_cell
        self.align_tissue = align_tissue
        self.align_hyperedge = align_hyperedge
        
        # 细胞粒度对齐器
        if align_cell:
            self.cell_aligner = PathoGraphAligner(
                hyperedge_dim=cell_dim,
                text_dim=text_dim,
                projection_dim=projection_dim,
                temperature=temperature,
                dropout=dropout,
            )
        
        # 组织粒度对齐器
        if align_tissue:
            self.tissue_aligner = PathoGraphAligner(
                hyperedge_dim=tissue_dim,
                text_dim=text_dim,
                projection_dim=projection_dim,
                temperature=temperature,
                dropout=dropout,
            )
        
        # 超边粒度对齐器 (主要)
        if align_hyperedge:
            self.hyperedge_aligner = PathoGraphAligner(
                hyperedge_dim=hyperedge_dim,
                text_dim=text_dim,
                projection_dim=projection_dim,
                temperature=temperature,
                dropout=dropout,
            )
    
    def forward(
        self,
        cell_features: Optional[Tensor] = None,
        tissue_features: Optional[Tensor] = None,
        hyperedge_features: Optional[Tensor] = None,
        cell_text: Optional[Tensor] = None,
        tissue_text: Optional[Tensor] = None,
        hyperedge_text: Optional[Tensor] = None,
    ) -> dict:
        """
        多粒度对齐前向传播。
        
        Returns:
            losses: dict with keys 'cell_loss', 'tissue_loss', 'hyperedge_loss', 'total_loss'
        """
        losses = {}
        total_loss = 0.0
        
        if self.align_cell and cell_features is not None and cell_text is not None:
            cell_loss, _ = self.cell_aligner(cell_features, cell_text)
            losses['cell_loss'] = cell_loss
            total_loss += cell_loss
        
        if self.align_tissue and tissue_features is not None and tissue_text is not None:
            tissue_loss, _ = self.tissue_aligner(tissue_features, tissue_text)
            losses['tissue_loss'] = tissue_loss
            total_loss += tissue_loss
        
        if self.align_hyperedge and hyperedge_features is not None and hyperedge_text is not None:
            hyperedge_loss, _ = self.hyperedge_aligner(hyperedge_features, hyperedge_text)
            losses['hyperedge_loss'] = hyperedge_loss
            total_loss += hyperedge_loss
        
        losses['total_loss'] = total_loss
        return losses


# 辅助函数: 加载PathoGraph文本embedding
def load_pathograph_text_embeddings(
    phenotype_json_path: str,
    text_encoder,
    device: torch.device,
) -> Tensor:
    """
    加载PathoGraph表型文本并用CONCH编码。
    
    Args:
        phenotype_json_path: pathograph_phenotypes.json路径
        text_encoder: CONCH text encoder
        device: 设备
    
    Returns:
        text_embeddings: (K, d_text) 表型文本embedding
    """
    import json
    from pathlib import Path
    
    with open(phenotype_json_path, 'r', encoding='utf-8') as f:
        phenotypes = json.load(f)
    
    texts = [p['text'] for p in phenotypes]
    
    # 用CONCH编码
    with torch.no_grad():
        if hasattr(text_encoder, 'encode_text'):
            # CONCH接口
            text_embeddings = text_encoder.encode_text(texts)
        else:
            # 通用CLIP接口
            import clip
            text_tokens = clip.tokenize(texts).to(device)
            text_embeddings = text_encoder.encode_text(text_tokens)
    
    return text_embeddings.to(device)


class LabelTextAligner(nn.Module):
    """Compatibility adapter for smoke pipeline."""

    def __init__(self, num_classes: int, emb_dim: int):
        super().__init__()
        self.text_table = nn.Embedding(num_classes, emb_dim)
        nn.init.normal_(self.text_table.weight, std=0.02)

    def forward(self, graph_emb: Tensor, labels: Tensor) -> Tensor:
        text_emb = self.text_table(labels)
        graph_emb = F.normalize(graph_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)
        sim = (graph_emb * text_emb).sum(dim=-1)
        return (1.0 - sim).mean()
