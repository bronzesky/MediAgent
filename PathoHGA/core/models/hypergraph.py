import math
import torch
import torch.nn as nn


class LearnableHypergraph(nn.Module):
    def __init__(self, hidden_dim: int, num_hyperedges: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_hyperedges = num_hyperedges
        self.prototypes = nn.Parameter(torch.randn(num_hyperedges, hidden_dim) * 0.02)

    def forward(self, cell_embeddings: torch.Tensor):
        scores = cell_embeddings @ self.prototypes.t() / math.sqrt(self.hidden_dim)
        incidence = scores.softmax(dim=1)
        hyper = incidence.t() @ cell_embeddings
        denom = incidence.sum(dim=0).unsqueeze(1).clamp_min(1e-6)
        hyper = hyper / denom
        hyper_pool = hyper.mean(dim=0)
        return hyper_pool, incidence
