import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelTextAligner(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int):
        super().__init__()
        self.text_table = nn.Embedding(num_classes, emb_dim)
        nn.init.normal_(self.text_table.weight, std=0.02)

    def forward(self, graph_emb: torch.Tensor, labels: torch.Tensor):
        text_emb = self.text_table(labels)
        graph_emb = F.normalize(graph_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)
        sim = (graph_emb * text_emb).sum(dim=-1)
        return (1.0 - sim).mean()
