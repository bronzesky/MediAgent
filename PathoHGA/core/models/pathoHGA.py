import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.hypergraph import LearnableHypergraph
from core.models.alignment import LabelTextAligner


class PathoHGAModel(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=128, num_classes=7, use_c1=False, use_c2=False, num_hyperedges=8):
        super().__init__()
        self.use_c1 = use_c1
        self.use_c2 = use_c2
        self.cell_encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.tissue_encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

        self.hyper = LearnableHypergraph(hidden_dim, num_hyperedges=num_hyperedges)
        graph_dim = hidden_dim * (3 if use_c1 else 2)
        self.classifier = nn.Sequential(nn.Linear(graph_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))
        self.aligner = LabelTextAligner(num_classes=num_classes, emb_dim=graph_dim)

    def forward(self, data):
        cell_h = self.cell_encoder(data["cell"].x)
        tissue_h = self.tissue_encoder(data["tissue"].x)
        cell_pool = cell_h.mean(dim=0)
        tissue_pool = tissue_h.mean(dim=0)

        features = [cell_pool, tissue_pool]
        aux = {}
        if self.use_c1:
            hyper_pool, incidence = self.hyper(cell_h)
            features.append(hyper_pool)
            aux["incidence"] = incidence

        graph_emb = torch.cat(features, dim=0).unsqueeze(0)
        logits = self.classifier(graph_emb)

        out = {"logits": logits, "graph_emb": graph_emb}
        if self.use_c2:
            loss_align = self.aligner(graph_emb, data.y.view(-1))
            out["loss_align"] = loss_align
        else:
            out["loss_align"] = torch.tensor(0.0, device=logits.device)
        out.update(aux)
        return out
