from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, root: str, split: str):
        self.root = Path(root)
        self.split = split
        self.files = sorted((self.root / split).rglob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = torch.load(path, map_location="cpu", weights_only=False)
        data.graph_path = str(path)
        return data


def limit_files(files: List[Path], limit: int):
    return files[:limit] if limit and limit > 0 else files
