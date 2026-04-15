# Phase 1 I/O Schema (Frozen)

> Version: 1.0  
> Date: 2026-04-15  
> Status: FROZEN - Do not modify without team sync

## HeteroData Structure

All graph files must follow this PyTorch Geometric HeteroData schema:

### Node Types

#### 1. cell nodes
- x: [N_cells, D_feat] - Node features (ResNet34: 512 + 2 coords = 514)
- pos: [N_cells, 2] - Absolute pixel coordinates (x, y)

#### 2. tissue nodes  
- x: [N_tissue, D_feat] - Tissue region features (ResNet34: 512)
- pos: [N_tissue, 2] - Region centroids

### Edge Types

#### 1. (cell, neighbors, cell)
- kNN edges (k=5, threshold=50 pixels)
- edge_index: [2, E_cell]

#### 2. (tissue, adjacent, tissue)
- RAG edges between adjacent tissue regions
- edge_index: [2, E_tissue]

#### 3. (cell, belongs_to, tissue)
- Assignment matrix (cell to tissue region)
- edge_index: [2, E_assign]

### Graph-Level Attributes

data.y = torch.tensor([label], dtype=torch.long)  # 0-6 for BRACS 7-class
data.split = train | val | test
data.image_id = str  # e.g., BRACS_123
data.num_cells = int
data.num_tissue = int

## Label Mapping (BRACS)

LABEL_MAP = {
    N: 0,      # Normal
    PB: 1,     # Pathological Benign
    UDH: 2,    # Usual Ductal Hyperplasia
    FEA: 3,    # Flat Epithelial Atypia
    ADH: 4,    # Atypical Ductal Hyperplasia
    DCIS: 5,   # Ductal Carcinoma In Situ
    IC: 6      # Invasive Carcinoma
}

## File Naming Convention

{split}/graphs/{image_id}.pt

Example:
data/BRACS/graphs/train/BRACS_0001_N.pt
data/BRACS/graphs/test/BRACS_0234_IC.pt

## Validation Checklist

Before committing any graph builder changes, verify:

- .pt file loads with torch.load()
- isinstance(data, torch_geometric.data.HeteroData)
- All required node types present: cell, tissue
- All required edge types present
- data.y is torch.long scalar tensor
- data.split is string
- No NaN/Inf in features
- edge_index is torch.long with shape [2, E]
