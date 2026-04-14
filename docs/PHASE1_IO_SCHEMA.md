# Phase 1 I/O Schema (Frozen for Smoke Fullstack)

## Graph file format (`.pt`)
Each BRACS sample is saved as `torch_geometric.data.HeteroData` with fields:

- `data["cell"].x`: `FloatTensor [N_cell, 64]`
- `data["cell"].pos`: `FloatTensor [N_cell, 2]`
- `data["tissue"].x`: `FloatTensor [N_tissue, 64]`
- `data["tissue"].pos`: `FloatTensor [N_tissue, 2]`
- `data["cell", "knn", "cell"].edge_index`: `LongTensor [2, E_cell]`
- `data["tissue", "rag", "tissue"].edge_index`: `LongTensor [2, E_tissue]`
- `data["cell", "belongs", "tissue"].edge_index`: `LongTensor [2, E_belongs]`
- `data.y`: `LongTensor [1]` (class id)
- `data.sample_id`: `str`
- `data.image_path`: `str`

## Directory layout
- Input BRACS: `/media/share/HDD_16T_1/AIFFPE/MediAgent/data/BRACS/{train,test}/{class}/*.png`
- Smoke graphs: `PathoHGA/data/smoke_bracs/{train,test}/{class}/*.pt`
- Manifest: `PathoHGA/data/smoke_bracs/manifest.json`

## Label rule
- label is parsed from class directory prefix: `0_N -> 0`, `1_PB -> 1`, etc.
