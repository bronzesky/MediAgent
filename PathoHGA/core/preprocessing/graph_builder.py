#!/usr/bin/env python3
import sys
import json
from pathlib import Path
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

sys.path.insert(0, '/media/share/HDD_16T_1/AIFFPE/MediAgent/hact-net')
from core.generate_hact_graphs import HACTBuilding, TUMOR_TYPE_TO_LABEL

def dgl_to_pyg(cell_graph, tissue_graph, assignment_matrix, label, split, image_id):
    data = HeteroData()
    
    cell_x = torch.from_numpy(cell_graph.ndata["feat"]).float()
    cell_pos = torch.from_numpy(cell_graph.ndata["centroid"]).float()
    data["cell"].x = cell_x
    data["cell"].pos = cell_pos
    
    tissue_x = torch.from_numpy(tissue_graph.ndata["feat"]).float()
    tissue_pos = torch.from_numpy(tissue_graph.ndata["centroid"]).float()
    data["tissue"].x = tissue_x
    data["tissue"].pos = tissue_pos
    
    cell_edges = cell_graph.edges()
    cell_edge_index = torch.stack([
        torch.from_numpy(cell_edges[0]),
        torch.from_numpy(cell_edges[1])
    ], dim=0).long()
    data["cell", "neighbors", "cell"].edge_index = cell_edge_index
    
    tissue_edges = tissue_graph.edges()
    tissue_edge_index = torch.stack([
        torch.from_numpy(tissue_edges[0]),
        torch.from_numpy(tissue_edges[1])
    ], dim=0).long()
    data["tissue", "adjacent", "tissue"].edge_index = tissue_edge_index
    
    assign_edges = assignment_matrix.nonzero()
    assign_edge_index = torch.from_numpy(assign_edges).long().t().contiguous()
    data["cell", "belongs_to", "tissue"].edge_index = assign_edge_index
    
    data.y = torch.tensor([label], dtype=torch.long)
    data.split = split
    data.image_id = image_id
    data.num_cells = int(cell_x.shape[0])
    data.num_tissue = int(tissue_x.shape[0])
    
    return data

def process_dataset(manifest_path, output_dir, target_image_path, subset='smoke', device='cuda'):
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    samples = manifest[subset]
    print(f'Processing {len(samples)} samples from {subset}')
    
    output_path = Path(output_dir) / subset
    output_path.mkdir(parents=True, exist_ok=True)
    
    builder = HACTBuilding()
    success = 0
    fail = 0
    
    base_dir = Path(manifest_path).parent
    
    for sample in tqdm(samples):
        image_path = base_dir / sample['path']
        if not image_path.exists():
            print(f'Missing: {image_path}')
            fail += 1
            continue
        
        try:
            cg, tg, am = builder.process(str(image_path))
            if cg is None or tg is None or am is None:
                fail += 1
                continue
            
            data = dgl_to_pyg(cg, tg, am, sample['label'], sample['split'], sample['image_id'])
            save_path = output_path / f"{sample['image_id']}.pt"
            torch.save(data, save_path)
            success += 1
        except Exception as e:
            print(f'Error: {e}')
            fail += 1
    
    print(f'Success: {success}, Failed: {fail}')
    return success, fail

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--subset', default='smoke')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    process_dataset(args.manifest, args.output, args.target, args.subset, args.device)
