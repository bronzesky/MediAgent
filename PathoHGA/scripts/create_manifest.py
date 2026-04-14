#!/usr/bin/env python3
"""
Create BRACS data manifest and smoke test subset.
"""
import os
import json
from pathlib import Path
from collections import defaultdict
import random

random.seed(42)

BASE_DIR = Path('/media/share/HDD_16T_1/AIFFPE/MediAgent')
BRACS_DIR = BASE_DIR / 'data' / 'BRACS'
OUTPUT_DIR = BASE_DIR / 'data' / 'BRACS'

LABEL_MAP = {
    'N': 0,
    'PB': 1,
    'UDH': 2,
    'FEA': 3,
    'ADH': 4,
    'DCIS': 5,
    'IC': 6
}

def scan_split(split_name):
    """Scan a split directory and collect image paths."""
    split_dir = BRACS_DIR / split_name
    if not split_dir.exists():
        return []
    
    samples = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue
        
        # Extract label from directory name (e.g., '0_N' -> 'N')
        parts = class_dir.name.split('_')
        if len(parts) < 2:
            continue
        label_str = parts[1]
        
        if label_str not in LABEL_MAP:
            print(f'Warning: Unknown label {label_str} in {class_dir}')
            continue
        
        label = LABEL_MAP[label_str]
        
        # Collect all PNG files
        for img_path in sorted(class_dir.glob('*.png')):
            samples.append({
                'path': str(img_path.relative_to(BRACS_DIR)),
                'image_id': img_path.stem,
                'label': label,
                'label_str': label_str,
                'split': split_name
            })
    
    return samples

def create_smoke_subset(samples, n_per_class=3):
    """Create a small smoke test subset."""
    by_label = defaultdict(list)
    for s in samples:
        by_label[s['label']].append(s)
    
    smoke = []
    for label in sorted(by_label.keys()):
        available = by_label[label]
        n_select = min(n_per_class, len(available))
        selected = random.sample(available, n_select)
        smoke.extend(selected)
    
    return smoke

def main():
    print('Scanning BRACS dataset...')
    
    # Scan all splits
    train_samples = scan_split('train')
    test_samples = scan_split('test')
    
    # Check if val exists
    val_samples = scan_split('val')
    if not val_samples:
        print('No val split found, will use train/test only')
    
    all_samples = train_samples + val_samples + test_samples
    
    # Statistics
    print(f'\nTotal samples: {len(all_samples)}')
    print(f'  Train: {len(train_samples)}')
    print(f'  Val: {len(val_samples)}')
    print(f'  Test: {len(test_samples)}')
    
    # Class distribution
    print('\nClass distribution (train):')
    train_by_label = defaultdict(int)
    for s in train_samples:
        train_by_label[s['label_str']] += 1
    for label_str in sorted(train_by_label.keys()):
        print(f'  {label_str}: {train_by_label[label_str]}')
    
    # Create smoke subset from train
    smoke_samples = create_smoke_subset(train_samples, n_per_class=3)
    print(f'\nSmoke test subset: {len(smoke_samples)} samples')
    
    # Save manifests
    manifest = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples,
        'smoke': smoke_samples,
        'label_map': LABEL_MAP,
        'stats': {
            'total': len(all_samples),
            'train': len(train_samples),
            'val': len(val_samples),
            'test': len(test_samples),
            'smoke': len(smoke_samples)
        }
    }
    
    output_path = OUTPUT_DIR / 'manifest.json'
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f'\nManifest saved to: {output_path}')
    
    # Save smoke list separately for easy access
    smoke_list_path = OUTPUT_DIR / 'smoke_list.txt'
    with open(smoke_list_path, 'w') as f:
        for s in smoke_samples:
            f.write(f"{s['path']}\n")
    
    print(f'Smoke list saved to: {smoke_list_path}')

if __name__ == '__main__':
    main()
