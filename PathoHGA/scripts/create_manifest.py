#!/usr/bin/env python3
"""
Create BRACS manifest with smoke subset covering both train and test.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

BASE_DIR = Path('/media/share/HDD_16T_1/AIFFPE/MediAgent')
BRACS_DIR = BASE_DIR / 'data' / 'BRACS'
OUTPUT_DIR = BRACS_DIR

LABEL_MAP = {
    'N': 0,
    'PB': 1,
    'UDH': 2,
    'FEA': 3,
    'ADH': 4,
    'DCIS': 5,
    'IC': 6,
}


def scan_split(split_name):
    split_dir = BRACS_DIR / split_name
    if not split_dir.exists():
        return []

    samples = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue
        parts = class_dir.name.split('_')
        if len(parts) < 2:
            continue
        label_str = parts[1]
        if label_str not in LABEL_MAP:
            continue
        label = LABEL_MAP[label_str]

        for img_path in sorted(class_dir.glob('*.png')):
            samples.append(
                {
                    'path': str(img_path.relative_to(BRACS_DIR)),
                    'image_id': img_path.stem,
                    'label': label,
                    'label_str': label_str,
                    'split': split_name,
                }
            )
    return samples


def _sample_by_class(samples, n_per_class):
    by_label = defaultdict(list)
    for sample in samples:
        by_label[sample['label']].append(sample)

    out = []
    for label in sorted(by_label.keys()):
        pool = by_label[label]
        n = min(n_per_class, len(pool))
        out.extend(random.sample(pool, n))
    return out


def create_smoke_subset(train_samples, test_samples, n_train_per_class=2, n_test_per_class=1):
    smoke = []
    smoke.extend(_sample_by_class(train_samples, n_train_per_class))
    smoke.extend(_sample_by_class(test_samples, n_test_per_class))
    return smoke


def main():
    train_samples = scan_split('train')
    val_samples = scan_split('val')
    test_samples = scan_split('test')

    smoke_samples = create_smoke_subset(train_samples, test_samples)

    manifest = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples,
        'smoke': smoke_samples,
        'label_map': LABEL_MAP,
        'stats': {
            'total': len(train_samples) + len(val_samples) + len(test_samples),
            'train': len(train_samples),
            'val': len(val_samples),
            'test': len(test_samples),
            'smoke': len(smoke_samples),
        },
    }

    output_path = OUTPUT_DIR / 'manifest.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    smoke_list_path = OUTPUT_DIR / 'smoke_list.txt'
    with open(smoke_list_path, 'w', encoding='utf-8') as f:
        for row in smoke_samples:
            f.write(f"{row['path']}\n")

    print(f"Manifest saved: {output_path}")
    print(f"Smoke list saved: {smoke_list_path}")
    print(f"Smoke samples: {len(smoke_samples)}")


if __name__ == '__main__':
    main()
