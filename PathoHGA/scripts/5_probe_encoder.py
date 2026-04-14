#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure PathoHGA package is importable when script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.preprocessing.encoder_loader import load_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe encoder loading for conch/conchv1_5/titan")
    parser.add_argument(
        "--encoder",
        required=True,
        choices=["conch", "conchv1_5", "titan"],
        help="Encoder name to load",
    )
    parser.add_argument(
        "--models-root",
        default=None,
        help="Path to models root (default: <MediAgent>/models)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. cpu or cuda:0",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional HF token, only used for conch if needed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model only and print metadata",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = load_encoder(
        encoder=args.encoder,
        models_root=args.models_root,
        device=args.device,
        hf_token=args.hf_token,
    )

    print("[OK] encoder loaded")
    print(f"  name            : {bundle.name}")
    print(f"  device          : {bundle.device}")
    print(f"  supports_text   : {bundle.supports_text}")
    print(f"  supports_slide  : {bundle.supports_slide}")
    print(f"  preprocess      : {type(bundle.preprocess).__name__ if bundle.preprocess else 'None'}")
    if bundle.extra:
        print(f"  extra keys      : {sorted(bundle.extra.keys())}")

    if not args.dry_run:
        print("[INFO] dry-run not set, but no forward pass was executed in this probe script.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
