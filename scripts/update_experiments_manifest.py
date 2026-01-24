#!/usr/bin/env python3
"""
Update the experiments manifest file.

Scans the experiments directory and creates/updates experiments.json
with metadata about each available experiment.

Usage:
    python update_experiments_manifest.py --data-root /path/to/data

The manifest is saved to {data_root}/experiments.json and contains:
{
    "experiments": [
        {
            "id": "gemma-3-27b-pt_layer31_65k_medium",
            "model": "gemma-3-27b-pt",
            "layer": 31,
            "sae_width": "65k",
            "sae_l0": "medium",
            "has_visualizer": true,
            "num_tokens": 10000000,
            "n_latents": 65536
        },
        ...
    ],
    "updated_at": "2024-01-24T..."
}
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


def parse_experiment_id(exp_id: str) -> dict:
    """Parse experiment ID into components."""
    # Pattern: {model}_layer{N}_{width}_{l0}
    # Example: gemma-3-27b-pt_layer31_65k_medium
    pattern = r"(.+)_layer(\d+)_(\d+k|\d+m)_(\w+)"
    match = re.match(pattern, exp_id)

    if match:
        return {
            "model": match.group(1),
            "layer": int(match.group(2)),
            "sae_width": match.group(3),
            "sae_l0": match.group(4),
        }
    return {}


def get_experiment_info(exp_dir: Path) -> dict | None:
    """Get metadata for an experiment directory."""
    exp_id = exp_dir.name

    # Check if visualizer data exists
    visualizer_dir = exp_dir / "visualizer"
    has_visualizer = (visualizer_dir / "index.json").exists()

    # Check if any data exists at all
    has_any_data = (
        (exp_dir / "metadata.json").exists() or
        (exp_dir / "raw_activations" / "metadata.json").exists()
    )

    if not has_any_data:
        return None

    # Parse experiment ID
    info = {
        "id": exp_id,
        "has_visualizer": has_visualizer,
        **parse_experiment_id(exp_id),
    }

    # Try to load metadata for more details
    metadata_path = exp_dir / "metadata.json"
    if not metadata_path.exists():
        metadata_path = exp_dir / "raw_activations" / "metadata.json"

    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            info["num_tokens"] = metadata.get("num_tokens")
            info["n_latents"] = metadata.get("n_latents")
            info["d_model"] = metadata.get("d_model")
        except Exception:
            pass

    return info


def update_manifest(data_root: Path) -> dict:
    """Scan experiments and update manifest."""
    experiments_dir = data_root / "experiments"

    if not experiments_dir.exists():
        return {"experiments": [], "updated_at": datetime.now().isoformat()}

    experiments = []

    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_dir.name.startswith("."):
            continue

        info = get_experiment_info(exp_dir)
        if info:
            experiments.append(info)

    # Sort by model, then layer, then width
    experiments.sort(key=lambda x: (
        x.get("model", ""),
        x.get("layer", 0),
        x.get("sae_width", ""),
        x.get("sae_l0", ""),
    ))

    manifest = {
        "experiments": experiments,
        "updated_at": datetime.now().isoformat(),
    }

    # Save manifest
    manifest_path = data_root / "experiments.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Updated {manifest_path}")
    print(f"Found {len(experiments)} experiments:")
    for exp in experiments:
        viz_status = "✓" if exp.get("has_visualizer") else "○"
        print(f"  [{viz_status}] {exp['id']}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Update experiments manifest")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root data directory")

    args = parser.parse_args()
    update_manifest(args.data_root)


if __name__ == "__main__":
    main()
