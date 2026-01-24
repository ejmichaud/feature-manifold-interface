"""
Export UMAP positions to JSON for the web visualizer.

Usage:
    python export_umap_json.py --positions-file ../data/graph/positions.npy --output ../visualizer/data/positions.json
"""

import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Export UMAP positions to JSON")
    parser.add_argument("--positions-file", type=str, required=True,
                        help="Path to positions.npy file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file path")

    args = parser.parse_args()

    print(f"Loading positions from {args.positions_file}...")
    positions = np.load(args.positions_file)
    print(f"  Shape: {positions.shape}")

    # Convert to list of [x, y] pairs
    positions_list = positions.tolist()

    print(f"Writing to {args.output}...")
    with open(args.output, "w") as f:
        json.dump({
            "n_latents": len(positions_list),
            "positions": positions_list
        }, f)

    print(f"Done. Exported {len(positions_list)} positions.")


if __name__ == "__main__":
    main()
