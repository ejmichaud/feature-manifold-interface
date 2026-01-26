#!/usr/bin/env python3
"""
Export sparse edge matrix to JSON for frontend loading.

Usage:
    python export_edges_json.py \
        --edges-file data/graph/decoder_similarity.npz \
        --output data/visualizer/edges.json \
        --max-edges-per-node 10

This script converts the sparse edge matrix from compute_edges.py
into a JSON format that the frontend can load directly.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def export_edges_to_json(
    edges_file: Path,
    output_file: Path,
    max_edges_per_node: int = 10,
    threshold: float = 0.0,
):
    """Export sparse edge matrix to JSON.

    Args:
        edges_file: Path to .npz sparse matrix file
        output_file: Output JSON file path
        max_edges_per_node: Maximum edges to keep per node (keeps highest weight)
        threshold: Minimum edge weight to include
    """
    print(f"Loading edges from {edges_file}...")
    edges_matrix = sparse.load_npz(edges_file)

    # Convert to COO for easy iteration
    coo = edges_matrix.tocoo()

    print(f"  Matrix shape: {edges_matrix.shape}")
    print(f"  Total non-zero entries: {edges_matrix.nnz}")

    # Filter by threshold
    mask = coo.data >= threshold
    rows = coo.row[mask]
    cols = coo.col[mask]
    weights = coo.data[mask]

    print(f"  After threshold {threshold}: {len(weights)} edges")

    # Build edge list (upper triangle only to avoid duplicates)
    edges_by_node = {}

    for i, j, w in zip(rows, cols, weights):
        if i >= j:  # Only upper triangle
            continue

        # Track edges for both endpoints
        for node in [i, j]:
            if node not in edges_by_node:
                edges_by_node[node] = []
            edges_by_node[node].append((i, j, w))

    # Keep only top-k edges per node
    kept_edges = set()

    for node, node_edges in edges_by_node.items():
        # Sort by weight descending
        node_edges.sort(key=lambda x: -x[2])

        # Keep top-k
        for i, j, w in node_edges[:max_edges_per_node]:
            kept_edges.add((i, j, w))

    # Convert to list format
    edge_list = [
        {"source": int(i), "target": int(j), "weight": float(w)}
        for i, j, w in sorted(kept_edges)
    ]

    print(f"  After max {max_edges_per_node} per node: {len(edge_list)} edges")

    # Save to JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "edges": edge_list,
            "n_edges": len(edge_list),
            "max_edges_per_node": max_edges_per_node,
            "threshold": threshold,
            "source_file": str(edges_file.name),
        }, f)

    print(f"Saved {len(edge_list)} edges to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Export edges to JSON")
    parser.add_argument("--edges-file", type=Path, required=True,
                        help="Input sparse matrix (.npz)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON file")
    parser.add_argument("--max-edges-per-node", type=int, default=10,
                        help="Maximum edges per node (default: 10)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Minimum edge weight (default: 0)")

    args = parser.parse_args()

    export_edges_to_json(
        edges_file=args.edges_file,
        output_file=args.output,
        max_edges_per_node=args.max_edges_per_node,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
