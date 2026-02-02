#!/usr/bin/env python3
"""
Export graph data to formats compatible with Graphia.

Exports both cosine similarity and Jaccard similarity graphs as pairwise
text files (.txt) and optionally as GraphML (.graphml) for richer metadata.

Node labels include both the latent ID and the token it activates most on.

Usage:
    # Export from an experiment directory (both similarity types)
    python scripts/export_graphia.py \
        --data-dir /remote/ericjm/feature-manifold-interface/data/experiments/gemma-3-27b-pt_layer31_65k_medium \
        --output-dir exports/graphia

    # With edge filtering
    python scripts/export_graphia.py \
        --data-dir /remote/ericjm/feature-manifold-interface/data/experiments/gemma-3-27b-pt_layer31_65k_medium \
        --output-dir exports/graphia \
        --threshold 0.3 \
        --max-edges-per-node 20

    # Export only cosine similarity as GraphML
    python scripts/export_graphia.py \
        --data-dir /remote/ericjm/feature-manifold-interface/data/experiments/gemma-3-27b-pt_layer31_65k_medium \
        --output-dir exports/graphia \
        --types cosine \
        --format graphml

    # Export both formats
    python scripts/export_graphia.py \
        --data-dir /remote/ericjm/feature-manifold-interface/data/experiments/gemma-3-27b-pt_layer31_65k_medium \
        --output-dir exports/graphia \
        --format both
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy import sparse

# Regex matching characters forbidden in XML 1.0:
# U+0000–U+0008, U+000B, U+000C, U+000E–U+001F, U+007F–U+009F, U+FFFE, U+FFFF
_XML_ILLEGAL_RE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffe\uffff]"
)


def _sanitize_xml(text: str) -> str:
    """Remove characters that are illegal in XML 1.0."""
    return _XML_ILLEGAL_RE.sub("", text)


def load_labels(data_dir: Path, n_latents: int) -> dict[int, str]:
    """Load latent labels from latent_labels.json.

    Returns dict mapping latent_id -> "L{id}_{token}" label string.
    """
    labels_path = data_dir / "raw_activations" / "latent_labels.json"
    if not labels_path.exists():
        # Fallback: just use latent IDs
        print(f"  Warning: {labels_path} not found, using numeric IDs only")
        return {i: f"L{i}" for i in range(n_latents)}

    with open(labels_path) as f:
        raw = json.load(f)

    labels = {}
    for i in range(n_latents):
        entry = raw.get(str(i))
        if entry and entry.get("token") is not None:
            token = entry["token"].strip()
            # Strip XML-illegal characters and replace whitespace escapes
            token = _sanitize_xml(token)
            token = token.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
            # Truncate long tokens
            if len(token) > 30:
                token = token[:27] + "..."
            labels[i] = f"L{i}_{token}"
        else:
            labels[i] = f"L{i}"

    return labels


def load_edges(
    edges_path: Path,
    threshold: float = 0.0,
    max_edges_per_node: int | None = None,
) -> list[tuple[int, int, float]]:
    """Load sparse edge matrix and return filtered edge list.

    Returns list of (source, target, weight) tuples (upper triangle only).
    """
    print(f"  Loading {edges_path.name}...")
    matrix = sparse.load_npz(edges_path)
    coo = matrix.tocoo()

    print(f"    Matrix shape: {matrix.shape}, nnz: {matrix.nnz}")

    # Filter to upper triangle and threshold
    mask = (coo.row < coo.col) & (coo.data >= threshold)
    rows = coo.row[mask]
    cols = coo.col[mask]
    weights = coo.data[mask]

    print(f"    After threshold {threshold}: {len(weights):,} edges")

    if max_edges_per_node is not None:
        # Build per-node edge lists, keep top-k
        edges_by_node: dict[int, list[tuple[int, int, float]]] = {}
        for r, c, w in zip(rows, cols, weights):
            for node in [int(r), int(c)]:
                if node not in edges_by_node:
                    edges_by_node[node] = []
                edges_by_node[node].append((int(r), int(c), float(w)))

        kept = set()
        for node, node_edges in edges_by_node.items():
            node_edges.sort(key=lambda x: -x[2])
            for edge in node_edges[:max_edges_per_node]:
                kept.add(edge)

        edge_list = sorted(kept)
        print(f"    After max {max_edges_per_node}/node: {len(edge_list):,} edges")
    else:
        edge_list = sorted(
            (int(r), int(c), float(w)) for r, c, w in zip(rows, cols, weights)
        )

    return edge_list


def export_pairwise(
    edge_list: list[tuple[int, int, float]],
    labels: dict[int, str],
    output_path: Path,
    edge_type: str,
):
    """Export edges as Graphia pairwise text file.

    Format: tab-separated source, target, weight with a header row.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"source\ttarget\t{edge_type}_weight\n")
        for src, tgt, weight in edge_list:
            f.write(f"{labels[src]}\t{labels[tgt]}\t{weight:.6f}\n")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {len(edge_list):,} edges to {output_path} ({size_mb:.1f} MB)")


def export_graphml(
    edge_list: list[tuple[int, int, float]],
    labels: dict[int, str],
    n_latents: int,
    output_path: Path,
    edge_type: str,
    label_data: dict | None = None,
):
    """Export as GraphML with node attributes.

    Includes node attributes: latent_id, token_label, total_firings.
    Edge attributes: weight.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect nodes that appear in edges
    node_ids = set()
    for src, tgt, _ in edge_list:
        node_ids.add(src)
        node_ids.add(tgt)

    ns = "http://graphml.graphdrawing.org/xmlns"
    ET.register_namespace("", ns)

    graphml = ET.Element("graphml", xmlns=ns)

    # Define attribute keys
    ET.SubElement(graphml, "key", id="label", attrib={
        "for": "node", "attr.name": "label", "attr.type": "string",
    })
    ET.SubElement(graphml, "key", id="latent_id", attrib={
        "for": "node", "attr.name": "latent_id", "attr.type": "int",
    })
    ET.SubElement(graphml, "key", id="token", attrib={
        "for": "node", "attr.name": "token", "attr.type": "string",
    })
    ET.SubElement(graphml, "key", id="total_firings", attrib={
        "for": "node", "attr.name": "total_firings", "attr.type": "int",
    })
    ET.SubElement(graphml, "key", id="weight", attrib={
        "for": "edge", "attr.name": "weight", "attr.type": "double",
    })

    graph = ET.SubElement(graphml, "graph", id="G", edgedefault="undirected")

    # Add nodes
    for nid in sorted(node_ids):
        node_el = ET.SubElement(graph, "node", id=str(nid))
        d = ET.SubElement(node_el, "data", key="label")
        d.text = f"L{nid}"
        d = ET.SubElement(node_el, "data", key="latent_id")
        d.text = str(nid)

        entry = (label_data or {}).get(str(nid), {})
        d = ET.SubElement(node_el, "data", key="token")
        d.text = _sanitize_xml(entry.get("token", "") or "")
        d = ET.SubElement(node_el, "data", key="total_firings")
        d.text = str(entry.get("total_firings", 0))

    # Add edges
    for i, (src, tgt, weight) in enumerate(edge_list):
        edge_el = ET.SubElement(graph, "edge", id=f"e{i}", source=str(src), target=str(tgt))
        d = ET.SubElement(edge_el, "data", key="weight")
        d.text = f"{weight:.6f}"

    tree = ET.ElementTree(graphml)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode", xml_declaration=True)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {len(node_ids):,} nodes, {len(edge_list):,} edges to {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Export graph data for Graphia visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Experiment data directory (e.g., data/experiments/gemma-3-27b-pt_layer31_65k_medium)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: {data-dir}/graphia)",
    )
    parser.add_argument(
        "--types", nargs="+", default=["cosine", "jaccard"],
        choices=["cosine", "jaccard", "coactivation"],
        help="Edge types to export (default: cosine jaccard)",
    )
    parser.add_argument(
        "--format", choices=["pairwise", "graphml", "both"], default="pairwise",
        help="Output format (default: pairwise)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="Minimum edge weight to include (default: 0.0)",
    )
    parser.add_argument(
        "--max-edges-per-node", type=int, default=None,
        help="Maximum edges per node (default: no limit, uses all stored edges)",
    )

    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir or (data_dir / "graphia")
    graph_dir = data_dir / "graph"

    # Load metadata
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {metadata_path}")
    with open(metadata_path) as f:
        metadata = json.load(f)
    n_latents = metadata["n_latents"]
    print(f"Experiment: {data_dir.name}")
    print(f"  n_latents: {n_latents:,}")

    # Load node labels
    labels = load_labels(data_dir, n_latents)
    print(f"  Loaded labels for {len(labels):,} latents")

    # Load raw label data for GraphML attributes
    label_data = None
    labels_path = data_dir / "raw_activations" / "latent_labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            label_data = json.load(f)

    # Map type names to files
    type_to_file = {
        "cosine": "decoder_similarity.npz",
        "jaccard": "jaccard_similarity.npz",
        "coactivation": "coactivation.npz",
    }

    for edge_type in args.types:
        filename = type_to_file[edge_type]
        edges_path = graph_dir / filename

        if not edges_path.exists():
            print(f"\nSkipping {edge_type}: {edges_path} not found")
            continue

        print(f"\nExporting {edge_type} similarity...")
        edge_list = load_edges(
            edges_path,
            threshold=args.threshold,
            max_edges_per_node=args.max_edges_per_node,
        )

        if not edge_list:
            print(f"  No edges after filtering, skipping")
            continue

        if args.format in ("pairwise", "both"):
            out_path = output_dir / f"{edge_type}_similarity.txt"
            export_pairwise(edge_list, labels, out_path, edge_type)

        if args.format in ("graphml", "both"):
            out_path = output_dir / f"{edge_type}_similarity.graphml"
            export_graphml(
                edge_list, labels, n_latents, out_path, edge_type, label_data,
            )

    print(f"\nDone! Output in {output_dir}")


if __name__ == "__main__":
    main()
