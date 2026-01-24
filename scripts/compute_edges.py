"""
Compute graph edges between SAE latents.

This script computes multiple types of edges for the graph visualization:
1. Decoder similarity edges: latents with similar decoder directions (cosine)
2. Jaccard similarity edges: latents with similar token sets (Jaccard index)
3. Co-activation edges: latents that frequently fire together on the same tokens

Also computes initial 2D positions using UMAP on decoder vectors.

SPARSE STORAGE (top-k approach):
    A dense 65k × 65k matrix would be ~17GB. Instead, we store only the
    top-k neighbors for each latent. This guarantees bounded storage:

    n_latents × top_k = 65k × 100 = 6.5M edges max

    The UI can then filter displayed edges by a threshold on the stored values.
    Every latent will have edges stored (no orphans due to hard threshold).

Input:
    data/latents/       # Per-latent activation files (from build_indices.py)
    data/decoder.npy    # SAE decoder matrix

Output:
    data/graph/
        positions.npy           # (n_latents, 2) UMAP positions
        decoder_similarity.npz  # Sparse matrix (values = cosine similarity)
        jaccard_similarity.npz  # Sparse matrix (values = Jaccard index)
        coactivation.npz        # Sparse matrix (values = co-occurrence count)
        metadata.json           # Computation parameters

Usage:
    # Full computation (recommended):
    python compute_edges.py --data-dir ../data --top-k 100

    # Quick (cosine similarity only):
    python compute_edges.py --data-dir ../data --top-k 50 \\
                            --skip-coactivation --skip-jaccard
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from tqdm import tqdm

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed. Will skip UMAP positioning.")


def load_latent_tokens(latents_dir: Path, n_latents: int) -> list[set[int]]:
    """Load token sets for each latent."""
    print("Loading latent token sets...")
    latent_tokens = []

    for latent_id in tqdm(range(n_latents)):
        path = latents_dir / f"{latent_id:05d}.npz"
        if path.exists():
            data = np.load(path)
            tokens = set(data["token_indices"].tolist())
        else:
            tokens = set()
        latent_tokens.append(tokens)

    return latent_tokens


def compute_coactivation_edges(
    latent_tokens: list[set[int]],
    top_k: int = 100,
    min_activations: int = 10,
) -> sparse.csr_matrix:
    """
    Compute co-activation edges.

    For each latent, stores its top-k neighbors by co-activation count.
    The UI can further filter by threshold at display time.

    Args:
        latent_tokens: List of token sets, one per latent
        top_k: Number of neighbors to store per latent
        min_activations: Minimum activations for a latent to be considered
    """
    n_latents = len(latent_tokens)
    print(f"Computing co-activation for {n_latents} latents (top-{top_k} per latent)...")

    # Filter to latents with enough activations
    active_latents = [i for i in range(n_latents) if len(latent_tokens[i]) >= min_activations]
    print(f"  {len(active_latents)} latents have >= {min_activations} activations")

    # Build inverted index: token -> latents that fire on it
    print("  Building inverted index...")
    token_to_latents = {}
    for latent_id in tqdm(active_latents, desc="  Indexing"):
        for tok in latent_tokens[latent_id]:
            if tok not in token_to_latents:
                token_to_latents[tok] = []
            token_to_latents[tok].append(latent_id)

    # Count co-occurrences
    print("  Counting co-occurrences...")
    cooccurrence = {}  # (i, j) -> count, where i < j

    for tok, latents in tqdm(token_to_latents.items(), desc="  Counting"):
        for i, lat_i in enumerate(latents):
            for lat_j in latents[i + 1:]:
                key = (min(lat_i, lat_j), max(lat_i, lat_j))
                cooccurrence[key] = cooccurrence.get(key, 0) + 1

    print(f"  Found {len(cooccurrence):,} latent pairs with co-occurrences")

    # Build edges per latent
    edges_per_latent = {i: [] for i in range(n_latents)}
    for (i, j), count in cooccurrence.items():
        edges_per_latent[i].append((j, count))
        edges_per_latent[j].append((i, count))

    # Keep top-k per latent
    print(f"  Keeping top-{top_k} edges per latent...")
    rows, cols, data = [], [], []

    for latent_id in range(n_latents):
        edges = edges_per_latent[latent_id]
        edges.sort(key=lambda x: -x[1])  # Sort by count descending
        for neighbor, count in edges[:top_k]:
            rows.append(latent_id)
            cols.append(neighbor)
            data.append(count)

    # Build sparse matrix
    coact_matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_latents, n_latents),
        dtype=np.float32,
    )

    # Symmetrize
    coact_matrix = (coact_matrix + coact_matrix.T) / 2

    n_edges = coact_matrix.nnz // 2
    print(f"  Created co-activation graph with {n_edges:,} edges")

    return coact_matrix


def compute_decoder_similarity_edges(
    decoder: np.ndarray,
    top_k: int = 100,
) -> sparse.csr_matrix:
    """
    Compute edges based on decoder direction similarity (cosine).

    For each latent, stores its top-k most similar neighbors.
    The UI can further filter by threshold at display time.

    Args:
        decoder: (n_latents, d_model) decoder matrix
        top_k: Number of neighbors to store per latent

    Returns:
        Sparse symmetric matrix where entry (i,j) = cosine similarity
    """
    n_latents = decoder.shape[0]
    print(f"Computing decoder similarity for {n_latents} latents (top-{top_k} per latent)...")

    # Normalize decoder vectors
    decoder_norm = normalize(decoder, axis=1)

    # Batched computation to avoid memory issues
    batch_size = 1000
    rows, cols, data = [], [], []

    for i in tqdm(range(0, n_latents, batch_size), desc="  Computing similarities"):
        batch_end = min(i + batch_size, n_latents)
        batch = decoder_norm[i:batch_end]

        # Compute similarities with all latents
        sims = batch @ decoder_norm.T  # (batch_size, n_latents)

        for j, sim_row in enumerate(sims):
            latent_id = i + j
            sim_row[latent_id] = -np.inf  # Exclude self

            # Get top-k indices
            top_indices = np.argpartition(sim_row, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(-sim_row[top_indices])]

            for neighbor in top_indices:
                rows.append(latent_id)
                cols.append(neighbor)
                data.append(float(sim_row[neighbor]))

    # Build sparse matrix
    sim_matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_latents, n_latents),
        dtype=np.float32,
    )

    # Symmetrize (take max of both directions)
    sim_matrix = sim_matrix.maximum(sim_matrix.T)

    n_edges = sim_matrix.nnz // 2
    print(f"  Created similarity graph with {n_edges:,} edges")

    return sim_matrix


def compute_jaccard_similarity_edges(
    latent_tokens: list[set[int]],
    top_k: int = 100,
    min_tokens: int = 10,
) -> sparse.csr_matrix:
    """
    Compute edges based on Jaccard similarity of token sets.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    For each latent, stores its top-k most similar neighbors by Jaccard.
    The UI can further filter by threshold at display time.

    Args:
        latent_tokens: List of token sets, one per latent
        top_k: Number of neighbors to store per latent
        min_tokens: Minimum tokens for a latent to be considered
    """
    n_latents = len(latent_tokens)
    print(f"Computing Jaccard similarity for {n_latents} latents (top-{top_k} per latent)...")

    # Filter to latents with enough tokens
    active_latents = [i for i in range(n_latents) if len(latent_tokens[i]) >= min_tokens]
    print(f"  {len(active_latents)} latents have >= {min_tokens} tokens")

    # Build inverted index for efficient intersection computation
    print("  Building inverted index...")
    token_to_latents: dict[int, list[int]] = {}
    for latent_id in tqdm(active_latents, desc="  Indexing"):
        for tok in latent_tokens[latent_id]:
            if tok not in token_to_latents:
                token_to_latents[tok] = []
            token_to_latents[tok].append(latent_id)

    # Compute intersection sizes using inverted index
    print("  Computing intersection sizes...")
    intersection_counts: dict[tuple[int, int], int] = {}

    for tok, latents_on_tok in tqdm(token_to_latents.items(), desc="  Counting intersections"):
        for i, lat_i in enumerate(latents_on_tok):
            for lat_j in latents_on_tok[i + 1:]:
                key = (min(lat_i, lat_j), max(lat_i, lat_j))
                intersection_counts[key] = intersection_counts.get(key, 0) + 1

    print(f"  Found {len(intersection_counts):,} latent pairs with shared tokens")

    # Compute Jaccard scores for all pairs
    print("  Computing Jaccard scores...")
    edges_per_latent: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n_latents)}

    for (i, j), intersection_size in tqdm(intersection_counts.items(), desc="  Jaccard"):
        union_size = len(latent_tokens[i]) + len(latent_tokens[j]) - intersection_size
        if union_size == 0:
            continue

        jaccard = intersection_size / union_size
        edges_per_latent[i].append((j, jaccard))
        edges_per_latent[j].append((i, jaccard))

    # Keep top-k per latent
    print(f"  Keeping top-{top_k} edges per latent...")
    rows, cols, data = [], [], []

    for latent_id in range(n_latents):
        edges = edges_per_latent[latent_id]
        edges.sort(key=lambda x: -x[1])  # Sort by Jaccard descending
        for neighbor, jaccard in edges[:top_k]:
            rows.append(latent_id)
            cols.append(neighbor)
            data.append(jaccard)

    # Build sparse matrix
    jaccard_matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_latents, n_latents),
        dtype=np.float32,
    )

    # Symmetrize (take max of both directions)
    jaccard_matrix = jaccard_matrix.maximum(jaccard_matrix.T)

    n_edges = jaccard_matrix.nnz // 2
    print(f"  Created Jaccard similarity graph with {n_edges:,} edges")

    return jaccard_matrix


def compute_umap_positions(decoder: np.ndarray, n_neighbors: int = 15) -> np.ndarray:
    """Compute 2D positions using UMAP on decoder vectors."""
    if not HAS_UMAP:
        print("UMAP not available, using random positions")
        n_latents = decoder.shape[0]
        return np.random.randn(n_latents, 2).astype(np.float32)

    print("Computing UMAP positions...")
    n_latents = decoder.shape[0]

    # For very large n_latents, sample then interpolate
    if n_latents > 50000:
        print(f"  Large latent count ({n_latents}), sampling for UMAP...")
        sample_size = 50000
        sample_idx = np.random.choice(n_latents, sample_size, replace=False)

        umap = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
        sample_positions = umap.fit_transform(decoder[sample_idx])

        # For remaining latents, find nearest sampled latent
        print("  Interpolating remaining positions...")
        positions = np.zeros((n_latents, 2), dtype=np.float32)
        positions[sample_idx] = sample_positions

        remaining_idx = np.setdiff1d(np.arange(n_latents), sample_idx)
        decoder_norm = normalize(decoder, axis=1)

        for i in tqdm(range(0, len(remaining_idx), 1000), desc="  Interpolating"):
            batch_idx = remaining_idx[i:i + 1000]
            batch = decoder_norm[batch_idx]

            # Find nearest sampled latent
            sims = batch @ decoder_norm[sample_idx].T
            nearest = sample_idx[sims.argmax(axis=1)]
            positions[batch_idx] = positions[nearest]

            # Add small jitter
            positions[batch_idx] += np.random.randn(len(batch_idx), 2) * 0.01
    else:
        umap = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
        positions = umap.fit_transform(decoder)

    # Normalize to [0, 1] range
    positions = positions - positions.min(axis=0)
    positions = positions / positions.max(axis=0)

    print(f"  Computed positions for {n_latents} latents")
    return positions.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Compute graph edges between SAE latents (sparse top-k storage)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Uses top-k storage: for each latent, stores its k most similar neighbors.
Total edges bounded by n_latents × top_k. UI can filter by threshold at display time.
        """
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Data directory containing decoder.npy and latents/")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of neighbors to store per latent (default: 100)")
    parser.add_argument("--skip-coactivation", action="store_true",
                        help="Skip co-activation computation (slow for large datasets)")
    parser.add_argument("--skip-jaccard", action="store_true",
                        help="Skip Jaccard similarity computation (requires latent indices)")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    latents_dir = data_dir / "latents"
    graph_dir = data_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    n_latents = metadata["n_latents"]

    # Load decoder
    decoder_path = data_dir / "decoder.npy"
    if not decoder_path.exists():
        raise FileNotFoundError(f"Decoder not found at {decoder_path}")
    decoder = np.load(decoder_path)
    print(f"Loaded decoder: shape {decoder.shape}")

    # Compute UMAP positions
    positions = compute_umap_positions(decoder)
    np.save(graph_dir / "positions.npy", positions)
    print(f"Saved positions to {graph_dir / 'positions.npy'}")

    # Compute decoder similarity edges (always computed)
    sim_edges = compute_decoder_similarity_edges(decoder, top_k=args.top_k)
    sparse.save_npz(graph_dir / "decoder_similarity.npz", sim_edges)
    sim_n_edges = sim_edges.nnz // 2

    # Load latent token sets if needed for co-activation or Jaccard
    latent_tokens = None
    coact_n_edges = 0
    jaccard_n_edges = 0

    if not args.skip_coactivation or not args.skip_jaccard:
        latent_tokens = load_latent_tokens(latents_dir, n_latents)

    # Compute co-activation edges
    if not args.skip_coactivation:
        coact_edges = compute_coactivation_edges(latent_tokens, top_k=args.top_k)
        sparse.save_npz(graph_dir / "coactivation.npz", coact_edges)
        coact_n_edges = coact_edges.nnz // 2
    else:
        print("Skipped co-activation computation")

    # Compute Jaccard similarity edges
    if not args.skip_jaccard:
        jaccard_edges = compute_jaccard_similarity_edges(latent_tokens, top_k=args.top_k)
        sparse.save_npz(graph_dir / "jaccard_similarity.npz", jaccard_edges)
        jaccard_n_edges = jaccard_edges.nnz // 2
    else:
        print("Skipped Jaccard similarity computation")

    # Save metadata (for UI to know what's available)
    edge_metadata = {
        "n_latents": n_latents,
        "top_k": args.top_k,
        "edges": {
            "decoder_similarity": {"n_edges": sim_n_edges, "value_type": "cosine"},
            "jaccard_similarity": {"n_edges": jaccard_n_edges, "value_type": "jaccard"} if not args.skip_jaccard else None,
            "coactivation": {"n_edges": coact_n_edges, "value_type": "count"} if not args.skip_coactivation else None,
        }
    }
    with open(graph_dir / "metadata.json", "w") as f:
        json.dump(edge_metadata, f, indent=2)

    print(f"\nEdge computation complete! Output in {graph_dir}")
    print(f"  decoder_similarity: {sim_n_edges:,} edges")
    if not args.skip_jaccard:
        print(f"  jaccard_similarity: {jaccard_n_edges:,} edges")
    if not args.skip_coactivation:
        print(f"  coactivation: {coact_n_edges:,} edges")


if __name__ == "__main__":
    main()
