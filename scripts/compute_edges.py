"""
Compute graph edges between SAE latents.

This script computes multiple types of edges for the graph visualization:
1. Co-activation edges: latents that frequently fire together on the same tokens
2. Decoder similarity edges: latents with similar decoder directions

Also computes initial 2D positions using UMAP on decoder vectors.

Input:
    data/latents/       # Per-latent activation files
    data/decoder.npy    # SAE decoder matrix

Output:
    data/graph/
        positions.npy           # (n_latents, 2) UMAP positions
        coactivation.npz        # Sparse edge matrix
        decoder_similarity.npz  # Sparse edge matrix
        metadata.json           # Edge computation parameters

Usage:
    python compute_edges.py --data-dir ../data \
                            --coact-threshold 100 \
                            --similarity-threshold 0.5 \
                            --top-k-edges 50
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
    threshold: int = 100,
    top_k: int = 50,
) -> sparse.csr_matrix:
    """
    Compute co-activation edges.

    Two latents are connected if they fire together on at least `threshold` tokens.
    Each latent keeps at most `top_k` edges to limit graph density.
    """
    n_latents = len(latent_tokens)
    print(f"Computing co-activation for {n_latents} latents...")

    # We'll build a sparse matrix
    rows, cols, data = [], [], []

    # For efficiency, only compare latents with enough activations
    active_latents = [i for i in range(n_latents) if len(latent_tokens[i]) >= threshold]
    print(f"  {len(active_latents)} latents have >= {threshold} activations")

    # This is O(n^2) in worst case, but we can optimize by:
    # 1. Only comparing latents with overlapping token sets
    # 2. Using inverted index

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
        # All pairs of latents that fire on this token
        for i, lat_i in enumerate(latents):
            for lat_j in latents[i + 1:]:
                key = (min(lat_i, lat_j), max(lat_i, lat_j))
                cooccurrence[key] = cooccurrence.get(key, 0) + 1

    print(f"  Found {len(cooccurrence)} latent pairs with co-occurrences")

    # Filter by threshold and top-k
    print(f"  Filtering to threshold={threshold}, top_k={top_k}...")
    edges_per_latent = {i: [] for i in range(n_latents)}

    for (i, j), count in cooccurrence.items():
        if count >= threshold:
            edges_per_latent[i].append((j, count))
            edges_per_latent[j].append((i, count))

    # Keep top-k per latent
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
    print(f"  Created co-activation graph with {n_edges} edges")

    return coact_matrix


def compute_decoder_similarity_edges(
    decoder: np.ndarray,
    threshold: float = 0.5,
    top_k: int = 50,
) -> sparse.csr_matrix:
    """
    Compute edges based on decoder direction similarity.

    Two latents are connected if their decoder vectors have cosine similarity
    above `threshold`. Each latent keeps at most `top_k` edges.
    """
    n_latents = decoder.shape[0]
    print(f"Computing decoder similarity for {n_latents} latents...")

    # Normalize decoder vectors
    decoder_norm = normalize(decoder, axis=1)

    # For large n_latents, we can't compute full similarity matrix
    # Use batched approach
    batch_size = 1000
    rows, cols, data = [], [], []

    for i in tqdm(range(0, n_latents, batch_size), desc="  Computing similarities"):
        batch_end = min(i + batch_size, n_latents)
        batch = decoder_norm[i:batch_end]

        # Compute similarities with all latents
        sims = batch @ decoder_norm.T  # (batch_size, n_latents)

        for j, sim_row in enumerate(sims):
            latent_id = i + j

            # Find top-k neighbors above threshold (excluding self)
            sim_row[latent_id] = -1  # Exclude self

            # Get indices above threshold
            above_thresh = np.where(sim_row >= threshold)[0]

            if len(above_thresh) == 0:
                continue

            # Sort by similarity and take top-k
            top_indices = above_thresh[np.argsort(-sim_row[above_thresh])[:top_k]]

            for neighbor in top_indices:
                rows.append(latent_id)
                cols.append(neighbor)
                data.append(sim_row[neighbor])

    # Build sparse matrix
    sim_matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_latents, n_latents),
        dtype=np.float32,
    )

    # Symmetrize (take max of both directions)
    sim_matrix = sim_matrix.maximum(sim_matrix.T)

    n_edges = sim_matrix.nnz // 2
    print(f"  Created similarity graph with {n_edges} edges")

    return sim_matrix


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
    parser = argparse.ArgumentParser(description="Compute graph edges")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    parser.add_argument("--coact-threshold", type=int, default=100,
                        help="Min co-activations for edge")
    parser.add_argument("--similarity-threshold", type=float, default=0.5,
                        help="Min cosine similarity for edge")
    parser.add_argument("--top-k-edges", type=int, default=50,
                        help="Max edges per latent")
    parser.add_argument("--skip-coactivation", action="store_true",
                        help="Skip co-activation computation (slow)")

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

    # Compute decoder similarity edges
    sim_edges = compute_decoder_similarity_edges(
        decoder,
        threshold=args.similarity_threshold,
        top_k=args.top_k_edges,
    )
    sparse.save_npz(graph_dir / "decoder_similarity.npz", sim_edges)
    print(f"Saved decoder similarity edges")

    # Compute co-activation edges
    if not args.skip_coactivation:
        latent_tokens = load_latent_tokens(latents_dir, n_latents)
        coact_edges = compute_coactivation_edges(
            latent_tokens,
            threshold=args.coact_threshold,
            top_k=args.top_k_edges,
        )
        sparse.save_npz(graph_dir / "coactivation.npz", coact_edges)
        print(f"Saved co-activation edges")
    else:
        print("Skipped co-activation computation")

    # Save edge metadata
    edge_metadata = {
        "coact_threshold": args.coact_threshold,
        "similarity_threshold": args.similarity_threshold,
        "top_k_edges": args.top_k_edges,
        "n_latents": n_latents,
    }
    with open(graph_dir / "metadata.json", "w") as f:
        json.dump(edge_metadata, f, indent=2)

    print(f"\nEdge computation complete! Output in {graph_dir}")


if __name__ == "__main__":
    main()
