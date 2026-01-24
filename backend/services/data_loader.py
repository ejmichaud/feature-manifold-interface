"""
Data loading utilities for the Feature Manifold Interface.

Handles loading and caching of:
- Decoder matrix
- Graph positions and edges
- Per-latent activation data
"""

import json
from pathlib import Path
from functools import lru_cache

import numpy as np
from scipy import sparse


class DataLoader:
    """
    Manages loading and caching of data files.

    Data is loaded on startup and kept in memory for fast access.
    Per-latent files are loaded on demand and cached.
    """

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.latents_dir = self.data_dir / "latents"
        self.graph_dir = self.data_dir / "graph"

        # Will be loaded on startup
        self.decoder: np.ndarray | None = None
        self.positions: np.ndarray | None = None
        self.metadata: dict | None = None

        # Edge matrices (loaded on demand)
        self._cosine_edges: sparse.csr_matrix | None = None
        self._jaccard_edges: sparse.csr_matrix | None = None
        self._coactivation_edges: sparse.csr_matrix | None = None

        # Latent data cache
        self._latent_cache: dict[int, dict] = {}
        self._cache_max_size = 1000  # Max latents to keep in memory

    @property
    def n_latents(self) -> int:
        """Number of latents in the SAE."""
        return self.decoder.shape[0] if self.decoder is not None else 0

    @property
    def d_model(self) -> int:
        """Model dimension."""
        return self.decoder.shape[1] if self.decoder is not None else 0

    async def load(self) -> None:
        """Load essential data on startup."""
        # Load decoder matrix
        decoder_path = self.data_dir / "decoder.npy"
        if decoder_path.exists():
            self.decoder = np.load(decoder_path)
            print(f"  Loaded decoder: {self.decoder.shape}")
        else:
            raise FileNotFoundError(f"Decoder not found at {decoder_path}")

        # Load positions
        positions_path = self.graph_dir / "positions.npy"
        if positions_path.exists():
            self.positions = np.load(positions_path)
            print(f"  Loaded positions: {self.positions.shape}")
        else:
            print(f"  Warning: Positions not found at {positions_path}")
            # Generate random positions as fallback
            self.positions = np.random.rand(self.n_latents, 2).astype(np.float32)

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            print(f"  Loaded metadata")
        else:
            self.metadata = {"n_latents": self.n_latents, "d_model": self.d_model}

    def get_edges(self, edge_type: str = "cosine", threshold: float = 0.0) -> sparse.csr_matrix:
        """
        Get edge matrix for specified type.

        Args:
            edge_type: "cosine", "jaccard", or "coactivation"
            threshold: Filter edges below this value

        Returns:
            Sparse matrix of edges
        """
        if edge_type == "cosine":
            if self._cosine_edges is None:
                path = self.graph_dir / "decoder_similarity.npz"
                if path.exists():
                    self._cosine_edges = sparse.load_npz(path)
                else:
                    raise FileNotFoundError(f"Cosine edges not found at {path}")
            edges = self._cosine_edges

        elif edge_type == "jaccard":
            if self._jaccard_edges is None:
                path = self.graph_dir / "jaccard_similarity.npz"
                if path.exists():
                    self._jaccard_edges = sparse.load_npz(path)
                else:
                    raise FileNotFoundError(f"Jaccard edges not found at {path}")
            edges = self._jaccard_edges

        elif edge_type == "coactivation":
            if self._coactivation_edges is None:
                path = self.graph_dir / "coactivation.npz"
                if path.exists():
                    self._coactivation_edges = sparse.load_npz(path)
                else:
                    raise FileNotFoundError(f"Coactivation edges not found at {path}")
            edges = self._coactivation_edges

        else:
            raise ValueError(f"Unknown edge type: {edge_type}")

        # Apply threshold filter
        if threshold > 0:
            edges = edges.multiply(edges >= threshold)
            edges.eliminate_zeros()

        return edges

    def get_latent_data(self, latent_id: int) -> dict:
        """
        Get token indices and activations for a latent.

        Returns:
            dict with "token_indices" (int64 array) and "activations" (float16 array)
        """
        if latent_id in self._latent_cache:
            return self._latent_cache[latent_id]

        path = self.latents_dir / f"{latent_id:05d}.npz"
        if not path.exists():
            return {"token_indices": np.array([], dtype=np.int64),
                    "activations": np.array([], dtype=np.float16)}

        data = dict(np.load(path))

        # Cache management
        if len(self._latent_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._latent_cache))
            del self._latent_cache[oldest_key]

        self._latent_cache[latent_id] = data
        return data

    def get_decoder_vector(self, latent_id: int) -> np.ndarray:
        """Get decoder vector for a latent."""
        return self.decoder[latent_id]

    def edges_to_list(self, edges: sparse.csr_matrix) -> list[dict]:
        """
        Convert sparse edge matrix to list format for JSON response.

        Returns:
            List of {"source": int, "target": int, "weight": float}
        """
        # Get COO format for easy iteration
        coo = edges.tocoo()

        # Only return upper triangle to avoid duplicates
        edge_list = []
        for i, j, v in zip(coo.row, coo.col, coo.data):
            if i < j:  # Upper triangle only
                edge_list.append({
                    "source": int(i),
                    "target": int(j),
                    "weight": float(v),
                })

        return edge_list
