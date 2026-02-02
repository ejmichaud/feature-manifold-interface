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

        # Corpus token data (for hover context)
        self._token_ids: np.ndarray | None = None  # global_token_index -> vocab_id
        self._token_map: np.ndarray | None = None  # global_token_index -> (doc_id, position)
        self._vocab: dict[int, str] | None = None   # vocab_id -> token string
        self._corpus_loaded = False

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

    def _load_corpus_data(self) -> bool:
        """Load corpus token data for hover context. Returns True if available."""
        if self._corpus_loaded:
            return self._token_ids is not None

        self._corpus_loaded = True
        corpus_dir = self.data_dir / "corpus"

        token_ids_path = corpus_dir / "token_ids.npy"
        vocab_path = corpus_dir / "vocab.json"
        token_map_path = corpus_dir / "token_map.npy"

        if not token_ids_path.exists() or not vocab_path.exists():
            print(f"  Corpus token data not found (run build_corpus_tokens.py)")
            return False

        self._token_ids = np.load(token_ids_path)
        print(f"  Loaded token_ids: {self._token_ids.shape}")

        with open(vocab_path) as f:
            raw_vocab = json.load(f)
        self._vocab = {int(k): v for k, v in raw_vocab.items()}
        print(f"  Loaded vocab: {len(self._vocab)} entries")

        if token_map_path.exists():
            self._token_map = np.load(token_map_path)
            print(f"  Loaded token_map: {self._token_map.shape}")

        return True

    def build_hover_texts(
        self,
        token_indices: list[int],
        context_size: int = 8,
    ) -> list[str] | None:
        """Build hover text strings for a list of token indices.

        Each hover text shows ~context_size preceding tokens plus the
        active token in bold. Document boundaries are respected.

        Args:
            token_indices: Global token indices to build hover for
            context_size: Number of preceding tokens to include

        Returns:
            List of HTML strings, or None if corpus data unavailable
        """
        if not self._load_corpus_data():
            return None

        token_ids = self._token_ids
        vocab = self._vocab
        token_map = self._token_map
        n_tokens = len(token_ids)

        hover_texts = []
        for tok_idx in token_indices:
            if tok_idx < 0 or tok_idx >= n_tokens:
                hover_texts.append("???")
                continue

            # Get the active token
            active_id = int(token_ids[tok_idx])
            if active_id < 0:
                hover_texts.append("???")
                continue

            active_str = vocab.get(active_id, "?")

            # Build context: walk backwards up to context_size tokens
            context_tokens = []
            if token_map is not None:
                doc_id = int(token_map[tok_idx, 0])
                for offset in range(1, context_size + 1):
                    prev_idx = tok_idx - offset
                    if prev_idx < 0:
                        break
                    # Check same document
                    if int(token_map[prev_idx, 0]) != doc_id:
                        break
                    prev_id = int(token_ids[prev_idx])
                    if prev_id < 0:
                        context_tokens.append("…")
                    else:
                        context_tokens.append(vocab.get(prev_id, "?"))
            else:
                # No token_map: just go backwards, hope for the best
                for offset in range(1, context_size + 1):
                    prev_idx = tok_idx - offset
                    if prev_idx < 0:
                        break
                    prev_id = int(token_ids[prev_idx])
                    if prev_id < 0:
                        context_tokens.append("…")
                    else:
                        context_tokens.append(vocab.get(prev_id, "?"))

            # Reverse to get natural order
            context_tokens.reverse()

            context_str = "".join(context_tokens)
            hover_texts.append(f"{context_str}<b>{active_str}</b>")

        return hover_texts
