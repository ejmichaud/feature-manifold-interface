"""
Point cloud generation service with incremental updates.

Implements the algorithm for building activation point clouds from
clusters of SAE latents.
"""

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from sklearn.decomposition import PCA

from .data_loader import DataLoader


@dataclass
class PointData:
    """Data for a single point in the cloud."""
    vector: np.ndarray  # Current point in d_model space
    active_latents: dict[int, float] = field(default_factory=dict)  # latent_id -> activation


class PointCloud:
    """
    Manages a point cloud with incremental updates.

    Each point corresponds to a token where at least one cluster latent fires.
    The point vector is the sum of (activation * decoder_vector) for each
    active cluster latent.
    """

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.points: dict[int, PointData] = {}  # token_idx -> PointData
        self.cluster_latents: set[int] = set()

    @property
    def n_points(self) -> int:
        """Number of points in the cloud."""
        return len(self.points)

    def add_latent(self, latent_id: int) -> int:
        """
        Add a latent to the cluster.

        Updates existing points and adds new points for tokens where
        this latent fires but no other cluster latent did.

        Args:
            latent_id: ID of latent to add

        Returns:
            Number of points added
        """
        if latent_id in self.cluster_latents:
            return 0

        self.cluster_latents.add(latent_id)

        # Load latent data
        latent_data = self.data_loader.get_latent_data(latent_id)
        token_indices = latent_data["token_indices"]
        activations = latent_data["activations"].astype(np.float32)

        # Get decoder vector
        decoder_vec = self.data_loader.get_decoder_vector(latent_id).astype(np.float32)

        points_added = 0

        for tok_idx, act in zip(token_indices, activations):
            tok_idx = int(tok_idx)
            act = float(act)

            if tok_idx in self.points:
                # Update existing point
                self.points[tok_idx].vector += act * decoder_vec
                self.points[tok_idx].active_latents[latent_id] = act
            else:
                # Create new point
                self.points[tok_idx] = PointData(
                    vector=(act * decoder_vec).copy(),
                    active_latents={latent_id: act},
                )
                points_added += 1

        return points_added

    def remove_latent(self, latent_id: int) -> int:
        """
        Remove a latent from the cluster.

        Updates points that had this latent and removes points that
        only had this latent.

        Args:
            latent_id: ID of latent to remove

        Returns:
            Number of points removed
        """
        if latent_id not in self.cluster_latents:
            return 0

        self.cluster_latents.discard(latent_id)

        # Get decoder vector
        decoder_vec = self.data_loader.get_decoder_vector(latent_id).astype(np.float32)

        tokens_to_remove = []
        points_removed = 0

        for tok_idx, point in self.points.items():
            if latent_id in point.active_latents:
                act = point.active_latents[latent_id]
                point.vector -= act * decoder_vec
                del point.active_latents[latent_id]

                # If no latents left, mark for removal
                if not point.active_latents:
                    tokens_to_remove.append(tok_idx)
                    points_removed += 1

        for tok_idx in tokens_to_remove:
            del self.points[tok_idx]

        return points_removed

    def clear(self) -> None:
        """Clear the point cloud."""
        self.points.clear()
        self.cluster_latents.clear()

    def set_cluster(self, latent_ids: list[int]) -> None:
        """
        Set the cluster to the specified latents.

        More efficient than add/remove when changing multiple latents.

        Args:
            latent_ids: List of latent IDs for the new cluster
        """
        self.clear()
        for latent_id in latent_ids:
            self.add_latent(latent_id)

    def get_vectors(self, max_points: int | None = None) -> np.ndarray:
        """
        Get point vectors as a matrix.

        Args:
            max_points: If set, subsample to this many points

        Returns:
            (n_points, d_model) array of point vectors
        """
        if not self.points:
            return np.zeros((0, self.data_loader.d_model), dtype=np.float32)

        # Get all point vectors
        token_indices = list(self.points.keys())
        vectors = np.array([self.points[t].vector for t in token_indices], dtype=np.float32)

        # Subsample if needed
        if max_points is not None and len(vectors) > max_points:
            indices = np.random.choice(len(vectors), max_points, replace=False)
            vectors = vectors[indices]
            token_indices = [token_indices[i] for i in indices]

        return vectors

    def get_token_indices(self) -> list[int]:
        """Get list of token indices in the cloud."""
        return list(self.points.keys())

    def compute_pca(
        self,
        n_components: int = 16,
        max_points: int = 100000,
    ) -> dict:
        """
        Compute PCA projection of the point cloud.

        Args:
            n_components: Number of PCA components
            max_points: Maximum points to use (subsamples if exceeded)

        Returns:
            dict with:
                - "points": (n_points, n_components) projected points
                - "explained_variance_ratio": variance explained by each component
                - "token_indices": corresponding token indices
                - "subsampled": whether subsampling was applied
        """
        vectors = self.get_vectors()

        if len(vectors) == 0:
            return {
                "points": np.zeros((0, n_components), dtype=np.float32),
                "explained_variance_ratio": np.zeros(n_components, dtype=np.float32),
                "token_indices": [],
                "subsampled": False,
            }

        # Subsample if needed
        subsampled = False
        token_indices = self.get_token_indices()

        if len(vectors) > max_points:
            indices = np.random.choice(len(vectors), max_points, replace=False)
            vectors = vectors[indices]
            token_indices = [token_indices[i] for i in indices]
            subsampled = True

        # Fit PCA
        n_components = min(n_components, len(vectors), vectors.shape[1])
        pca = PCA(n_components=n_components)
        projected = pca.fit_transform(vectors)

        return {
            "points": projected.astype(np.float32),
            "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
            "token_indices": token_indices,
            "subsampled": subsampled,
        }


class PointCloudManager:
    """
    Manages point cloud instances for different sessions/users.

    For now, uses a simple in-memory storage. Could be extended with
    session management for multi-user support.
    """

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self._clouds: dict[str, PointCloud] = {}

    def get_or_create(self, session_id: str = "default") -> PointCloud:
        """Get or create a point cloud for a session."""
        if session_id not in self._clouds:
            self._clouds[session_id] = PointCloud(self.data_loader)
        return self._clouds[session_id]

    def delete(self, session_id: str) -> None:
        """Delete a session's point cloud."""
        self._clouds.pop(session_id, None)
