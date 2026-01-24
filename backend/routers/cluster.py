"""
Cluster and point cloud endpoints.

Provides operations for:
- Building point clouds from latent clusters
- Computing PCA projections
- Incremental cluster updates
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from services.data_loader import DataLoader
from services.point_cloud import PointCloud, PointCloudManager


router = APIRouter()

# Global point cloud manager (initialized on first request)
_cloud_manager: PointCloudManager | None = None


def get_data_loader(request: Request) -> DataLoader:
    """Dependency to get data loader from app state."""
    return request.app.state.data_loader


def get_cloud_manager(request: Request) -> PointCloudManager:
    """Dependency to get or create point cloud manager."""
    global _cloud_manager
    if _cloud_manager is None:
        _cloud_manager = PointCloudManager(request.app.state.data_loader)
    return _cloud_manager


# Request/Response models

class ClusterRequest(BaseModel):
    """Request to set or update a cluster."""
    latent_ids: list[int] = Field(..., description="List of latent IDs in the cluster")
    session_id: str = Field(default="default", description="Session ID for the point cloud")


class AddLatentRequest(BaseModel):
    """Request to add a latent to the cluster."""
    latent_id: int = Field(..., description="Latent ID to add")
    session_id: str = Field(default="default")


class RemoveLatentRequest(BaseModel):
    """Request to remove a latent from the cluster."""
    latent_id: int = Field(..., description="Latent ID to remove")
    session_id: str = Field(default="default")


class PCARequest(BaseModel):
    """Request for PCA projection."""
    latent_ids: list[int] = Field(..., description="List of latent IDs")
    n_components: int = Field(default=16, ge=1, le=100)
    max_points: int = Field(default=100000, ge=1)
    session_id: str = Field(default="default")


class CloudResponse(BaseModel):
    """Response with point cloud info."""
    n_points: int
    cluster_latents: list[int]
    session_id: str


class PCAResponse(BaseModel):
    """Response with PCA projection."""
    n_points: int
    n_components: int
    points: list[list[float]]  # (n_points, n_components)
    explained_variance_ratio: list[float]
    subsampled: bool
    cluster_latents: list[int]


# Endpoints

@router.post("/cluster/set", response_model=CloudResponse)
async def set_cluster(
    request: ClusterRequest,
    data_loader: Annotated[DataLoader, Depends(get_data_loader)],
    cloud_manager: Annotated[PointCloudManager, Depends(get_cloud_manager)],
) -> CloudResponse:
    """
    Set the cluster to the specified latents.

    Replaces the current cluster entirely.
    """
    # Validate latent IDs
    for latent_id in request.latent_ids:
        if latent_id < 0 or latent_id >= data_loader.n_latents:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid latent ID: {latent_id}",
            )

    cloud = cloud_manager.get_or_create(request.session_id)
    cloud.set_cluster(request.latent_ids)

    return CloudResponse(
        n_points=cloud.n_points,
        cluster_latents=list(cloud.cluster_latents),
        session_id=request.session_id,
    )


@router.post("/cluster/add", response_model=CloudResponse)
async def add_to_cluster(
    request: AddLatentRequest,
    data_loader: Annotated[DataLoader, Depends(get_data_loader)],
    cloud_manager: Annotated[PointCloudManager, Depends(get_cloud_manager)],
) -> CloudResponse:
    """
    Add a latent to the current cluster.

    Uses incremental update for efficiency.
    """
    if request.latent_id < 0 or request.latent_id >= data_loader.n_latents:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid latent ID: {request.latent_id}",
        )

    cloud = cloud_manager.get_or_create(request.session_id)
    cloud.add_latent(request.latent_id)

    return CloudResponse(
        n_points=cloud.n_points,
        cluster_latents=list(cloud.cluster_latents),
        session_id=request.session_id,
    )


@router.post("/cluster/remove", response_model=CloudResponse)
async def remove_from_cluster(
    request: RemoveLatentRequest,
    data_loader: Annotated[DataLoader, Depends(get_data_loader)],
    cloud_manager: Annotated[PointCloudManager, Depends(get_cloud_manager)],
) -> CloudResponse:
    """
    Remove a latent from the current cluster.

    Uses incremental update for efficiency.
    """
    cloud = cloud_manager.get_or_create(request.session_id)
    cloud.remove_latent(request.latent_id)

    return CloudResponse(
        n_points=cloud.n_points,
        cluster_latents=list(cloud.cluster_latents),
        session_id=request.session_id,
    )


@router.post("/cluster/clear", response_model=CloudResponse)
async def clear_cluster(
    session_id: str = "default",
    cloud_manager: Annotated[PointCloudManager, Depends(get_cloud_manager)] = None,
) -> CloudResponse:
    """Clear the current cluster."""
    cloud = cloud_manager.get_or_create(session_id)
    cloud.clear()

    return CloudResponse(
        n_points=0,
        cluster_latents=[],
        session_id=session_id,
    )


@router.post("/cluster/pca", response_model=PCAResponse)
async def compute_pca(
    request: PCARequest,
    data_loader: Annotated[DataLoader, Depends(get_data_loader)],
    cloud_manager: Annotated[PointCloudManager, Depends(get_cloud_manager)],
) -> PCAResponse:
    """
    Compute PCA projection of the point cloud.

    If latent_ids differ from the current cluster, updates the cluster first.
    """
    # Validate latent IDs
    for latent_id in request.latent_ids:
        if latent_id < 0 or latent_id >= data_loader.n_latents:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid latent ID: {latent_id}",
            )

    cloud = cloud_manager.get_or_create(request.session_id)

    # Update cluster if different
    current_latents = set(cloud.cluster_latents)
    requested_latents = set(request.latent_ids)

    if current_latents != requested_latents:
        cloud.set_cluster(request.latent_ids)

    # Compute PCA
    result = cloud.compute_pca(
        n_components=request.n_components,
        max_points=request.max_points,
    )

    return PCAResponse(
        n_points=len(result["points"]),
        n_components=result["points"].shape[1] if len(result["points"]) > 0 else 0,
        points=result["points"].tolist(),
        explained_variance_ratio=result["explained_variance_ratio"].tolist(),
        subsampled=result["subsampled"],
        cluster_latents=list(cloud.cluster_latents),
    )


@router.get("/cluster/info", response_model=CloudResponse)
async def get_cluster_info(
    session_id: str = "default",
    cloud_manager: Annotated[PointCloudManager, Depends(get_cloud_manager)] = None,
) -> CloudResponse:
    """Get current cluster information."""
    cloud = cloud_manager.get_or_create(session_id)

    return CloudResponse(
        n_points=cloud.n_points,
        cluster_latents=list(cloud.cluster_latents),
        session_id=session_id,
    )
