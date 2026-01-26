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

# Point cloud managers per experiment
_cloud_managers: dict[str, PointCloudManager] = {}


async def get_data_loader(request: Request, experiment_id: str | None = None) -> DataLoader:
    """Dependency to get data loader for an experiment."""
    manager = request.app.state.experiment_manager

    # Use default if not specified
    if experiment_id is None:
        experiment_id = request.app.state.default_experiment
        if experiment_id is None:
            raise HTTPException(
                status_code=400,
                detail="No experiment specified and no default available"
            )

    return await manager.get_loader(experiment_id)


def get_cloud_manager(experiment_id: str, data_loader: DataLoader) -> PointCloudManager:
    """Get or create point cloud manager for an experiment."""
    if experiment_id not in _cloud_managers:
        _cloud_managers[experiment_id] = PointCloudManager(data_loader)
    return _cloud_managers[experiment_id]


# Request/Response models

class ClusterRequest(BaseModel):
    """Request to set or update a cluster."""
    latent_ids: list[int] = Field(..., description="List of latent IDs in the cluster")
    session_id: str = Field(default="default", description="Session ID for the point cloud")
    experiment_id: str | None = Field(default=None, description="Experiment ID")


class AddLatentRequest(BaseModel):
    """Request to add a latent to the cluster."""
    latent_id: int = Field(..., description="Latent ID to add")
    session_id: str = Field(default="default")
    experiment_id: str | None = Field(default=None)


class RemoveLatentRequest(BaseModel):
    """Request to remove a latent from the cluster."""
    latent_id: int = Field(..., description="Latent ID to remove")
    session_id: str = Field(default="default")
    experiment_id: str | None = Field(default=None)


class PCARequest(BaseModel):
    """Request for PCA projection."""
    latent_ids: list[int] = Field(..., description="List of latent IDs")
    n_components: int = Field(default=13, ge=1, le=100)
    max_points: int = Field(default=100000, ge=1)
    session_id: str = Field(default="default")
    experiment_id: str | None = Field(default=None)


class CloudResponse(BaseModel):
    """Response with point cloud info."""
    n_points: int
    cluster_latents: list[int]
    session_id: str
    experiment_id: str | None = None


class PCAResponse(BaseModel):
    """Response with PCA projection."""
    n_points: int
    n_components: int
    points: list[list[float]]  # (n_points, n_components)
    explained_variance_ratio: list[float]
    subsampled: bool
    cluster_latents: list[int]
    experiment_id: str | None = None


# Endpoints

@router.post("/cluster/set", response_model=CloudResponse)
async def set_cluster(
    request_body: ClusterRequest,
    request: Request,
) -> CloudResponse:
    """
    Set the cluster to the specified latents.

    Replaces the current cluster entirely.
    """
    experiment_id = request_body.experiment_id or request.app.state.default_experiment
    data_loader = await get_data_loader(request, experiment_id)

    # Validate latent IDs
    for latent_id in request_body.latent_ids:
        if latent_id < 0 or latent_id >= data_loader.n_latents:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid latent ID: {latent_id}",
            )

    cloud_manager = get_cloud_manager(experiment_id, data_loader)
    cloud = cloud_manager.get_or_create(request_body.session_id)
    cloud.set_cluster(request_body.latent_ids)

    return CloudResponse(
        n_points=cloud.n_points,
        cluster_latents=list(cloud.cluster_latents),
        session_id=request_body.session_id,
        experiment_id=experiment_id,
    )


@router.post("/cluster/add", response_model=CloudResponse)
async def add_to_cluster(
    request_body: AddLatentRequest,
    request: Request,
) -> CloudResponse:
    """
    Add a latent to the current cluster.

    Uses incremental update for efficiency.
    """
    experiment_id = request_body.experiment_id or request.app.state.default_experiment
    data_loader = await get_data_loader(request, experiment_id)

    if request_body.latent_id < 0 or request_body.latent_id >= data_loader.n_latents:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid latent ID: {request_body.latent_id}",
        )

    cloud_manager = get_cloud_manager(experiment_id, data_loader)
    cloud = cloud_manager.get_or_create(request_body.session_id)
    cloud.add_latent(request_body.latent_id)

    return CloudResponse(
        n_points=cloud.n_points,
        cluster_latents=list(cloud.cluster_latents),
        session_id=request_body.session_id,
        experiment_id=experiment_id,
    )


@router.post("/cluster/remove", response_model=CloudResponse)
async def remove_from_cluster(
    request_body: RemoveLatentRequest,
    request: Request,
) -> CloudResponse:
    """
    Remove a latent from the current cluster.

    Uses incremental update for efficiency.
    """
    experiment_id = request_body.experiment_id or request.app.state.default_experiment
    data_loader = await get_data_loader(request, experiment_id)

    cloud_manager = get_cloud_manager(experiment_id, data_loader)
    cloud = cloud_manager.get_or_create(request_body.session_id)
    cloud.remove_latent(request_body.latent_id)

    return CloudResponse(
        n_points=cloud.n_points,
        cluster_latents=list(cloud.cluster_latents),
        session_id=request_body.session_id,
        experiment_id=experiment_id,
    )


@router.post("/cluster/clear", response_model=CloudResponse)
async def clear_cluster(
    request: Request,
    session_id: str = "default",
    experiment_id: str | None = None,
) -> CloudResponse:
    """Clear the current cluster."""
    experiment_id = experiment_id or request.app.state.default_experiment
    data_loader = await get_data_loader(request, experiment_id)

    cloud_manager = get_cloud_manager(experiment_id, data_loader)
    cloud = cloud_manager.get_or_create(session_id)
    cloud.clear()

    return CloudResponse(
        n_points=0,
        cluster_latents=[],
        session_id=session_id,
        experiment_id=experiment_id,
    )


@router.post("/cluster/pca", response_model=PCAResponse)
async def compute_pca(
    request_body: PCARequest,
    request: Request,
) -> PCAResponse:
    """
    Compute PCA projection of the point cloud.

    If latent_ids differ from the current cluster, updates the cluster first.
    """
    experiment_id = request_body.experiment_id or request.app.state.default_experiment
    data_loader = await get_data_loader(request, experiment_id)

    # Validate latent IDs
    for latent_id in request_body.latent_ids:
        if latent_id < 0 or latent_id >= data_loader.n_latents:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid latent ID: {latent_id}",
            )

    cloud_manager = get_cloud_manager(experiment_id, data_loader)
    cloud = cloud_manager.get_or_create(request_body.session_id)

    # Update cluster if different
    current_latents = set(cloud.cluster_latents)
    requested_latents = set(request_body.latent_ids)

    if current_latents != requested_latents:
        cloud.set_cluster(request_body.latent_ids)

    # Compute PCA
    result = cloud.compute_pca(
        n_components=request_body.n_components,
        max_points=request_body.max_points,
    )

    return PCAResponse(
        n_points=len(result["points"]),
        n_components=result["points"].shape[1] if len(result["points"]) > 0 else 0,
        points=result["points"].tolist(),
        explained_variance_ratio=result["explained_variance_ratio"].tolist(),
        subsampled=result["subsampled"],
        cluster_latents=list(cloud.cluster_latents),
        experiment_id=experiment_id,
    )


@router.get("/cluster/info", response_model=CloudResponse)
async def get_cluster_info(
    request: Request,
    session_id: str = "default",
    experiment_id: str | None = None,
) -> CloudResponse:
    """Get current cluster information."""
    experiment_id = experiment_id or request.app.state.default_experiment
    data_loader = await get_data_loader(request, experiment_id)

    cloud_manager = get_cloud_manager(experiment_id, data_loader)
    cloud = cloud_manager.get_or_create(session_id)

    return CloudResponse(
        n_points=cloud.n_points,
        cluster_latents=list(cloud.cluster_latents),
        session_id=session_id,
        experiment_id=experiment_id,
    )
