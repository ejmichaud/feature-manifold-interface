"""
Graph data endpoints.

Provides access to:
- Node positions (from UMAP)
- Edges (cosine similarity, Jaccard similarity, co-activation)
- Individual latent data
"""

from enum import Enum
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from services.data_loader import DataLoader


router = APIRouter()


class EdgeType(str, Enum):
    """Available edge types."""
    cosine = "cosine"
    jaccard = "jaccard"
    coactivation = "coactivation"


class GraphResponse(BaseModel):
    """Response for graph endpoint."""
    n_latents: int
    positions: list[list[float]]  # [[x, y], ...]
    edge_types: list[str]
    experiment_id: str | None = None


class EdgeResponse(BaseModel):
    """Response for edges endpoint."""
    edge_type: str
    threshold: float
    n_edges: int
    edges: list[dict]  # [{source, target, weight}, ...]
    experiment_id: str | None = None


class LatentResponse(BaseModel):
    """Response for latent endpoint."""
    latent_id: int
    n_tokens: int
    token_indices: list[int]
    activations: list[float]
    experiment_id: str | None = None


async def get_data_loader(request: Request, experiment_id: str | None = None) -> tuple[DataLoader, str]:
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

    loader = await manager.get_loader(experiment_id)
    return loader, experiment_id


@router.get("/graph", response_model=GraphResponse)
async def get_graph(
    request: Request,
    experiment_id: str | None = None,
) -> GraphResponse:
    """
    Get graph node positions.

    Returns UMAP-projected positions for all latents.
    """
    data_loader, exp_id = await get_data_loader(request, experiment_id)
    positions = data_loader.positions.tolist()

    return GraphResponse(
        n_latents=data_loader.n_latents,
        positions=positions,
        edge_types=["cosine", "jaccard", "coactivation"],
        experiment_id=exp_id,
    )


@router.get("/edges", response_model=EdgeResponse)
async def get_edges(
    request: Request,
    type: EdgeType = EdgeType.cosine,
    threshold: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
    experiment_id: str | None = None,
) -> EdgeResponse:
    """
    Get edges between latents.

    Args:
        type: Type of similarity (cosine, jaccard, coactivation)
        threshold: Minimum edge weight to include
        experiment_id: Experiment ID

    Returns:
        List of edges with source, target, and weight
    """
    data_loader, exp_id = await get_data_loader(request, experiment_id)

    try:
        edges_matrix = data_loader.get_edges(type.value, threshold)
        edge_list = data_loader.edges_to_list(edges_matrix)

        return EdgeResponse(
            edge_type=type.value,
            threshold=threshold,
            n_edges=len(edge_list),
            edges=edge_list,
            experiment_id=exp_id,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/latent/{latent_id}", response_model=LatentResponse)
async def get_latent(
    latent_id: int,
    request: Request,
    experiment_id: str | None = None,
) -> LatentResponse:
    """
    Get data for a specific latent.

    Returns token indices and activation values for this latent.
    """
    data_loader, exp_id = await get_data_loader(request, experiment_id)

    if latent_id < 0 or latent_id >= data_loader.n_latents:
        raise HTTPException(
            status_code=404,
            detail=f"Latent {latent_id} not found. Valid range: 0-{data_loader.n_latents - 1}",
        )

    latent_data = data_loader.get_latent_data(latent_id)

    return LatentResponse(
        latent_id=latent_id,
        n_tokens=len(latent_data["token_indices"]),
        token_indices=latent_data["token_indices"].tolist(),
        activations=latent_data["activations"].astype(float).tolist(),
        experiment_id=exp_id,
    )
