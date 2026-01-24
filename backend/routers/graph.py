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


class EdgeResponse(BaseModel):
    """Response for edges endpoint."""
    edge_type: str
    threshold: float
    n_edges: int
    edges: list[dict]  # [{source, target, weight}, ...]


class LatentResponse(BaseModel):
    """Response for latent endpoint."""
    latent_id: int
    n_tokens: int
    token_indices: list[int]
    activations: list[float]


def get_data_loader(request: Request) -> DataLoader:
    """Dependency to get data loader from app state."""
    return request.app.state.data_loader


@router.get("/graph", response_model=GraphResponse)
async def get_graph(
    data_loader: Annotated[DataLoader, Depends(get_data_loader)],
) -> GraphResponse:
    """
    Get graph node positions.

    Returns UMAP-projected positions for all latents.
    """
    positions = data_loader.positions.tolist()

    return GraphResponse(
        n_latents=data_loader.n_latents,
        positions=positions,
        edge_types=["cosine", "jaccard", "coactivation"],
    )


@router.get("/edges", response_model=EdgeResponse)
async def get_edges(
    data_loader: Annotated[DataLoader, Depends(get_data_loader)],
    type: EdgeType = EdgeType.cosine,
    threshold: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
) -> EdgeResponse:
    """
    Get edges between latents.

    Args:
        type: Type of similarity (cosine, jaccard, coactivation)
        threshold: Minimum edge weight to include

    Returns:
        List of edges with source, target, and weight
    """
    try:
        edges_matrix = data_loader.get_edges(type.value, threshold)
        edge_list = data_loader.edges_to_list(edges_matrix)

        return EdgeResponse(
            edge_type=type.value,
            threshold=threshold,
            n_edges=len(edge_list),
            edges=edge_list,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/latent/{latent_id}", response_model=LatentResponse)
async def get_latent(
    latent_id: int,
    data_loader: Annotated[DataLoader, Depends(get_data_loader)],
) -> LatentResponse:
    """
    Get data for a specific latent.

    Returns token indices and activation values for this latent.
    """
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
    )
