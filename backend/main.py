"""
FastAPI backend for Feature Manifold Interface.

Provides API endpoints for:
- Graph data (positions, edges)
- Latent data (token indices, activations)
- Cluster operations (point cloud generation, PCA)

Supports multiple experiments - specify experiment_id in API requests.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from routers import graph, cluster
from services.data_loader import DataLoader


def get_experiments_root() -> Path:
    """Get experiments root directory from environment or default."""
    data_root = os.environ.get("DATA_ROOT", "../data")
    return Path(data_root) / "experiments"


class ExperimentManager:
    """Manages DataLoader instances for multiple experiments."""

    def __init__(self, experiments_root: Path):
        self.experiments_root = experiments_root
        self._loaders: dict[str, DataLoader] = {}

    def list_experiments(self) -> list[str]:
        """List available experiment IDs."""
        if not self.experiments_root.exists():
            return []
        return [
            d.name for d in self.experiments_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    async def get_loader(self, experiment_id: str) -> DataLoader:
        """Get or create DataLoader for an experiment."""
        if experiment_id in self._loaders:
            return self._loaders[experiment_id]

        exp_dir = self.experiments_root / experiment_id
        if not exp_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Experiment '{experiment_id}' not found"
            )

        print(f"Loading experiment: {experiment_id}...")
        loader = DataLoader(exp_dir)
        await loader.load()
        print(f"  Loaded: {loader.n_latents} latents, {loader.d_model} dimensions")

        self._loaders[experiment_id] = loader
        return loader

    def clear_cache(self):
        """Clear all cached loaders."""
        self._loaders.clear()


# Global experiment manager
experiment_manager: ExperimentManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize experiment manager on startup."""
    global experiment_manager

    experiments_root = get_experiments_root()
    print(f"Experiments root: {experiments_root}")

    experiment_manager = ExperimentManager(experiments_root)
    experiments = experiment_manager.list_experiments()
    print(f"Found {len(experiments)} experiments: {experiments}")

    # Make manager available to routers
    app.state.experiment_manager = experiment_manager

    # For backwards compatibility, pre-load first experiment if only one exists
    if len(experiments) == 1:
        loader = await experiment_manager.get_loader(experiments[0])
        app.state.data_loader = loader
        app.state.default_experiment = experiments[0]
    else:
        app.state.data_loader = None
        app.state.default_experiment = experiments[0] if experiments else None

    yield

    # Cleanup
    print("Shutting down...")
    experiment_manager.clear_cache()


app = FastAPI(
    title="Feature Manifold Interface API",
    description="API for exploring SAE latent geometry",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(graph.router, prefix="/api", tags=["graph"])
app.include_router(cluster.router, prefix="/api", tags=["cluster"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "feature-manifold-interface"}


@app.get("/api/experiments")
async def list_experiments():
    """List available experiments."""
    manager = app.state.experiment_manager
    experiments = manager.list_experiments()
    return {
        "experiments": experiments,
        "default": app.state.default_experiment,
    }


@app.get("/api/metadata")
async def get_metadata(experiment_id: str | None = None):
    """Get dataset metadata for an experiment."""
    manager = app.state.experiment_manager

    # Use default experiment if not specified
    if experiment_id is None:
        experiment_id = app.state.default_experiment
        if experiment_id is None:
            raise HTTPException(
                status_code=400,
                detail="No experiment specified and no default available"
            )

    loader = await manager.get_loader(experiment_id)
    return {
        "experiment_id": experiment_id,
        "n_latents": loader.n_latents,
        "d_model": loader.d_model,
        "metadata": loader.metadata,
    }
