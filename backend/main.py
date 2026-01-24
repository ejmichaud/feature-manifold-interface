"""
FastAPI backend for Feature Manifold Interface.

Provides API endpoints for:
- Graph data (positions, edges)
- Latent data (token indices, activations)
- Cluster operations (point cloud generation, PCA)
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import graph, cluster
from services.data_loader import DataLoader


# Global data loader instance
data_loader: DataLoader | None = None


def get_data_dir() -> Path:
    """Get data directory from environment or default."""
    import os
    data_dir = os.environ.get("DATA_DIR", "../data")
    return Path(data_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data on startup, cleanup on shutdown."""
    global data_loader

    data_dir = get_data_dir()
    print(f"Loading data from {data_dir}...")

    data_loader = DataLoader(data_dir)
    await data_loader.load()

    print(f"Data loaded: {data_loader.n_latents} latents, {data_loader.d_model} dimensions")

    # Make data_loader available to routers
    app.state.data_loader = data_loader

    yield

    # Cleanup
    print("Shutting down...")


app = FastAPI(
    title="Feature Manifold Interface API",
    description="API for exploring SAE latent geometry",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
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


@app.get("/api/metadata")
async def get_metadata():
    """Get dataset metadata."""
    dl = app.state.data_loader
    return {
        "n_latents": dl.n_latents,
        "d_model": dl.d_model,
        "metadata": dl.metadata,
    }
