# Feature Manifold Interface

Interactive visualization interface for exploring LLM feature geometry, inspired by Engels et al. (2024) "Not all language model features are one-dimensionally linear".

## Project Overview

This tool allows researchers to explore the geometry of LLM features through:
1. **Latent browser** - View max-activating examples for individual latents
2. **UMAP explorer** - Click latents on 2D projection to browse examples
3. **Graph visualization** - Latent relationships (cosine/Jaccard similarity) *(planned)*
4. **Point cloud visualization** - Activation reconstructions using selected feature clusters *(planned)*

## Configuration

- **Model**: Gemma 3 27B pretrained (`google/gemma-3-27b-pt`)
- **SAEs**: Gemma Scope 2 residual stream SAEs (`gemma-scope-2-27b-pt-res`)
- **SAE Width**: 65k latents
- **SAE L0**: medium
- **Layer**: 31
- **Corpus**: pile_uncopyrighted (~10M tokens)

## Data Pipeline

### Prerequisites

```bash
cd /data/users/ericjm/feature-manifold-interface
source .venv/bin/activate
```

### Quick Start: Full Pipeline

Run the entire pipeline for an experiment with a single command:

```bash
# From config file (recommended)
python scripts/run_pipeline.py --config experiments/gemma3_27b_layer31.yaml

# Or with CLI arguments
python scripts/run_pipeline.py \
    --model google/gemma-3-27b-pt \
    --layer 31 \
    --sae-width 65k \
    --sae-l0 medium \
    --num-tokens 10000000 \
    --data-root /remote/ericjm/feature-manifold-interface/data \
    --device cuda:0

# Dry run to see what would be executed
python scripts/run_pipeline.py --config experiments/gemma3_27b_layer31.yaml --dry-run

# Skip harvesting if raw data already exists
python scripts/run_pipeline.py --config experiments/gemma3_27b_layer31.yaml --skip-harvest
```

The pipeline automatically:
1. Creates proper directory structure for the experiment
2. Runs each step in sequence
3. Tracks completion with marker files (skips completed steps on re-run)
4. Supports parallelization via `--n-workers`

### Running Multiple Experiments

To run experiments on different SAEs/layers in parallel across GPUs:

```bash
# Terminal 1: Layer 31 on GPU 0
python scripts/run_pipeline.py --config experiments/gemma3_27b_layer31.yaml --device cuda:0

# Terminal 2: Layer 15 on GPU 1
python scripts/run_pipeline.py --config experiments/gemma3_27b_layer15.yaml --device cuda:1

# Terminal 3: 262k SAE on GPU 2
python scripts/run_pipeline.py --config experiments/gemma3_27b_layer31_262k.yaml --device cuda:2
```

### Data Organization (Multi-Experiment)

Data is organized by experiment ID (`{model}_{layer}_{sae_width}_{sae_l0}`):

```
data/
├── experiments/
│   ├── gemma-3-27b-pt_layer31_65k_medium/
│   │   ├── raw_activations/
│   │   ├── latents/
│   │   ├── graph/
│   │   ├── decoder.npy
│   │   └── metadata.json
│   └── gemma-3-27b-pt_layer15_65k_medium/
│       └── ...
└── visualizer_data/
    ├── gemma-3-27b-pt_layer31_65k_medium/
    │   ├── index.json
    │   ├── positions.json
    │   └── latents/
    └── ...
```

### Manual Pipeline Steps

If you need to run individual steps:

#### Step 1: Harvest Activations

```bash
python scripts/harvest_activations_v2.py \
    --layer 31 \
    --sae-width 65k \
    --sae-l0 medium \
    --output-dir /path/to/data/raw_activations \
    --num-tokens 10000000 \
    --seq-len 1024 \
    --device cuda:0
```

**Output:** `raw_activations/shard_*.npz`, `decoder.npy`, `metadata.json`, `latent_labels.json`

#### Step 2: Build Per-Latent Indices

```bash
python scripts/build_indices.py \
    --input-dir /path/to/data/raw_activations \
    --output-dir data \
    --n-workers 8  # Parallel shard loading
```

**Output:** `data/latents/*.npz`, `data/decoder.npy`, `data/metadata.json`

#### Step 3: Compute Graph Edges & UMAP Positions

```bash
# Full computation
python scripts/compute_edges.py \
    --data-dir data \
    --top-k 100

# Quick mode (UMAP + cosine similarity only)
python scripts/compute_edges.py \
    --data-dir data \
    --top-k 100 \
    --skip-coactivation \
    --skip-jaccard
```

**Output:** `data/graph/positions.npy`, `decoder_similarity.npz`, `jaccard_similarity.npz`, `coactivation.npz`

#### Step 4: Extract Top Activations for Visualizer

```bash
# Full extraction with parallel processing
python scripts/extract_top_activations.py \
    --raw-dir /path/to/data/raw_activations \
    --output-dir visualizer/data \
    --top-k 50 \
    --context-size 10 \
    --sample-percentiles \
    --n-workers 8

# Quick test (~10% of data)
python scripts/extract_top_activations.py \
    --raw-dir /path/to/data/raw_activations \
    --output-dir visualizer/data \
    --top-k 50 \
    --max-shards 64
```

**Output:** `visualizer/data/index.json`, `visualizer/data/latents/*.json`

#### Step 5: Export UMAP Positions for Web

```bash
python scripts/export_umap_json.py \
    --positions-file data/graph/positions.npy \
    --output visualizer/data/positions.json
```

**Output:** `visualizer/data/positions.json`

#### Step 6: Serve Visualizer

```bash
cd visualizer
python -m http.server 8080
```

- http://localhost:8080 - Simple latent browser
- http://localhost:8080/umap.html - UMAP explorer

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `run_pipeline.py` | **Orchestrate full pipeline** from config file |
| `harvest_activations_v2.py` | Run model + SAE on corpus, collect activations |
| `build_indices.py` | Create per-latent indices from harvest shards (parallel) |
| `compute_edges.py` | Compute UMAP positions + similarity edges |
| `extract_top_activations.py` | Extract top-k examples per latent for visualizer (parallel) |
| `export_umap_json.py` | Convert positions.npy to JSON for web |
| `sanity_check_sae.py` | Measure SAE variance explained and L0 |
| `test_sae_loading.py` | Validate SAE loading works |

## Visualizers

### Simple Browser (`visualizer/index.html`)

Browse latents by ID with max-activating examples:
- Arrow keys (← →) to navigate
- Press `g` to jump to specific latent
- Toggle Grid/List view
- Histogram of activation distribution
- Token highlighting by activation strength

### UMAP Explorer (`visualizer/umap.html`)

Click latents on 2D projection:
- **Left panel**: UMAP scatter plot (pan/zoom/click)
- **Right panel**: Max-activating examples for selected latent
- Press `r` to reset view

## Data Format

```
data/
├── raw_activations/            # Harvest output
│   ├── shard_*.npz            # {token_indices, latent_indices, activations, token_ids, doc_ids, positions}
│   ├── decoder.npy            # (n_latents, d_model)
│   ├── latent_labels.json     # Most frequent token per latent
│   └── metadata.json
├── latents/                    # Per-latent indices (from build_indices.py)
│   └── *.npz                  # {token_indices, activations}
├── graph/                      # Graph data (from compute_edges.py)
│   ├── positions.npy          # (n_latents, 2) UMAP positions
│   ├── decoder_similarity.npz # Sparse cosine similarity (top-k per latent)
│   ├── jaccard_similarity.npz # Sparse Jaccard similarity
│   ├── coactivation.npz       # Sparse co-activation counts
│   └── metadata.json
├── decoder.npy
└── metadata.json

visualizer/data/                # Visualizer data
├── index.json                 # Latent metadata
├── positions.json             # UMAP positions for web
└── latents/
    └── *.json                 # Per-latent examples
```

## Planned: Full Graph Interface

Architecture for the full interactive application (not yet built):

```
┌─────────────────────┐     ┌─────────────────────┐
│   React Frontend    │────▶│   FastAPI Backend   │
│   (sigma.js graph)  │     │   (Python)          │
│   (Plotly.js plots) │◀────│                     │
└─────────────────────┘     └──────────┬──────────┘
                                       │
                            ┌──────────▼──────────┐
                            │   Data Files        │
                            │   - latents/*.npz   │
                            │   - decoder.npy     │
                            │   - graph/*.npz     │
                            └─────────────────────┘
```

### API Endpoints (Planned)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/graph` | GET | Returns node positions |
| `/api/edges` | GET | Returns edges (type=cosine\|jaccard) |
| `/api/latent/{id}` | GET | Returns latent's token data |
| `/api/cluster/cloud` | POST | Returns point cloud for cluster |
| `/api/cluster/pca` | POST | Returns PCA projection |

### Point Cloud Generation (Incremental)

When a user clicks latents to form a cluster, compute a point cloud of reconstructed activations:

**Add latent L:**
```python
for token_idx, activation in latent_L_data:
    if token_idx in cloud:
        cloud[token_idx].vector += activation * decoder[L]
    else:
        cloud[token_idx] = activation * decoder[L]
```

**Remove latent L:**
```python
for token_idx in cloud:
    if L active on token_idx:
        cloud[token_idx].vector -= activation * decoder[L]
        if no latents left:
            remove point
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Graph library | sigma.js | WebGL rendering for 65k+ nodes |
| Plotting | Plotly.js | Interactive 3D, good React integration |
| Backend | FastAPI | Async, fast, good numpy integration |
| Point updates | Incremental | Fast cluster modifications |
| SAE width | 65k | Balance of coverage and performance |
| Edge storage | Sparse top-k | 65k×65k dense = 17GB; top-100 per latent = 6.5M edges |

## References

- Engels et al. (2024) "Not all language model features are one-dimensionally linear"
- Gemma Scope 2: https://huggingface.co/google/gemma-scope-2
- SAELens: https://github.com/jbloomAus/SAELens
