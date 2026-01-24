# SAE Latent Visualizer

Minimal, clean interfaces for exploring SAE latent activations.

## Visualizers

| File | Description |
|------|-------------|
| `index.html` | Simple latent browser - navigate by ID, view max-activating examples |
| `umap.html` | UMAP explorer - click latents on 2D projection, view examples on right |

## Full Pipeline

### Prerequisites

Activate the project environment:
```bash
cd /data/users/ericjm/feature-manifold-interface
source .venv/bin/activate
```

### Step 1: Harvest Activations

Run the model and SAE on a corpus to collect activations:

```bash
python scripts/harvest_activations.py \
    --layer 31 \
    --sae-width 65k \
    --sae-l0 medium \
    --output-dir /path/to/data/raw_activations \
    --num-tokens 10000000
```

**Output:** `raw_activations/shard_*.npz`, `metadata.json`, `decoder.npy`

### Step 2: Build Per-Latent Indices

Create indexed files for each latent (required for compute_edges.py):

```bash
python scripts/build_indices.py \
    --input-dir /path/to/data/raw_activations \
    --output-dir data
```

**Output:** `data/latents/*.npz`, `data/metadata.json`, `data/decoder.npy`

### Step 3: Compute Graph Edges & UMAP Positions

Compute latent similarities and 2D layout:

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

**Output:** `data/graph/positions.npy`, `decoder_similarity.npz`, etc.

### Step 4: Extract Top Activations for Visualizer

Process harvest data into visualization-friendly format:

```bash
# Full extraction
python scripts/extract_top_activations.py \
    --raw-dir /path/to/data/raw_activations \
    --output-dir visualizer/data \
    --top-k 50 \
    --context-size 10 \
    --skip-context-activations

# Quick test (~10% of data)
python scripts/extract_top_activations.py \
    --raw-dir /path/to/data/raw_activations \
    --output-dir visualizer/data \
    --top-k 20 \
    --context-size 10 \
    --skip-context-activations \
    --max-shards 64
```

**Output:** `visualizer/data/index.json`, `visualizer/data/latents/*.json`

### Step 5: Export UMAP Positions for Web

Convert numpy positions to JSON:

```bash
python scripts/export_umap_json.py \
    --positions-file data/graph/positions.npy \
    --output visualizer/data/positions.json
```

**Output:** `visualizer/data/positions.json`

### Step 6: Serve the Visualizer

```bash
cd visualizer
python -m http.server 8080
```

Then open:
- http://localhost:8080 - Simple latent browser
- http://localhost:8080/umap.html - UMAP explorer

## Quick Start (Minimal)

If you just want to browse latents without UMAP:

```bash
# Extract examples (quick test with limited shards)
python scripts/extract_top_activations.py \
    --raw-dir /path/to/raw_activations \
    --output-dir visualizer/data \
    --top-k 50 \
    --context-size 10 \
    --skip-context-activations \
    --max-shards 64

# Serve
cd visualizer && python -m http.server 8080
```

Open http://localhost:8080

## Usage

### Simple Browser (index.html)

- **Enter latent ID** in the input box and press Enter
- **Navigate**: Arrow keys (← →) or Previous/Next buttons
- **Jump to latent**: Press `g` to focus input
- **View modes**: Toggle between Grid and List views
- **Histogram**: Shows distribution of top-K activation values
- **Hover**: See exact activation value on highlighted tokens

### UMAP Explorer (umap.html)

**Left panel (UMAP):**
- **Click** a point to select that latent
- **Drag** to pan the view
- **Scroll** to zoom in/out
- **Hover** to see latent ID and label
- Press `r` to reset view

**Right panel (Examples):**
- Same controls as the simple browser
- Arrow keys navigate between latents
- Press `g` to jump to specific latent ID

## Data Format

### visualizer/data/

```
visualizer/data/
├── index.json              # Metadata for all latents
├── positions.json          # UMAP 2D coordinates (for umap.html)
└── latents/
    ├── 00000.json          # Full data for latent 0
    ├── 00001.json          # Full data for latent 1
    └── ...
```

### index.json

```json
{
  "n_latents": 65536,
  "top_k": 50,
  "context_size": 10,
  "latents": {
    "0": {"label": "brown", "total_firings": 1247, "n_examples": 50},
    "1": {"label": "the", "total_firings": 5832, "n_examples": 50}
  }
}
```

### positions.json

```json
{
  "n_latents": 65536,
  "positions": [[0.123, 0.456], [0.789, 0.012], ...]
}
```

### Individual latent files (latents/00000.json)

```json
{
  "latent_id": 0,
  "label": "brown",
  "total_firings": 1247,
  "top_examples": [
    {
      "activation": 12.5,
      "context": "The quick brown fox",
      "tokens": ["The", " quick", " brown", " fox"],
      "token_activations": [0.0, 0.0, 12.5, 0.0],
      "center_idx": 2
    }
  ]
}
```

## Performance Notes

- Index loads once (~2-3MB for 65k latents)
- UMAP positions load once (~1-2MB for 65k latents)
- Individual latent files load on demand (~10-50KB each)
- 50 most recent latents are cached in browser memory
- UMAP canvas handles 65k+ points with pan/zoom

## Design

- **Background**: Light gray (#fafafa)
- **Cards**: White
- **Text**: Dark gray (#1a1a1a)
- **Accent**: Orange-red (#ff4500)
- **Activation highlighting**: 9 intensity levels
- **Font**: SF Mono / Menlo / Monaco / Consolas
