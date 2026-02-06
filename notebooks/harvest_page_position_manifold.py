"""
Harvest 2D page-position manifold activations from a single model across all layers.

Extends the line-width manifold experiment to track both horizontal position
(character width within line) AND vertical position (line number), testing
whether LLMs build a 2D representation of page layout.

For each model, extracts residual stream activations from every layer in a single
forward pass using PyTorch forward hooks, groups token activations by (line_number,
line_width) on a 2D grid, computes mean activations per grid cell, runs PCA,
and saves per-layer results as JSON.

Usage:
    python notebooks/harvest_page_position_manifold.py --model gemma-2-2b --device cuda:0
    python notebooks/harvest_page_position_manifold.py --model gemma-3-27b --device cuda:0 --data-file notebooks/LineBreakManifold/linebreak_data/linebreak_width_150_expanded.txt
"""

import argparse
import json
import gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from tqdm import tqdm


# ── Model configurations ──────────────────────────────────────────────
MODEL_CONFIGS = {
    "gemma-2-2b": {
        "hf_name": "google/gemma-2-2b",
        "family": "gemma2",
        "layer_path": "model.layers",
    },
    "gemma-2-9b": {
        "hf_name": "google/gemma-2-9b",
        "family": "gemma2",
        "layer_path": "model.layers",
    },
    "gemma-2-27b": {
        "hf_name": "google/gemma-2-27b",
        "family": "gemma2",
        "layer_path": "model.layers",
    },
    "gemma-3-4b": {
        "hf_name": "google/gemma-3-4b-pt",
        "family": "gemma3",
        "layer_path": "model.language_model.layers",
    },
    "gemma-3-12b": {
        "hf_name": "google/gemma-3-12b-pt",
        "family": "gemma3",
        "layer_path": "model.language_model.layers",
    },
    "gemma-3-27b": {
        "hf_name": "google/gemma-3-27b-pt",
        "family": "gemma3",
        "layer_path": "model.language_model.layers",
    },
    "qwen3-4b": {
        "hf_name": "Qwen/Qwen3-4B-Base",
        "family": "qwen3",
        "layer_path": "model.layers",
    },
    "qwen3-8b": {
        "hf_name": "Qwen/Qwen3-8B-Base",
        "family": "qwen3",
        "layer_path": "model.layers",
    },
    "qwen3-14b": {
        "hf_name": "Qwen/Qwen3-14B-Base",
        "family": "qwen3",
        "layer_path": "model.layers",
    },
}


def get_layers_module(model, layer_path):
    """Navigate dotted path to get the layers module."""
    module = model
    for attr in layer_path.split("."):
        module = getattr(module, attr)
    return module


def split_into_documents(text, seq_len, tokenizer):
    """Split text into documents and tokenize, truncating to seq_len tokens."""
    raw_documents = text.split("\n\n=====\n\n")
    documents = []
    for doc in raw_documents:
        doc = doc.strip()
        if not doc:
            continue
        tokens = tokenizer.encode(doc, add_special_tokens=False)
        if len(tokens) > 10:
            documents.append(tokens[:seq_len])
    return documents


def find_newline_positions(token_ids, newline_token_id):
    """Find all positions of newline tokens."""
    return [i for i, tok_id in enumerate(token_ids) if tok_id == newline_token_id]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harvest 2D page-position manifold activations")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()),
                        help="Model key to process")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--output-dir", default="notebooks/page_position_manifold",
                        help="Output directory for results")
    parser.add_argument("--data-file", default="notebooks/LineBreakManifold/linebreak_data/linebreak_width_150_expanded.txt",
                        help="Path to linebreak text file")
    parser.add_argument("--seq-len", type=int, default=2048, help="Max sequence length per document")
    parser.add_argument("--min-width", type=int, default=10, help="Minimum width to include in PCA")
    parser.add_argument("--max-width", type=int, default=150, help="Maximum width to track")
    parser.add_argument("--min-line", type=int, default=1, help="Minimum line number for PCA")
    parser.add_argument("--max-line", type=int, default=30, help="Maximum line number to track")
    parser.add_argument("--min-count", type=int, default=10, help="Minimum tokens per grid cell to include in PCA")
    parser.add_argument("--n-components", type=int, default=10, help="Number of PCA components")
    parser.add_argument("--multi-gpu", action="store_true", help="Use device_map='auto' for multi-GPU")
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    hf_name = config["hf_name"]
    layer_path = config["layer_path"]

    output_dir = Path(args.output_dir)
    # Use -- instead of / for directory name (e.g. google--gemma-2-2b)
    model_dir = output_dir / hf_name.replace("/", "--")
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Load text data ────────────────────────────────────────────────
    print(f"Loading text data from {args.data_file}...")
    text = Path(args.data_file).read_text()
    print(f"  {len(text):,} characters")

    # ── Load model and tokenizer ──────────────────────────────────────
    print(f"Loading model: {hf_name}...")
    dtype = torch.bfloat16
    if args.multi_gpu:
        device_map = "auto"
    else:
        device_map = {"": args.device}

    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        device_map=device_map,
        torch_dtype=dtype,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Tokenize documents ────────────────────────────────────────────
    print("Tokenizing documents...")
    documents = split_into_documents(text, args.seq_len, tokenizer)
    print(f"  {len(documents)} documents")

    newline_token_id = tokenizer.encode('\n', add_special_tokens=False)[0]
    print(f"  Newline token ID: {newline_token_id}")

    # ── Get layer module and count ────────────────────────────────────
    layers_module = get_layers_module(model, layer_path)
    n_layers = len(layers_module)
    d_model = getattr(model.config, 'hidden_size', None) or model.config.text_config.hidden_size
    print(f"  {n_layers} layers, d_model={d_model}")

    # ── Register forward hooks on all layers ──────────────────────────
    layer_activations = {}  # layer_idx -> (seq_len, d_model) numpy array

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden = output[0]
            layer_activations[layer_idx] = hidden[0].detach().float().cpu().numpy()
        return hook_fn

    hooks = []
    for i, layer in enumerate(layers_module):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    # ── Accumulate per-(line_number, width) activations across all layers ─
    # sum_per_grid[layer_idx][(line_number, width)] = numpy array of shape (d_model,)
    # count_per_grid[layer_idx][(line_number, width)] = int
    sum_per_grid = defaultdict(lambda: defaultdict(lambda: np.zeros(d_model, dtype=np.float64)))
    count_per_grid = defaultdict(lambda: defaultdict(int))

    print("Processing documents...")
    for token_ids in tqdm(documents):
        input_ids = torch.tensor([token_ids], device=args.device)
        newline_positions = set(find_newline_positions(token_ids, newline_token_id))

        # Forward pass — all hooks fire, populating layer_activations
        with torch.no_grad():
            model(input_ids)

        # Iterate tokens and accumulate activations by (line_number, line_width)
        line_width = 0
        line_number = 0
        for idx in range(len(token_ids)):
            if idx in newline_positions:
                line_width = 0
                line_number += 1
            else:
                line_width += len(tokenizer.decode(token_ids[idx]))
                if line_width <= args.max_width and line_number <= args.max_line:
                    for layer_idx in range(n_layers):
                        sum_per_grid[layer_idx][(line_number, line_width)] += layer_activations[layer_idx][idx]
                        count_per_grid[layer_idx][(line_number, line_width)] += 1

        layer_activations.clear()

    # Remove hooks
    for h in hooks:
        h.remove()

    # ── Compute mean activations and PCA per layer ────────────────────
    print("Computing PCA per layer...")
    for layer_idx in tqdm(range(n_layers)):
        # Filter grid cells by min/max constraints and min_count
        valid_cells = sorted(
            k for k in sum_per_grid[layer_idx].keys()
            if k[0] >= args.min_line
            and k[0] <= args.max_line
            and k[1] >= args.min_width
            and k[1] <= args.max_width
            and count_per_grid[layer_idx][k] >= args.min_count
        )

        if len(valid_cells) < args.n_components:
            print(f"  Layer {layer_idx}: only {len(valid_cells)} valid grid cells, skipping PCA")
            continue

        line_numbers = [k[0] for k in valid_cells]
        widths = [k[1] for k in valid_cells]

        mean_activations = np.array([
            sum_per_grid[layer_idx][k] / count_per_grid[layer_idx][k]
            for k in valid_cells
        ])

        n_comp = min(args.n_components, len(valid_cells))
        pca = PCA(n_components=n_comp)
        projections = pca.fit_transform(mean_activations)

        layer_data = {
            "line_numbers": line_numbers,
            "widths": widths,
            "pca_projections": projections.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        }

        layer_file = model_dir / f"layer_{layer_idx:02d}.json"
        with open(layer_file, "w") as f:
            json.dump(layer_data, f)

    # ── Save metadata ─────────────────────────────────────────────────
    metadata = {
        "model_key": args.model,
        "hf_name": hf_name,
        "family": config["family"],
        "n_layers": n_layers,
        "d_model": d_model,
        "n_documents": len(documents),
        "seq_len": args.seq_len,
        "min_width": args.min_width,
        "max_width": args.max_width,
        "min_line": args.min_line,
        "max_line": args.max_line,
        "min_count": args.min_count,
        "n_pca_components": args.n_components,
        "data_file": args.data_file,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Update manifest ───────────────────────────────────────────────
    manifest_file = output_dir / "manifest.json"
    if manifest_file.exists():
        with open(manifest_file) as f:
            manifest = json.load(f)
    else:
        manifest = {"models": []}

    # Remove existing entry for this model if present
    manifest["models"] = [m for m in manifest["models"] if m["model_key"] != args.model]
    manifest["models"].append({
        "model_key": args.model,
        "hf_name": hf_name,
        "family": config["family"],
        "n_layers": n_layers,
        "d_model": d_model,
        "dir_name": hf_name.replace("/", "--"),
    })
    manifest["models"].sort(key=lambda m: m["model_key"])

    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done! Results saved to {model_dir}")

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
