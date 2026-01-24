"""
Extract top activating examples for each latent from the raw harvest data.

This creates a visualization-friendly format with actual text context.
Outputs separate files per latent for lazy loading.

Features:
- Merges overlapping context windows to avoid duplicate text
- Samples from multiple percentile ranges (not just top-k)
- Computes full activation histogram for each latent
- Parallel processing for speed

Output structure:
    output_dir/
        index.json          # Metadata for all latents (for quick navigation)
        latents/
            00000.json      # Full data for latent 0
            00001.json      # Full data for latent 1
            ...

Each latent file contains:
{
    "latent_id": 0,
    "label": "brown",
    "total_firings": 1247,
    "activation_stats": {"min": 0.1, "max": 15.3, "mean": 2.1, "p10": 0.5, "p50": 1.8, "p90": 5.2},
    "histogram": {"bins": [...], "counts": [...]},
    "examples": {
        "top": [...],      // 90-100 percentile
        "high": [...],     // 70-90 percentile
        "medium": [...],   // 40-70 percentile
        "low": [...]       // 10-40 percentile
    }
}

Usage:
    # With percentile sampling (recommended):
    python extract_top_activations.py \\
        --raw-dir /path/to/raw_activations \\
        --output-dir visualizer/data \\
        --top-k 20 \\
        --sample-percentiles \\
        --context-size 10 \\
        --n-workers 8

    # Quick test with limited shards:
    python extract_top_activations.py \\
        --raw-dir /path/to/raw_activations \\
        --output-dir visualizer/data \\
        --top-k 10 \\
        --sample-percentiles \\
        --max-shards 64
"""

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def load_raw_activations(raw_dir: Path, n_latents: int, max_shards: int = None):
    """
    Load all activations from shards and organize by latent.

    Returns: dict[latent_id] -> list[dict with activation, token_idx, token_id, doc_id, position]
    """
    print("Loading raw activation shards...")
    latent_activations = defaultdict(list)

    shard_files = sorted(raw_dir.glob("shard_*.npz"))
    if max_shards is not None:
        shard_files = shard_files[:max_shards]
        print(f"  (limited to {max_shards} shards)")

    for shard_file in tqdm(shard_files, desc="Loading shards"):
        data = np.load(shard_file)

        latent_indices = data["latent_indices"]
        activations = data["activations"]
        token_indices = data["token_indices"]
        token_ids = data["token_ids"]
        doc_ids = data["doc_ids"]
        positions = data["positions"]

        # Group by latent
        for lat_idx, act, tok_idx, tok_id, doc_id, pos in zip(
            latent_indices, activations, token_indices, token_ids, doc_ids, positions
        ):
            latent_activations[int(lat_idx)].append({
                "activation": float(act),
                "token_idx": int(tok_idx),
                "token_id": int(tok_id),
                "doc_id": int(doc_id),
                "position": int(pos),
            })

    return latent_activations


def compute_activation_stats(activations: list) -> dict:
    """Compute statistics over all activations for a latent."""
    if not activations:
        return {"min": 0, "max": 0, "mean": 0, "p10": 0, "p50": 0, "p90": 0}

    acts = np.array([a["activation"] for a in activations])
    return {
        "min": float(np.min(acts)),
        "max": float(np.max(acts)),
        "mean": float(np.mean(acts)),
        "p10": float(np.percentile(acts, 10)),
        "p50": float(np.percentile(acts, 50)),
        "p90": float(np.percentile(acts, 90)),
    }


def compute_histogram(activations: list, n_bins: int = 30) -> dict:
    """Compute histogram over all activations for a latent."""
    if not activations:
        return {"bins": [], "counts": []}

    acts = np.array([a["activation"] for a in activations])
    counts, bin_edges = np.histogram(acts, bins=n_bins)

    return {
        "bins": [float(b) for b in bin_edges],
        "counts": [int(c) for c in counts],
    }


def sample_by_percentiles(
    activations: list,
    top_k: int = 20,
    high_k: int = 10,
    medium_k: int = 10,
    low_k: int = 10,
) -> dict:
    """
    Sample activations from different percentile ranges.

    Returns dict with keys: top, high, medium, low
    Each containing a list of activation dicts.
    """
    if not activations:
        return {"top": [], "high": [], "medium": [], "low": []}

    # Sort by activation descending
    sorted_acts = sorted(activations, key=lambda x: -x["activation"])
    n = len(sorted_acts)

    # Define percentile boundaries (indices)
    p90_idx = int(n * 0.10)  # top 10% = indices 0 to p90_idx
    p70_idx = int(n * 0.30)  # 70-90% = indices p90_idx to p70_idx
    p40_idx = int(n * 0.60)  # 40-70% = indices p70_idx to p40_idx
    p10_idx = int(n * 0.90)  # 10-40% = indices p40_idx to p10_idx

    # Extract ranges
    top_range = sorted_acts[:p90_idx] if p90_idx > 0 else sorted_acts[:1]
    high_range = sorted_acts[p90_idx:p70_idx] if p70_idx > p90_idx else []
    medium_range = sorted_acts[p70_idx:p40_idx] if p40_idx > p70_idx else []
    low_range = sorted_acts[p40_idx:p10_idx] if p10_idx > p40_idx else []

    # Sample from each range
    def sample_from_range(range_list, k):
        if len(range_list) <= k:
            return range_list
        # For top, take the top k; for others, random sample
        return random.sample(range_list, k)

    return {
        "top": sorted_acts[:top_k],  # Always take actual top-k for top
        "high": sample_from_range(high_range, high_k),
        "medium": sample_from_range(medium_range, medium_k),
        "low": sample_from_range(low_range, low_k),
    }


def merge_overlapping_contexts(activations: list, context_size: int) -> list:
    """
    Merge activations that would have overlapping context windows.

    Returns list of merged groups, each containing:
    {
        "doc_id": int,
        "start": int,  # start position of merged window
        "end": int,    # end position of merged window
        "firings": [activation_dicts...]  # all firings in this window
    }
    """
    if not activations:
        return []

    # Sort by (doc_id, position)
    sorted_acts = sorted(activations, key=lambda x: (x["doc_id"], x["position"]))

    merged = []
    current_group = None

    for act in sorted_acts:
        if current_group is None:
            current_group = {
                "doc_id": act["doc_id"],
                "start": act["position"] - context_size,
                "end": act["position"] + context_size,
                "firings": [act],
            }
        elif (act["doc_id"] == current_group["doc_id"] and
              act["position"] <= current_group["end"] + 1):
            # Overlapping or adjacent - extend the window and add firing
            current_group["end"] = max(current_group["end"], act["position"] + context_size)
            current_group["firings"].append(act)
        else:
            # Non-overlapping - save current group and start new one
            merged.append(current_group)
            current_group = {
                "doc_id": act["doc_id"],
                "start": act["position"] - context_size,
                "end": act["position"] + context_size,
                "firings": [act],
            }

    if current_group:
        merged.append(current_group)

    return merged


def build_token_position_map(raw_dir: Path, max_shards: int = None):
    """
    Build a map from (doc_id, position) -> token_id by scanning all shards.
    """
    print("Building token position map...")
    token_map = {}

    shard_files = sorted(raw_dir.glob("shard_*.npz"))
    if max_shards is not None:
        shard_files = shard_files[:max_shards]

    for shard_file in tqdm(shard_files, desc="Mapping tokens"):
        data = np.load(shard_file)

        doc_ids = data["doc_ids"]
        positions = data["positions"]
        token_ids = data["token_ids"]

        for doc_id, pos, tok_id in zip(doc_ids, positions, token_ids):
            key = (int(doc_id), int(pos))
            if key not in token_map:
                token_map[key] = int(tok_id)

    print(f"  Mapped {len(token_map)} token positions")
    return token_map


def build_context_for_merged_group(
    merged_group: dict,
    latent_id: int,
    token_position_map: dict,
    tokenizer,
) -> dict:
    """
    Build a context window for a merged group of activations.

    Returns an example dict with all firing tokens highlighted.
    """
    doc_id = merged_group["doc_id"]
    start_pos = merged_group["start"]
    end_pos = merged_group["end"]
    firings = merged_group["firings"]

    # Build position -> activation map for this latent
    position_to_activation = {f["position"]: f["activation"] for f in firings}

    # Get all tokens in the window
    context_token_ids = []
    context_activations = []

    for pos in range(start_pos, end_pos + 1):
        key = (doc_id, pos)
        if key in token_position_map:
            token_id = token_position_map[key]
            context_token_ids.append(token_id)
            # Get activation if this position fired
            activation = position_to_activation.get(pos, 0.0)
            context_activations.append(activation)
        else:
            # Missing token - show placeholder
            context_token_ids.append(None)
            context_activations.append(0.0)

    # Decode tokens
    decoded_tokens = []
    for tok_id in context_token_ids:
        if tok_id is not None:
            decoded_tokens.append(tokenizer.decode([tok_id]))
        else:
            decoded_tokens.append("⋯")

    # Build context string
    context_str = "".join(decoded_tokens)

    # Find max activation and firing positions
    max_activation = max(f["activation"] for f in firings)
    firing_positions = [
        pos - start_pos for pos in position_to_activation.keys()
        if 0 <= pos - start_pos < len(decoded_tokens)
    ]

    return {
        "max_activation": max_activation,
        "context": context_str,
        "tokens": decoded_tokens,
        "token_activations": context_activations,
        "firing_positions": firing_positions,
        "n_firings": len(firings),
    }


def build_examples_for_percentile_group(
    activations: list,
    latent_id: int,
    token_position_map: dict,
    tokenizer,
    context_size: int,
) -> list:
    """
    Build examples for a group of activations, merging overlapping contexts.
    """
    if not activations:
        return []

    # Merge overlapping contexts
    merged_groups = merge_overlapping_contexts(activations, context_size)

    # Build context for each merged group
    examples = []
    for group in merged_groups:
        example = build_context_for_merged_group(
            group, latent_id, token_position_map, tokenizer
        )
        examples.append(example)

    # Sort by max_activation descending
    examples.sort(key=lambda x: -x["max_activation"])

    return examples


def load_labels(raw_dir: Path):
    """Load latent labels if available."""
    labels_path = raw_dir / "latent_labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            return json.load(f)
    return {}


def process_single_latent(
    latent_id: int,
    activations: list,
    label_info: dict,
    token_position_map: dict,
    tokenizer_name: str,
    latents_dir: Path,
    top_k: int,
    high_k: int,
    medium_k: int,
    low_k: int,
    context_size: int,
    histogram_bins: int,
    sample_percentiles: bool,
) -> dict:
    """
    Process a single latent and save its data file.

    Returns index entry for this latent.
    """
    # Load tokenizer in worker process
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    label = label_info.get("token")
    total_firings = len(activations)

    # Compute stats and histogram over all activations
    activation_stats = compute_activation_stats(activations)
    histogram = compute_histogram(activations, histogram_bins)

    # Sample activations by percentile
    if sample_percentiles:
        sampled = sample_by_percentiles(
            activations,
            top_k=top_k,
            high_k=high_k,
            medium_k=medium_k,
            low_k=low_k,
        )
    else:
        # Just top-k
        sorted_acts = sorted(activations, key=lambda x: -x["activation"])
        sampled = {
            "top": sorted_acts[:top_k],
            "high": [],
            "medium": [],
            "low": [],
        }

    # Build examples for each percentile group (with context merging)
    examples = {}
    for group_name, group_acts in sampled.items():
        examples[group_name] = build_examples_for_percentile_group(
            group_acts,
            latent_id,
            token_position_map,
            tokenizer,
            context_size,
        )

    # Save individual latent file
    latent_data = {
        "latent_id": latent_id,
        "label": label,
        "total_firings": total_firings,
        "activation_stats": activation_stats,
        "histogram": histogram,
        "examples": examples,
    }

    latent_path = latents_dir / f"{latent_id:05d}.json"
    with open(latent_path, "w") as f:
        json.dump(latent_data, f)

    # Return index entry
    n_examples = sum(len(examples[g]) for g in examples)
    return {
        "latent_id": latent_id,
        "label": label,
        "total_firings": total_firings,
        "n_examples": n_examples,
    }


def process_latent_batch(
    batch: list,  # list of (latent_id, activations, label_info)
    token_position_map: dict,
    tokenizer_name: str,
    latents_dir: Path,
    top_k: int,
    high_k: int,
    medium_k: int,
    low_k: int,
    context_size: int,
    histogram_bins: int,
    sample_percentiles: bool,
) -> list:
    """Process a batch of latents in a worker process."""
    # Load tokenizer once per worker
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    results = []
    for latent_id, activations, label_info in batch:
        label = label_info.get("token")
        total_firings = len(activations)

        # Compute stats and histogram
        activation_stats = compute_activation_stats(activations)
        histogram = compute_histogram(activations, histogram_bins)

        # Sample activations by percentile
        if sample_percentiles:
            sampled = sample_by_percentiles(
                activations,
                top_k=top_k,
                high_k=high_k,
                medium_k=medium_k,
                low_k=low_k,
            )
        else:
            sorted_acts = sorted(activations, key=lambda x: -x["activation"])
            sampled = {"top": sorted_acts[:top_k], "high": [], "medium": [], "low": []}

        # Build examples
        examples = {}
        for group_name, group_acts in sampled.items():
            examples[group_name] = build_examples_for_percentile_group(
                group_acts, latent_id, token_position_map, tokenizer, context_size
            )

        # Save file
        latent_data = {
            "latent_id": latent_id,
            "label": label,
            "total_firings": total_firings,
            "activation_stats": activation_stats,
            "histogram": histogram,
            "examples": examples,
        }

        latent_path = latents_dir / f"{latent_id:05d}.json"
        with open(latent_path, "w") as f:
            json.dump(latent_data, f)

        n_examples = sum(len(examples[g]) for g in examples)
        results.append({
            "latent_id": latent_id,
            "label": label,
            "total_firings": total_firings,
            "n_examples": n_examples,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract top activations for visualization")
    parser.add_argument("--raw-dir", type=str, required=True,
                        help="Raw activations directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for latent files")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of examples from top percentile (90-100%%)")
    parser.add_argument("--sample-percentiles", action="store_true",
                        help="Also sample from lower percentile ranges (high, medium, low)")
    parser.add_argument("--high-k", type=int, default=10,
                        help="Number of examples from high percentile (70-90%%)")
    parser.add_argument("--medium-k", type=int, default=10,
                        help="Number of examples from medium percentile (40-70%%)")
    parser.add_argument("--low-k", type=int, default=10,
                        help="Number of examples from low percentile (10-40%%)")
    parser.add_argument("--context-size", type=int, default=10,
                        help="Context window size (tokens on each side)")
    parser.add_argument("--histogram-bins", type=int, default=30,
                        help="Number of histogram bins")
    parser.add_argument("--model-name", type=str, default="google/gemma-3-27b-pt",
                        help="Model name for tokenizer")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Only process first N shards (for quick testing)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Number of parallel workers (default: 1, sequential)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Latents per batch for parallel processing")

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    latents_dir = output_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata to get n_latents
    with open(raw_dir / "metadata.json") as f:
        metadata = json.load(f)
    n_latents = metadata["n_latents"]

    print(f"Processing {n_latents} latents...")

    # Build token position map (needed by all workers)
    token_position_map = build_token_position_map(raw_dir, max_shards=args.max_shards)

    # Load all activations
    latent_activations = load_raw_activations(raw_dir, n_latents, max_shards=args.max_shards)

    # Load labels
    labels = load_labels(raw_dir)

    # Build index
    print("Building context windows and saving latent files...")
    index_data = {
        "n_latents": n_latents,
        "top_k": args.top_k,
        "sample_percentiles": args.sample_percentiles,
        "context_size": args.context_size,
        "latents": {}
    }

    if args.n_workers > 1:
        # Parallel processing
        print(f"Using {args.n_workers} workers with batch size {args.batch_size}")

        # Prepare batches
        batches = []
        current_batch = []
        for latent_id in range(n_latents):
            activations = latent_activations.get(latent_id, [])
            label_info = labels.get(str(latent_id), {})
            current_batch.append((latent_id, activations, label_info))

            if len(current_batch) >= args.batch_size:
                batches.append(current_batch)
                current_batch = []

        if current_batch:
            batches.append(current_batch)

        print(f"Processing {len(batches)} batches...")

        # Process batches in parallel
        process_fn = partial(
            process_latent_batch,
            token_position_map=token_position_map,
            tokenizer_name=args.model_name,
            latents_dir=latents_dir,
            top_k=args.top_k,
            high_k=args.high_k,
            medium_k=args.medium_k,
            low_k=args.low_k,
            context_size=args.context_size,
            histogram_bins=args.histogram_bins,
            sample_percentiles=args.sample_percentiles,
        )

        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {executor.submit(process_fn, batch): i for i, batch in enumerate(batches)}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                batch_results = future.result()
                for result in batch_results:
                    index_data["latents"][str(result["latent_id"])] = {
                        "label": result["label"],
                        "total_firings": result["total_firings"],
                        "n_examples": result["n_examples"],
                    }
    else:
        # Sequential processing (original behavior)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        for latent_id in tqdm(range(n_latents), desc="Saving latents"):
            activations = latent_activations.get(latent_id, [])

            # Get label info
            label_info = labels.get(str(latent_id), {})
            label = label_info.get("token")
            total_firings = len(activations)

            # Compute stats and histogram over all activations
            activation_stats = compute_activation_stats(activations)
            histogram = compute_histogram(activations, args.histogram_bins)

            # Sample activations by percentile
            if args.sample_percentiles:
                sampled = sample_by_percentiles(
                    activations,
                    top_k=args.top_k,
                    high_k=args.high_k,
                    medium_k=args.medium_k,
                    low_k=args.low_k,
                )
            else:
                # Just top-k
                sorted_acts = sorted(activations, key=lambda x: -x["activation"])
                sampled = {
                    "top": sorted_acts[:args.top_k],
                    "high": [],
                    "medium": [],
                    "low": [],
                }

            # Build examples for each percentile group (with context merging)
            examples = {}
            for group_name, group_acts in sampled.items():
                examples[group_name] = build_examples_for_percentile_group(
                    group_acts,
                    latent_id,
                    token_position_map,
                    tokenizer,
                    args.context_size,
                )

            # Save individual latent file
            latent_data = {
                "latent_id": latent_id,
                "label": label,
                "total_firings": total_firings,
                "activation_stats": activation_stats,
                "histogram": histogram,
                "examples": examples,
            }

            latent_path = latents_dir / f"{latent_id:05d}.json"
            with open(latent_path, "w") as f:
                json.dump(latent_data, f)

            # Add to index (metadata only)
            n_examples = sum(len(examples[g]) for g in examples)
            index_data["latents"][str(latent_id)] = {
                "label": label,
                "total_firings": total_firings,
                "n_examples": n_examples,
            }

    # Save index
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index_data, f)

    print(f"\nSaved visualization data to {output_dir}")
    print(f"  index.json: metadata for {n_latents} latents")
    print(f"  latents/: {n_latents} individual latent files")
    print(f"  Top examples: {args.top_k}")
    if args.sample_percentiles:
        print(f"  High/Medium/Low: {args.high_k}/{args.medium_k}/{args.low_k}")
    print(f"  Context: ±{args.context_size} tokens")
    print(f"  Histogram bins: {args.histogram_bins}")
    if args.max_shards is not None:
        print(f"  Shards processed: {args.max_shards} (partial data)")


if __name__ == "__main__":
    main()
