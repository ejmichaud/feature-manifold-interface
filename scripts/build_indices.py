"""
Build per-latent indices from raw activation shards.

This script takes the output of harvest_activations.py and reorganizes it
into per-latent storage for fast querying:

Input (from harvest):
    data/raw_activations/
        shard_00000.npz  # {token_indices, latent_indices, activations, doc_ids, positions}
        shard_00001.npz
        ...
        metadata.json
        decoder.npy

Output:
    data/latents/
        00000.npz  # {token_indices: sorted, activations: corresponding values}
        00001.npz
        ...
    data/corpus/
        token_map.npy  # (n_tokens, 3): [doc_id, position, token_id placeholder]
    data/
        decoder.npy  # copied from raw
        metadata.json  # updated with index info

Usage:
    python build_indices.py --input-dir ../data/raw_activations \
                            --output-dir ../data
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_shards(input_dir: Path) -> tuple[dict, dict]:
    """
    Load all shards and aggregate by latent.

    Returns:
        latent_data: dict[latent_id] -> list[(token_idx, activation)]
        token_data: dict[token_idx] -> (doc_id, position)
    """
    shard_files = sorted(input_dir.glob("shard_*.npz"))
    print(f"Found {len(shard_files)} shards")

    latent_data = defaultdict(list)
    token_data = {}

    for shard_path in tqdm(shard_files, desc="Loading shards"):
        shard = np.load(shard_path)

        token_indices = shard["token_indices"]
        latent_indices = shard["latent_indices"]
        activations = shard["activations"]
        doc_ids = shard["doc_ids"]
        positions = shard["positions"]

        for i in range(len(token_indices)):
            tok_idx = int(token_indices[i])
            lat_idx = int(latent_indices[i])
            act = float(activations[i])
            doc_id = int(doc_ids[i])
            pos = int(positions[i])

            latent_data[lat_idx].append((tok_idx, act))

            # Store token context info (may be duplicated but that's fine)
            if tok_idx not in token_data:
                token_data[tok_idx] = (doc_id, pos)

    return latent_data, token_data


def save_latent_indices(latent_data: dict, output_dir: Path, n_latents: int):
    """Save per-latent indices."""
    latents_dir = output_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_activations": 0,
        "latents_with_activations": 0,
        "activations_per_latent": [],
    }

    for latent_id in tqdm(range(n_latents), desc="Saving latent indices"):
        data = latent_data.get(latent_id, [])

        if len(data) == 0:
            # Empty latent - still save empty arrays for consistency
            token_indices = np.array([], dtype=np.int64)
            activations = np.array([], dtype=np.float16)
        else:
            # Sort by token index for binary search
            data.sort(key=lambda x: x[0])
            token_indices = np.array([d[0] for d in data], dtype=np.int64)
            activations = np.array([d[1] for d in data], dtype=np.float16)

            stats["latents_with_activations"] += 1
            stats["total_activations"] += len(data)
            stats["activations_per_latent"].append(len(data))

        # Save
        output_path = latents_dir / f"{latent_id:05d}.npz"
        np.savez_compressed(
            output_path,
            token_indices=token_indices,
            activations=activations,
        )

    # Compute stats
    if stats["activations_per_latent"]:
        stats["mean_activations_per_latent"] = np.mean(stats["activations_per_latent"])
        stats["median_activations_per_latent"] = np.median(stats["activations_per_latent"])
        stats["max_activations_per_latent"] = max(stats["activations_per_latent"])
    del stats["activations_per_latent"]  # Don't save full list

    return stats


def save_token_map(token_data: dict, output_dir: Path):
    """Save token-to-context mapping."""
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    if not token_data:
        print("Warning: No token data to save")
        return

    max_token_idx = max(token_data.keys())
    print(f"Max token index: {max_token_idx}")

    # Create dense array (could use sparse if needed)
    # Shape: (n_tokens, 2) -> [doc_id, position]
    token_map = np.zeros((max_token_idx + 1, 2), dtype=np.int32)

    for tok_idx, (doc_id, pos) in token_data.items():
        token_map[tok_idx, 0] = doc_id
        token_map[tok_idx, 1] = pos

    np.save(corpus_dir / "token_map.npy", token_map)
    print(f"Saved token map with {len(token_data)} entries")


def main():
    parser = argparse.ArgumentParser(description="Build per-latent indices")
    parser.add_argument("--input-dir", type=str, required=True, help="Raw activations directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Load metadata
    metadata_path = input_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    print(f"Processing activations for {metadata['n_latents']} latents")
    print(f"From {metadata['num_tokens']} tokens across {metadata['num_shards']} shards")

    # Load all shards
    latent_data, token_data = load_shards(input_dir)
    print(f"Loaded activations for {len(latent_data)} unique latents")
    print(f"Covering {len(token_data)} unique tokens")

    # Save per-latent indices
    stats = save_latent_indices(latent_data, output_dir, metadata["n_latents"])
    print(f"Index stats: {stats}")

    # Save token map
    save_token_map(token_data, output_dir)

    # Copy decoder
    decoder_src = input_dir / "decoder.npy"
    decoder_dst = output_dir / "decoder.npy"
    if decoder_src.exists():
        import shutil
        shutil.copy(decoder_src, decoder_dst)
        print(f"Copied decoder to {decoder_dst}")

    # Update and save metadata
    metadata["index_stats"] = stats
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nIndexing complete! Output in {output_dir}")
    print(f"  - {metadata['n_latents']} latent files in {output_dir / 'latents'}")
    print(f"  - Token map in {output_dir / 'corpus'}")
    print(f"  - Decoder matrix: {output_dir / 'decoder.npy'}")


if __name__ == "__main__":
    main()
