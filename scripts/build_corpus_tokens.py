#!/usr/bin/env python3
"""
Build corpus token data for hover context in the cluster explorer.

Scans raw activation shards to build:
1. token_ids.npy — maps global_token_index → vocab_token_id (int32)
2. vocab.json — maps vocab_id → decoded token string

These files enable the backend to provide hover text showing
token context when users hover over points in the PCA scatter plots.

Usage:
    python build_corpus_tokens.py \
        --experiment-dir data/experiments/gemma-3-27b-pt_layer31_65k_medium

    # Specify model explicitly (overrides metadata)
    python build_corpus_tokens.py \
        --experiment-dir data/experiments/gemma-3-27b-pt_layer31_65k_medium \
        --model google/gemma-3-27b-pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def build_token_ids(raw_dir: Path, n_tokens: int) -> np.ndarray:
    """Scan all raw shards and extract token_index → token_id mapping.

    Args:
        raw_dir: Directory containing shard_*.npz files
        n_tokens: Total number of tokens in the corpus

    Returns:
        np.ndarray of shape (n_tokens,), dtype int32.
        Entries are vocab token IDs, or -1 if unknown.
    """
    token_ids = np.full(n_tokens, -1, dtype=np.int32)
    shards = sorted(raw_dir.glob("shard_*.npz"))

    if not shards:
        raise FileNotFoundError(f"No shard files found in {raw_dir}")

    print(f"Scanning {len(shards)} shards for token IDs...")

    for i, shard_path in enumerate(shards):
        if (i + 1) % 100 == 0 or i == len(shards) - 1:
            print(f"  Shard {i+1}/{len(shards)}")

        data = np.load(shard_path)
        indices = data["token_indices"]
        ids = data["token_ids"]

        # Only write where we haven't already filled in
        # (each token appears in multiple shards, once per active latent,
        # but the token_id is the same regardless of which latent)
        mask = indices < n_tokens  # safety check
        token_ids[indices[mask]] = ids[mask]

    filled = np.sum(token_ids >= 0)
    pct = filled / n_tokens * 100
    print(f"Filled {filled:,} / {n_tokens:,} token positions ({pct:.1f}%)")

    return token_ids


def build_vocab(model_name: str) -> dict[str, str]:
    """Build vocabulary lookup from tokenizer.

    Args:
        model_name: HuggingFace model name (e.g., "google/gemma-3-27b-pt")

    Returns:
        Dict mapping str(vocab_id) → decoded token string
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers package required for vocab building.")
        print("Install with: pip install transformers")
        sys.exit(1)

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    vocab_size = tokenizer.vocab_size
    print(f"Building vocab lookup ({vocab_size:,} entries)...")

    vocab = {}
    for i in range(vocab_size):
        try:
            vocab[str(i)] = tokenizer.decode([i])
        except Exception:
            vocab[str(i)] = f"<unk:{i}>"

    return vocab


def main():
    parser = argparse.ArgumentParser(
        description="Build corpus token data for hover context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiment-dir", type=Path, required=True,
        help="Path to experiment directory",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="HuggingFace model name (default: from metadata.json)",
    )
    parser.add_argument(
        "--skip-vocab", action="store_true",
        help="Skip vocab.json generation (only build token_ids.npy)",
    )
    args = parser.parse_args()

    exp_dir = args.experiment_dir
    raw_dir = exp_dir / "raw_activations"
    corpus_dir = exp_dir / "corpus"

    # Load metadata
    metadata_path = exp_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"ERROR: metadata.json not found at {metadata_path}")
        sys.exit(1)

    with open(metadata_path) as f:
        metadata = json.load(f)

    n_tokens = metadata["num_tokens"]
    model_name = args.model or metadata.get("model_name")

    if not model_name and not args.skip_vocab:
        print("ERROR: model_name not in metadata and --model not specified")
        sys.exit(1)

    print(f"Experiment: {exp_dir.name}")
    print(f"Tokens: {n_tokens:,}")
    print(f"Model: {model_name}")

    # Ensure corpus dir exists
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build token_ids.npy
    token_ids_path = corpus_dir / "token_ids.npy"
    if token_ids_path.exists():
        print(f"\ntoken_ids.npy already exists at {token_ids_path}")
        print("Use --force or delete it to rebuild.")
    else:
        token_ids = build_token_ids(raw_dir, n_tokens)
        np.save(token_ids_path, token_ids)
        size_mb = token_ids_path.stat().st_size / (1024 * 1024)
        print(f"Saved: {token_ids_path} ({size_mb:.1f} MB)")

    # Step 2: Build vocab.json
    vocab_path = corpus_dir / "vocab.json"
    if args.skip_vocab:
        print("\nSkipping vocab.json (--skip-vocab)")
    elif vocab_path.exists():
        print(f"\nvocab.json already exists at {vocab_path}")
        print("Use --force or delete it to rebuild.")
    else:
        vocab = build_vocab(model_name)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)
        size_mb = vocab_path.stat().st_size / (1024 * 1024)
        print(f"Saved: {vocab_path} ({size_mb:.1f} MB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
