#!/usr/bin/env python3
"""
Build a search index for the latent visualizer.

Reads per-latent JSON files and creates an inverted index mapping
token strings to the latent IDs that fire on them. The output file
(search_index.json) is loaded by the visualizer frontend for
client-side text search across latents.

Usage:
    python build_search_index.py \
        --visualizer-dir data/experiments/.../visualizer

    # Limit to top examples only (smaller index)
    python build_search_index.py \
        --visualizer-dir data/experiments/.../visualizer \
        --sections top

    # Cap latent IDs per token for size control
    python build_search_index.py \
        --visualizer-dir data/experiments/.../visualizer \
        --max-latents-per-token 500
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def build_search_index(
    visualizer_dir: Path,
    sections: list[str] | None = None,
    max_latents_per_token: int = 0,
) -> dict:
    """Build inverted token -> latent_ids index from per-latent JSON files.

    Args:
        visualizer_dir: Path to visualizer directory containing latents/
        sections: Which example sections to include (default: all).
            Options: "top", "high", "medium", "low"
        max_latents_per_token: Max latent IDs to store per token (0 = unlimited).
            Useful for controlling output size; very common tokens like " the"
            may fire on thousands of latents.

    Returns:
        dict with "version", "n_latents", "n_tokens", and "tokens" mapping.
    """
    latents_dir = visualizer_dir / "latents"

    if not latents_dir.exists():
        raise FileNotFoundError(f"Latents directory not found: {latents_dir}")

    # Collect all latent JSON files
    latent_files = sorted(latents_dir.glob("*.json"))
    if not latent_files:
        raise FileNotFoundError(f"No latent JSON files found in {latents_dir}")

    print(f"Found {len(latent_files)} latent files")
    if sections:
        print(f"Including sections: {sections}")

    # Build inverted index: token_text -> set of latent_ids
    token_to_latents: dict[str, set[int]] = defaultdict(set)

    for i, fpath in enumerate(latent_files):
        if (i + 1) % 5000 == 0:
            print(f"  Processing {i+1}/{len(latent_files)}...")

        with open(fpath) as f:
            data = json.load(f)

        latent_id = data.get("latent_id", int(fpath.stem))
        examples = data.get("examples", {})

        # Handle old format (top_examples only)
        if "top_examples" in data and not examples:
            examples = {"top": data["top_examples"]}

        # Collect unique tokens with non-zero activation
        seen_tokens: set[str] = set()

        for section_key, section_examples in examples.items():
            # Filter sections if specified
            if sections and section_key not in sections:
                continue

            for example in (section_examples or []):
                tokens = example.get("tokens", [])
                activations = example.get("token_activations", [])

                for tok, act in zip(tokens, activations):
                    if act > 0 and tok not in seen_tokens:
                        seen_tokens.add(tok)
                        token_to_latents[tok].add(latent_id)

    # Convert sets to sorted lists, optionally capping size
    tokens_dict = {}
    n_capped = 0

    for tok in sorted(token_to_latents.keys()):
        latent_ids = sorted(token_to_latents[tok])
        if max_latents_per_token > 0 and len(latent_ids) > max_latents_per_token:
            latent_ids = latent_ids[:max_latents_per_token]
            n_capped += 1
        tokens_dict[tok] = latent_ids

    if n_capped > 0:
        print(f"  Capped {n_capped} tokens to {max_latents_per_token} latent IDs each")

    index = {
        "version": 1,
        "n_latents": len(latent_files),
        "n_tokens": len(tokens_dict),
        "tokens": tokens_dict,
    }

    print(f"Built index: {len(tokens_dict)} unique tokens")
    return index


def main():
    parser = argparse.ArgumentParser(
        description="Build search index for latent visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--visualizer-dir", type=Path, required=True,
        help="Path to visualizer directory containing latents/",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path (default: visualizer_dir/search_index.json)",
    )
    parser.add_argument(
        "--sections", nargs="+", default=None,
        choices=["top", "high", "medium", "low"],
        help="Which example sections to include (default: all)",
    )
    parser.add_argument(
        "--max-latents-per-token", type=int, default=0,
        help="Max latent IDs per token, 0=unlimited (default: 0)",
    )

    args = parser.parse_args()

    output = args.output or args.visualizer_dir / "search_index.json"

    index = build_search_index(
        visualizer_dir=args.visualizer_dir,
        sections=args.sections,
        max_latents_per_token=args.max_latents_per_token,
    )

    print(f"Writing to {output}...")
    with open(output, "w") as f:
        json.dump(index, f, separators=(",", ":"))

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Saved search index: {output} ({size_mb:.1f} MB)")
    print(f"  {index['n_tokens']} unique tokens across {index['n_latents']} latents")


if __name__ == "__main__":
    main()
