#!/usr/bin/env python3
"""
Full pipeline orchestrator for Feature Manifold Interface.

Runs the complete pipeline for a given model/SAE configuration:
1. Harvest activations (optional - skip if raw data exists)
2. Build per-latent indices
3. Compute graph edges and UMAP positions
4. Extract top activations for visualizer
5. Export UMAP positions to JSON

Usage:
    # Run full pipeline for a new configuration
    python run_pipeline.py \
        --model google/gemma-3-27b-pt \
        --layer 31 \
        --sae-width 65k \
        --sae-l0 medium \
        --num-tokens 10000000 \
        --data-root /remote/ericjm/feature-manifold-interface/data \
        --device cuda:0

    # Skip harvesting (use existing raw data)
    python run_pipeline.py \
        --model google/gemma-3-27b-pt \
        --layer 31 \
        --sae-width 65k \
        --sae-l0 medium \
        --data-root /remote/ericjm/feature-manifold-interface/data \
        --skip-harvest

    # Run from config file
    python run_pipeline.py --config experiments/gemma3_layer31.yaml

    # List what would be run (dry run)
    python run_pipeline.py --config experiments/gemma3_layer31.yaml --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_experiment_id(model: str, layer: int, sae_width: str, sae_l0: str) -> str:
    """Generate experiment ID from parameters."""
    # Extract model short name (e.g., "gemma-3-27b-pt" from "google/gemma-3-27b-pt")
    model_short = model.split("/")[-1]
    return f"{model_short}_layer{layer}_{sae_width}_{sae_l0}"


def setup_directories(data_root: Path, experiment_id: str) -> dict:
    """Create directory structure for an experiment.

    All data for one experiment lives under a single directory:
        data/experiments/{id}/
        ├── raw_activations/
        ├── latents/
        ├── graph/
        ├── corpus/
        ├── decoder.npy
        ├── metadata.json
        └── visualizer/
            ├── index.json
            ├── positions.json
            └── latents/
    """
    exp_dir = data_root / "experiments" / experiment_id

    dirs = {
        "experiment": exp_dir,
        "raw_activations": exp_dir / "raw_activations",
        "latents": exp_dir / "latents",
        "graph": exp_dir / "graph",
        "corpus": exp_dir / "corpus",
        "visualizer": exp_dir / "visualizer",
        "visualizer_latents": exp_dir / "visualizer" / "latents",
    }

    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def run_command(cmd: list, description: str, dry_run: bool = False) -> bool:
    """Run a command with logging."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n{'=' * 60}")
    print(f"STEP: {description}")
    print(f"{'=' * 60}")
    print(f"Command: {cmd_str}\n")

    if dry_run:
        print("[DRY RUN] Would execute above command")
        return True

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError as e:
        print(f"\n✗ Command not found: {e}")
        return False


def check_step_complete(marker_file: Path) -> bool:
    """Check if a step has already been completed."""
    return marker_file.exists()


def mark_step_complete(marker_file: Path, metadata: dict = None):
    """Mark a step as complete."""
    data = {
        "completed_at": datetime.now().isoformat(),
        **(metadata or {}),
    }
    with open(marker_file, "w") as f:
        json.dump(data, f, indent=2)


def run_pipeline(
    model: str,
    layer: int,
    sae_width: str,
    sae_l0: str,
    data_root: Path,
    num_tokens: int = 10_000_000,
    device: str = "cuda:0",
    batch_size: int = 2,
    seq_len: int = 1024,
    top_k: int = 50,
    context_size: int = 10,
    skip_harvest: bool = False,
    skip_edges: bool = False,
    force: bool = False,
    dry_run: bool = False,
    n_workers: int = 8,
):
    """Run the full pipeline for a configuration."""

    experiment_id = get_experiment_id(model, layer, sae_width, sae_l0)
    print(f"\n{'#' * 60}")
    print(f"# Pipeline: {experiment_id}")
    print(f"{'#' * 60}")
    print(f"Model: {model}")
    print(f"Layer: {layer}")
    print(f"SAE: width={sae_width}, l0={sae_l0}")
    print(f"Tokens: {num_tokens:,}")
    print(f"Device: {device}")
    print(f"Data root: {data_root}")

    # Setup directories
    dirs = setup_directories(data_root, experiment_id)
    print(f"\nExperiment directory: {dirs['experiment']}")

    scripts_dir = Path(__file__).parent

    # Track overall success
    all_success = True

    # Step 1: Harvest activations
    harvest_marker = dirs["raw_activations"] / ".harvest_complete"
    if skip_harvest:
        print("\n[SKIP] Harvest activations (--skip-harvest)")
    elif check_step_complete(harvest_marker) and not force:
        print("\n[SKIP] Harvest activations (already complete)")
    else:
        success = run_command(
            [
                sys.executable, scripts_dir / "harvest_activations_v2.py",
                "--layer", str(layer),
                "--sae-width", sae_width,
                "--sae-l0", sae_l0,
                "--output-dir", str(dirs["raw_activations"]),
                "--num-tokens", str(num_tokens),
                "--batch-size", str(batch_size),
                "--seq-len", str(seq_len),
                "--device", device,
            ],
            "Harvest activations",
            dry_run=dry_run,
        )
        if success and not dry_run:
            mark_step_complete(harvest_marker, {"num_tokens": num_tokens})
        all_success = all_success and success

    # Step 2: Build per-latent indices
    index_marker = dirs["experiment"] / ".index_complete"
    if check_step_complete(index_marker) and not force:
        print("\n[SKIP] Build indices (already complete)")
    else:
        success = run_command(
            [
                sys.executable, scripts_dir / "build_indices.py",
                "--input-dir", str(dirs["raw_activations"]),
                "--output-dir", str(dirs["experiment"]),
                "--n-workers", str(n_workers),
            ],
            "Build per-latent indices",
            dry_run=dry_run,
        )
        if success and not dry_run:
            mark_step_complete(index_marker)
        all_success = all_success and success

    # Step 3: Compute graph edges and UMAP
    graph_marker = dirs["graph"] / ".graph_complete"
    if skip_edges:
        # Still need UMAP, so run with --skip-coactivation --skip-jaccard
        if check_step_complete(graph_marker) and not force:
            print("\n[SKIP] Compute UMAP (already complete)")
        else:
            success = run_command(
                [
                    sys.executable, scripts_dir / "compute_edges.py",
                    "--data-dir", str(dirs["experiment"]),
                    "--top-k", "100",
                    "--skip-coactivation",
                    "--skip-jaccard",
                ],
                "Compute UMAP positions (edges skipped)",
                dry_run=dry_run,
            )
            if success and not dry_run:
                mark_step_complete(graph_marker, {"edges_skipped": True})
            all_success = all_success and success
    elif check_step_complete(graph_marker) and not force:
        print("\n[SKIP] Compute graph edges (already complete)")
    else:
        success = run_command(
            [
                sys.executable, scripts_dir / "compute_edges.py",
                "--data-dir", str(dirs["experiment"]),
                "--top-k", "100",
            ],
            "Compute graph edges and UMAP positions",
            dry_run=dry_run,
        )
        if success and not dry_run:
            mark_step_complete(graph_marker)
        all_success = all_success and success

    # Step 4: Extract top activations for visualizer
    extract_marker = dirs["visualizer"] / ".extract_complete"
    if check_step_complete(extract_marker) and not force:
        print("\n[SKIP] Extract top activations (already complete)")
    else:
        success = run_command(
            [
                sys.executable, scripts_dir / "extract_top_activations.py",
                "--raw-dir", str(dirs["raw_activations"]),
                "--output-dir", str(dirs["visualizer"]),
                "--top-k", str(top_k),
                "--context-size", str(context_size),
                "--sample-percentiles",
                "--model-name", model,
                "--n-workers", str(n_workers),
            ],
            "Extract top activations for visualizer",
            dry_run=dry_run,
        )
        if success and not dry_run:
            mark_step_complete(extract_marker, {"top_k": top_k})
        all_success = all_success and success

    # Step 5: Export UMAP positions to JSON
    positions_json = dirs["visualizer"] / "positions.json"
    if positions_json.exists() and not force:
        print("\n[SKIP] Export UMAP positions (already exists)")
    else:
        success = run_command(
            [
                sys.executable, scripts_dir / "export_umap_json.py",
                "--positions-file", str(dirs["graph"] / "positions.npy"),
                "--output", str(positions_json),
            ],
            "Export UMAP positions to JSON",
            dry_run=dry_run,
        )
        all_success = all_success and success

    # Update experiments manifest
    run_command(
        [
            sys.executable, scripts_dir / "update_experiments_manifest.py",
            "--data-root", str(data_root),
        ],
        "Update experiments manifest",
        dry_run=dry_run,
    )

    # Summary
    print(f"\n{'#' * 60}")
    if all_success:
        print(f"# Pipeline completed successfully!")
    else:
        print(f"# Pipeline completed with errors")
    print(f"{'#' * 60}")
    print(f"\nExperiment directory: {dirs['experiment']}")
    print(f"  ├── raw_activations/")
    print(f"  ├── latents/")
    print(f"  ├── graph/")
    print(f"  └── visualizer/")
    print(f"\nTo serve the visualizer:")
    print(f"  cd {data_root.parent}/visualizer")
    print(f"  # Update DATA_DIR in umap.html to point to:")
    print(f"  #   ../data/experiments/{experiment_id}/visualizer")
    print(f"  python -m http.server 8080")

    return all_success


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML or JSON file."""
    import yaml  # Optional dependency

    with open(config_path) as f:
        if config_path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run full Feature Manifold Interface pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config file option
    parser.add_argument("--config", type=Path, help="Path to config file (YAML/JSON)")

    # Model/SAE parameters
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-pt",
                        help="Model name")
    parser.add_argument("--layer", type=int, default=31,
                        help="Layer to extract from")
    parser.add_argument("--sae-width", type=str, default="65k",
                        choices=["16k", "65k", "262k", "1m"],
                        help="SAE width")
    parser.add_argument("--sae-l0", type=str, default="medium",
                        choices=["small", "medium", "big"],
                        help="SAE L0 sparsity")

    # Data parameters
    parser.add_argument("--num-tokens", type=int, default=10_000_000,
                        help="Number of tokens to harvest")
    parser.add_argument("--data-root", type=Path,
                        default=Path("/remote/ericjm/feature-manifold-interface/data"),
                        help="Root directory for all data")

    # Compute parameters
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for model/SAE")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size for harvesting")
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="Sequence length")
    parser.add_argument("--n-workers", type=int, default=8,
                        help="Number of parallel workers")

    # Visualizer parameters
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k examples per latent")
    parser.add_argument("--context-size", type=int, default=10,
                        help="Context window size")

    # Control flags
    parser.add_argument("--skip-harvest", action="store_true",
                        help="Skip harvesting (use existing raw data)")
    parser.add_argument("--skip-edges", action="store_true",
                        help="Skip edge computation (only UMAP)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run of all steps")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        # Override with config values (CLI args take precedence)
        for key, value in config.items():
            key_underscore = key.replace("-", "_")
            if hasattr(args, key_underscore) and getattr(args, key_underscore) == parser.get_default(key_underscore):
                setattr(args, key_underscore, value)

    # Ensure data_root is a Path
    if isinstance(args.data_root, str):
        args.data_root = Path(args.data_root)

    success = run_pipeline(
        model=args.model,
        layer=args.layer,
        sae_width=args.sae_width,
        sae_l0=args.sae_l0,
        data_root=args.data_root,
        num_tokens=args.num_tokens,
        device=args.device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        top_k=args.top_k,
        context_size=args.context_size,
        skip_harvest=args.skip_harvest,
        skip_edges=args.skip_edges,
        force=args.force,
        dry_run=args.dry_run,
        n_workers=args.n_workers,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
