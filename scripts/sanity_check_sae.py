"""
Sanity check script for SAE quality.

Runs documents through the model and SAE to measure:
1. Variance explained by SAE reconstruction
2. L0 (average number of active latents per token)
3. Activation statistics

Usage:
    python sanity_check_sae.py --layer 31 --sae-width 65k --sae-l0 medium --num-docs 100
"""

import argparse

import torch
import numpy as np
from datasets import load_dataset
from sae_lens import SAE
from tqdm import tqdm
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration


def setup_activation_hook(model, layer):
    """Set up hook to capture activations at the specified layer."""
    captured = {"activations": None}

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["activations"] = hidden.detach()

    layer_module = model.model.language_model.layers[layer]
    handle = layer_module.register_forward_hook(hook_fn)

    return handle, captured


def compute_variance_explained(original: torch.Tensor, reconstruction: torch.Tensor) -> float:
    """
    Compute variance explained: 1 - Var(residual) / Var(original)

    Args:
        original: (batch, seq, d_model) original activations
        reconstruction: (batch, seq, d_model) SAE reconstruction

    Returns:
        Variance explained (0 to 1, higher is better)
    """
    residual = original - reconstruction

    # Flatten to compute overall variance
    original_flat = original.reshape(-1, original.shape[-1])
    residual_flat = residual.reshape(-1, residual.shape[-1])

    # Variance per dimension, then mean
    var_original = original_flat.var(dim=0).mean()
    var_residual = residual_flat.var(dim=0).mean()

    variance_explained = 1.0 - (var_residual / var_original)
    return variance_explained.item()


def compute_l0(latent_acts: torch.Tensor) -> float:
    """
    Compute L0: average number of non-zero latents per token.

    Args:
        latent_acts: (batch, seq, d_sae) latent activations

    Returns:
        Average L0 per token
    """
    # Count non-zero activations per token
    nonzero_per_token = (latent_acts > 0).sum(dim=-1).float()  # (batch, seq)
    return nonzero_per_token.mean().item()


def main():
    parser = argparse.ArgumentParser(description="Sanity check SAE quality")
    parser.add_argument("--model-name", type=str, default="google/gemma-3-27b-pt",
                        help="Model name")
    parser.add_argument("--layer", type=int, default=31,
                        help="Layer to analyze")
    parser.add_argument("--sae-release", type=str, default="gemma-scope-2-27b-pt-res",
                        help="SAE release name")
    parser.add_argument("--sae-width", type=str, default="65k",
                        choices=["16k", "65k", "262k", "1m"],
                        help="SAE width")
    parser.add_argument("--sae-l0", type=str, default="medium",
                        choices=["small", "medium", "big"],
                        help="SAE L0 target")
    parser.add_argument("--num-docs", type=int, default=100,
                        help="Number of documents to process")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (e.g., cuda, cuda:0)")

    args = parser.parse_args()

    # Construct SAE ID
    sae_id = f"layer_{args.layer}_width_{args.sae_width}_l0_{args.sae_l0}"
    print(f"SAE: {args.sae_release} / {sae_id}")

    # Load tokenizer
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model {args.model_name}...")
    dtype = torch.bfloat16
    if args.device.startswith("cuda:"):
        device_map = {"": args.device}
    else:
        device_map = "auto"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    print(f"  Model loaded with {len(model.model.language_model.layers)} layers")

    # Load SAE
    print(f"Loading SAE...")
    sae, _, _ = SAE.from_pretrained(release=args.sae_release, sae_id=sae_id)
    sae = sae.to(args.device).to(dtype)
    sae.eval()
    print(f"  SAE: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    # Set up activation hook
    hook_handle, captured = setup_activation_hook(model, args.layer)

    # Get special tokens to skip
    special_token_ids = set(tokenizer.all_special_ids)

    # Load dataset
    print(f"Loading pile-uncopyrighted dataset...")
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

    # Collect statistics
    all_variance_explained = []
    all_l0 = []
    all_max_acts = []
    all_mean_acts = []
    total_tokens = 0

    print(f"\nProcessing {args.num_docs} documents...")
    doc_iter = iter(dataset)

    for doc_idx in tqdm(range(args.num_docs), desc="Documents"):
        try:
            doc = next(doc_iter)
        except StopIteration:
            print(f"Ran out of documents at {doc_idx}")
            break

        text = doc["text"]
        if not text or not text.strip():
            continue

        # Tokenize
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.seq_len,
            padding=False,
        )
        input_ids = tokens["input_ids"].to(args.device)

        if input_ids.shape[1] < 10:  # Skip very short docs
            continue

        # Forward pass
        with torch.no_grad():
            model(input_ids=input_ids)

        # Get activations
        hidden = captured["activations"]  # (1, seq, d_model)

        # SAE encode and decode
        latent_acts = sae.encode(hidden)  # (1, seq, d_sae)
        reconstruction = sae.decode(latent_acts)  # (1, seq, d_model)

        # Compute metrics
        var_exp = compute_variance_explained(hidden, reconstruction)
        l0 = compute_l0(latent_acts)

        # Activation statistics (only for non-zero activations)
        nonzero_acts = latent_acts[latent_acts > 0]
        if len(nonzero_acts) > 0:
            max_act = nonzero_acts.max().item()
            mean_act = nonzero_acts.mean().item()
        else:
            max_act = 0
            mean_act = 0

        all_variance_explained.append(var_exp)
        all_l0.append(l0)
        all_max_acts.append(max_act)
        all_mean_acts.append(mean_act)
        total_tokens += input_ids.shape[1]

    # Remove hook
    hook_handle.remove()

    # Print results
    print("\n" + "=" * 60)
    print("SAE SANITY CHECK RESULTS")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Layer: {args.layer}")
    print(f"  SAE: {args.sae_release} / {sae_id}")
    print(f"  Documents: {len(all_variance_explained)}")
    print(f"  Total tokens: {total_tokens:,}")

    print(f"\nReconstruction Quality:")
    print(f"  Variance explained: {np.mean(all_variance_explained):.4f} ± {np.std(all_variance_explained):.4f}")
    print(f"    (min: {np.min(all_variance_explained):.4f}, max: {np.max(all_variance_explained):.4f})")

    print(f"\nSparsity (L0):")
    print(f"  Avg active latents/token: {np.mean(all_l0):.1f} ± {np.std(all_l0):.1f}")
    print(f"    (min: {np.min(all_l0):.1f}, max: {np.max(all_l0):.1f})")
    print(f"  Sparsity: {100 * (1 - np.mean(all_l0) / sae.cfg.d_sae):.2f}% zeros")

    print(f"\nActivation Statistics:")
    print(f"  Max activation: {np.mean(all_max_acts):.2f} ± {np.std(all_max_acts):.2f}")
    print(f"  Mean activation (when active): {np.mean(all_mean_acts):.3f} ± {np.std(all_mean_acts):.3f}")

    # Interpretation
    print(f"\n" + "-" * 60)
    print("Interpretation:")
    var_exp_mean = np.mean(all_variance_explained)
    l0_mean = np.mean(all_l0)

    if var_exp_mean > 0.9:
        print(f"  ✓ Variance explained ({var_exp_mean:.1%}) is excellent (>90%)")
    elif var_exp_mean > 0.8:
        print(f"  ~ Variance explained ({var_exp_mean:.1%}) is good (80-90%)")
    else:
        print(f"  ✗ Variance explained ({var_exp_mean:.1%}) is low (<80%)")

    expected_l0 = {"small": 10, "medium": 50, "big": 150}.get(args.sae_l0, 50)
    if 0.5 * expected_l0 < l0_mean < 2 * expected_l0:
        print(f"  ✓ L0 ({l0_mean:.0f}) is in expected range for '{args.sae_l0}' (~{expected_l0})")
    else:
        print(f"  ? L0 ({l0_mean:.0f}) differs from expected for '{args.sae_l0}' (~{expected_l0})")

    print("=" * 60)


if __name__ == "__main__":
    main()
