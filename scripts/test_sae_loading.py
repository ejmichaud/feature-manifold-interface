"""
Test script to verify SAE loading and activation capture for Gemma 3 + Gemma Scope 2.

Run this before the full harvest to ensure:
1. The SAE loads correctly via SAELens and has expected shape
2. Activations are captured at the right layer
3. SAE encoding produces reasonable sparse activations

Usage:
    python test_sae_loading.py --layer 12 --sae-width 16k --sae-l0 small

    # Or test with a small model first (if 27B is too slow):
    python test_sae_loading.py --layer 12 --sae-width 16k --sae-l0 small --quick
"""

import argparse
import torch
from transformers import AutoTokenizer
from transformers import Gemma3ForConditionalGeneration
from sae_lens import SAE


def test_sae_loading(sae_release: str, sae_id: str):
    """Test that SAE loads correctly via SAELens."""
    print("\n" + "=" * 60)
    print("TEST 1: SAE Loading via SAELens")
    print("=" * 60)

    try:
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
        )

        print(f"  SAE loaded successfully!")
        print(f"  d_in (model dim): {sae.cfg.d_in}")
        print(f"  d_sae (latent dim): {sae.cfg.d_sae}")
        print(f"  hook_name: {sae.cfg.hook_name}")
        print(f"  activation_fn: {sae.cfg.activation_fn}")

        # Check W_dec shape
        print(f"  W_dec shape: {sae.W_dec.shape}")
        print(f"  W_enc shape: {sae.W_enc.shape}")

        if hasattr(sae, 'threshold'):
            print(f"  Has threshold (JumpReLU): {sae.threshold is not None}")

        print("  PASSED")
        return sae
    except Exception as e:
        print(f"  FAILED: {e}")
        raise


def test_model_loading(model_name: str, layer: int, device: str = "cuda"):
    """Test that we can load the model and access layers."""
    print("\n" + "=" * 60)
    print("TEST 2: Model Loading")
    print("=" * 60)

    try:
        print(f"  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"  Loading model (this may take a while for 27B)...")
        # Use specific device if provided (e.g., cuda:5), otherwise use auto
        if device.startswith("cuda:"):
            device_map = {"": device}
        else:
            device_map = "auto"
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        model.eval()

        num_layers = len(model.model.layers)
        print(f"  Model has {num_layers} layers")

        if layer >= num_layers:
            print(f"  FAILED: Layer {layer} doesn't exist (max is {num_layers - 1})")
            raise ValueError(f"Layer {layer} out of range")

        print(f"  Layer {layer} module: {type(model.model.layers[layer])}")
        print("  PASSED")
        return model, tokenizer

    except Exception as e:
        print(f"  FAILED: {e}")
        raise


def test_hook_capture(model, tokenizer, layer: int, device: str = "cuda"):
    """Test that we can capture activations from the model."""
    print("\n" + "=" * 60)
    print("TEST 3: Activation Capture via Hook")
    print("=" * 60)

    captured = [None]

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured[0] = output[0].detach()
        else:
            captured[0] = output.detach()

    handle = model.model.layers[layer].register_forward_hook(hook_fn)

    try:
        # Run forward pass with a simple text
        test_text = "The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(test_text, return_tensors="pt")

        # Move to model device (use specified device or auto-detect from model)
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            model(**inputs)

        if captured[0] is None:
            print("  FAILED: No activations captured")
            raise ValueError("Hook did not capture activations")

        hidden = captured[0]
        print(f"  Captured tensor shape: {hidden.shape}")
        print(f"  Expected shape: (1, {inputs['input_ids'].shape[1]}, d_model)")
        print(f"  Hidden dim: {hidden.shape[-1]}")
        print(f"  Dtype: {hidden.dtype}")
        print(f"  Mean: {hidden.float().mean().item():.4f}")
        print(f"  Std: {hidden.float().std().item():.4f}")
        print("  PASSED")
        return hidden

    finally:
        handle.remove()


def test_sae_encoding(sae, hidden: torch.Tensor, device: str = "cuda"):
    """Test SAE encoding produces reasonable outputs."""
    print("\n" + "=" * 60)
    print("TEST 4: SAE Encoding")
    print("=" * 60)

    # Move SAE to same device/dtype as hidden
    target_device = hidden.device
    dtype = hidden.dtype
    sae = sae.to(target_device).to(dtype)

    # Check dimension compatibility
    if hidden.shape[-1] != sae.cfg.d_in:
        print(f"  FAILED: Hidden dim {hidden.shape[-1]} != SAE d_in {sae.cfg.d_in}")
        raise ValueError("Dimension mismatch between model and SAE")

    print(f"  Dimensions match: {hidden.shape[-1]} == {sae.cfg.d_in}")

    # Encode
    with torch.no_grad():
        latents = sae.encode(hidden)

    print(f"  Latent shape: {latents.shape}")
    print(f"  Expected shape: {hidden.shape[:-1]} + ({sae.cfg.d_sae},)")

    # Check sparsity
    nonzero_frac = (latents > 0).float().mean().item()
    print(f"  Sparsity (fraction nonzero): {nonzero_frac:.4f} ({nonzero_frac * 100:.2f}%)")

    if nonzero_frac > 0.1:
        print(f"  WARNING: Low sparsity (>10% active). May be unusual.")
    elif nonzero_frac < 0.0001:
        print(f"  WARNING: Very high sparsity (<0.01% active). Check if SAE is working.")
    else:
        print(f"  Sparsity looks reasonable for SAE")

    # Check activation magnitudes
    nonzero_acts = latents[latents > 0]
    if len(nonzero_acts) > 0:
        print(f"  Mean nonzero activation: {nonzero_acts.mean().item():.4f}")
        print(f"  Max activation: {latents.max().item():.4f}")

        # Count active latents per token
        active_per_token = (latents > 0).sum(dim=-1).float()
        print(f"  Mean active latents per token: {active_per_token.mean().item():.1f}")
        print(f"  Max active latents per token: {active_per_token.max().item():.0f}")

    # Sample top latents for first token
    first_token_latents = latents[0, 0]  # (d_sae,)
    top_k = min(10, (first_token_latents > 0).sum().item())
    if top_k > 0:
        top_latents = first_token_latents.topk(top_k)
        print(f"  Top {top_k} latents for first token:")
        for idx, val in zip(top_latents.indices[:5], top_latents.values[:5]):
            print(f"    Latent {idx.item()}: {val.item():.4f}")

    print("  PASSED")
    return latents


def test_reconstruction(sae, hidden: torch.Tensor, latents: torch.Tensor):
    """Test that reconstruction makes sense."""
    print("\n" + "=" * 60)
    print("TEST 5: Decoder Reconstruction")
    print("=" * 60)

    with torch.no_grad():
        # Decode: latents @ W_dec
        # SAELens convention: W_dec is (d_sae, d_in)
        reconstructed = latents @ sae.W_dec

    # Check reconstruction quality
    # Note: This won't be perfect since SAE is sparse
    error = (hidden.float() - reconstructed.float()).pow(2).mean().sqrt()
    original_norm = hidden.float().pow(2).mean().sqrt()
    relative_error = error / original_norm

    print(f"  Reconstruction RMSE: {error.item():.4f}")
    print(f"  Original RMS: {original_norm.item():.4f}")
    print(f"  Relative error: {relative_error.item():.2%}")

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        hidden.float().flatten(),
        reconstructed.float().flatten(),
        dim=0
    )
    print(f"  Cosine similarity: {cos_sim.item():.4f}")

    if relative_error > 0.8:
        print("  WARNING: High reconstruction error. This is expected for very sparse SAEs.")
    elif relative_error > 0.5:
        print("  Moderate reconstruction error (typical for sparse SAEs)")
    else:
        print("  Good reconstruction quality")

    print("  PASSED")


def main():
    parser = argparse.ArgumentParser(description="Test SAE loading and activation capture")
    parser.add_argument("--layer", type=int, default=31, help="Layer to test (default: 31)")
    parser.add_argument("--sae-width", type=str, default="65k",
                        choices=["16k", "65k", "262k", "1m"], help="SAE width (default: 65k)")
    parser.add_argument("--sae-l0", type=str, default="medium",
                        choices=["small", "medium", "big"], help="SAE L0 variant (default: medium)")
    parser.add_argument("--quick", action="store_true",
                        help="Skip model loading (only test SAE)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for SAE and tensors (e.g., cuda:5)")

    args = parser.parse_args()

    model_name = "google/gemma-3-27b-pt"
    sae_release = "gemma-scope-2-27b-pt-res"
    sae_id = f"layer_{args.layer}_width_{args.sae_width}_l0_{args.sae_l0}"

    print(f"Testing configuration:")
    print(f"  Model: {model_name}")
    print(f"  SAE release: {sae_release}")
    print(f"  SAE ID: {sae_id}")
    print(f"  Layer: {args.layer}")

    # Test 1: SAE loading
    sae = test_sae_loading(sae_release, sae_id)

    if args.quick:
        print("\n" + "=" * 60)
        print("QUICK MODE: Skipping model tests")
        print("=" * 60)
        print("\nSAE loading test passed! Run without --quick to test full pipeline.")
        return

    # Test 2: Model loading
    model, tokenizer = test_model_loading(model_name, args.layer, args.device)

    # Test 3: Hook capture
    hidden = test_hook_capture(model, tokenizer, args.layer, args.device)

    # Test 4: SAE encoding
    latents = test_sae_encoding(sae, hidden, args.device)

    # Test 5: Reconstruction
    test_reconstruction(sae, hidden, latents)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print("\nYou can now run the full harvest with:")
    print(f"  python harvest_activations.py \\")
    print(f"      --layer {args.layer} \\")
    print(f"      --sae-width {args.sae_width} \\")
    print(f"      --sae-l0 {args.sae_l0} \\")
    print(f"      --output-dir /remote/ericjm/feature-manifold-interface/data/raw_activations \\")
    print(f"      --num-tokens 10000000")


if __name__ == "__main__":
    main()
