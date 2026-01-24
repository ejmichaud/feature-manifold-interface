"""
Activation harvesting script for Gemma 3 + Gemma Scope 2 SAEs.

Simplified procedural version - just loops through the corpus and collects activations.

Usage:
    python harvest_activations_v2.py \
        --layer 31 \
        --sae-width 65k \
        --sae-l0 medium \
        --output-dir /remote/ericjm/feature-manifold-interface/data/raw_activations \
        --num-tokens 10000000
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
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
    print(f"Registered hook at layer {layer}")

    return handle, captured


def save_shard(output_dir, shard_idx, shard_data):
    """Save a shard to disk."""
    if not shard_data["token_indices"]:
        return

    shard_path = output_dir / f"shard_{shard_idx:05d}.npz"
    np.savez_compressed(
        shard_path,
        token_indices=np.array(shard_data["token_indices"], dtype=np.int64),
        latent_indices=np.array(shard_data["latent_indices"], dtype=np.int32),
        activations=np.array(shard_data["activations"], dtype=np.float16),
        token_ids=np.array(shard_data["token_ids"], dtype=np.int32),
        doc_ids=np.array(shard_data["doc_ids"], dtype=np.int32),
        positions=np.array(shard_data["positions"], dtype=np.int16),
    )
    print(f"Saved shard {shard_idx} with {len(shard_data['token_indices'])} activations")


def save_latent_labels(output_dir, latent_token_counts, tokenizer, n_latents):
    """Compute and save the most frequent token for each latent."""
    labels_path = output_dir / "latent_labels.json"
    print("Computing latent labels...")

    labels = {}
    for latent_id in tqdm(range(n_latents), desc="Computing labels"):
        if latent_id in latent_token_counts and latent_token_counts[latent_id]:
            token_counter = latent_token_counts[latent_id]
            most_common_token_id, count = token_counter.most_common(1)[0]
            token_text = tokenizer.decode([most_common_token_id])

            labels[str(latent_id)] = {
                "token": token_text,
                "token_id": int(most_common_token_id),
                "count": int(count),
                "total_firings": sum(token_counter.values()),
            }
        else:
            labels[str(latent_id)] = {
                "token": None,
                "token_id": None,
                "count": 0,
                "total_firings": 0,
            }

    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    active_latents = sum(1 for l in labels.values() if l["total_firings"] > 0)
    print(f"Saved latent labels: {active_latents}/{n_latents} latents fired")


def harvest_activations(
    model_name="google/gemma-3-27b-pt",
    sae_release="gemma-scope-2-27b-pt-res",
    layer=31,
    sae_width="65k",
    sae_l0="medium",
    output_dir="/remote/ericjm/feature-manifold-interface/data/raw_activations",
    num_tokens=10_000_000,
    batch_size=2,
    seq_len=1024,
    shard_size=1_000_000,
    device="cuda",
    dtype=torch.bfloat16,
):
    """Main harvesting function."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct SAE ID
    sae_id = f"layer_{layer}_width_{sae_width}_l0_{sae_l0}"

    # Load tokenizer
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model {model_name}...")
    # Use specific device if provided (e.g., cuda:5), otherwise use auto
    if device.startswith("cuda:"):
        device_map = {"": device}
    else:
        device_map = "auto"
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    print(f"  Model has {len(model.model.language_model.layers)} layers")

    # Load SAE
    print(f"Loading SAE: {sae_release} / {sae_id}...")
    sae, _, _ = SAE.from_pretrained(release=sae_release, sae_id=sae_id)
    sae = sae.to(device).to(dtype)
    sae.eval()
    print(f"  SAE: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    # Set up activation hook
    hook_handle, captured = setup_activation_hook(model, layer)

    # Get special tokens to skip
    special_token_ids = set(tokenizer.all_special_ids)
    print(f"Skipping {len(special_token_ids)} special tokens: {special_token_ids}")

    # Initialize tracking
    shard_data = {
        "token_indices": [],
        "latent_indices": [],
        "activations": [],
        "token_ids": [],
        "doc_ids": [],
        "positions": [],
    }
    latent_token_counts = defaultdict(Counter)
    current_shard = 0
    global_token_idx = 0
    doc_id = 0

    # Load corpus
    print("Loading pile-uncopyrighted dataset...")
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

    # Process corpus
    batch_texts = []
    batch_doc_ids = []
    pbar = tqdm(total=num_tokens, desc="Harvesting")

    try:
        for doc in dataset:
            text = doc["text"]
            if not text or not text.strip():
                continue

            batch_texts.append(text)
            batch_doc_ids.append(doc_id)
            doc_id += 1

            # Process batch when full
            if len(batch_texts) >= batch_size:
                # Tokenize
                tokens = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=seq_len,
                )
                input_ids = tokens["input_ids"].to(device)
                attention_mask = tokens["attention_mask"].to(device)

                # Forward pass (hook captures activations)
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                # Encode through SAE
                hidden = captured["activations"]
                latent_acts = sae.encode(hidden)  # (batch, seq, d_sae)

                # Process each token
                batch_size_actual, seq_len_actual = input_ids.shape
                for batch_idx in range(batch_size_actual):
                    for pos in range(seq_len_actual):
                        # Skip padding and special tokens
                        if attention_mask[batch_idx, pos] == 0:
                            continue

                        token_id = input_ids[batch_idx, pos].item()
                        if token_id in special_token_ids:
                            continue

                        # Get non-zero latent activations
                        acts = latent_acts[batch_idx, pos]
                        nonzero_indices = (acts > 0).nonzero(as_tuple=True)[0]

                        # Store each activation
                        for latent_idx in nonzero_indices:
                            latent_idx_int = latent_idx.item()
                            act_value = acts[latent_idx].item()

                            shard_data["token_indices"].append(global_token_idx)
                            shard_data["latent_indices"].append(latent_idx_int)
                            shard_data["activations"].append(act_value)
                            shard_data["token_ids"].append(token_id)
                            shard_data["doc_ids"].append(batch_doc_ids[batch_idx])
                            shard_data["positions"].append(pos)

                            # Track for labels
                            latent_token_counts[latent_idx_int][token_id] += 1

                            # Save shard if full
                            if len(shard_data["token_indices"]) >= shard_size:
                                save_shard(output_dir, current_shard, shard_data)
                                current_shard += 1
                                shard_data = {k: [] for k in shard_data}

                        global_token_idx += 1

                pbar.update(global_token_idx - pbar.n)
                batch_texts = []
                batch_doc_ids = []

                if global_token_idx >= num_tokens:
                    break

    finally:
        # Save remaining data
        save_shard(output_dir, current_shard, shard_data)

        # Save decoder
        decoder = sae.W_dec.detach().cpu().float().numpy()
        np.save(output_dir / "decoder.npy", decoder)
        print(f"Saved decoder: shape {decoder.shape}")

        # Save labels
        save_latent_labels(output_dir, latent_token_counts, tokenizer, sae.cfg.d_sae)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "sae_release": sae_release,
            "sae_id": sae_id,
            "layer": layer,
            "num_tokens": global_token_idx,
            "n_latents": sae.cfg.d_sae,
            "d_model": sae.cfg.d_in,
            "num_shards": current_shard + 1,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata")

        hook_handle.remove()
        pbar.close()

    print(f"\nHarvesting complete!")
    print(f"  Processed: {global_token_idx:,} tokens")
    print(f"  Shards: {current_shard + 1}")


def main():
    parser = argparse.ArgumentParser(description="Harvest SAE activations")
    parser.add_argument("--layer", type=int, default=31)
    parser.add_argument("--sae-width", type=str, default="65k", choices=["16k", "65k", "262k", "1m"])
    parser.add_argument("--sae-l0", type=str, default="medium", choices=["small", "medium", "big"])
    parser.add_argument("--output-dir", type=str, default="/remote/ericjm/feature-manifold-interface/data/raw_activations")
    parser.add_argument("--num-tokens", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--shard-size", type=int, default=1_000_000)
    parser.add_argument("--device", type=str, default="cuda", help="Device for SAE and tensors (e.g., cuda:5)")

    args = parser.parse_args()

    harvest_activations(
        layer=args.layer,
        sae_width=args.sae_width,
        sae_l0=args.sae_l0,
        output_dir=args.output_dir,
        num_tokens=args.num_tokens,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        shard_size=args.shard_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
