"""
Activation harvesting script for Gemma 3 + Gemma Scope 2 SAEs.

This script:
1. Loads Gemma 3 27B-pt and a Gemma Scope 2 SAE via SAELens
2. Runs inference on the pile-uncopyrighted corpus (text-only)
3. Captures residual stream activations at the target layer
4. Encodes through the SAE to get sparse latent activations
5. Stores activations in a format optimized for later indexing

Usage:
    python harvest_activations.py \
        --layer 31 \
        --sae-width 65k \
        --sae-l0 medium \
        --output-dir /remote/ericjm/feature-manifold-interface/data/raw_activations \
        --num-tokens 10000000

    Available SAE configs (layer_X_width_Y_l0_Z):
        widths: 16k, 65k, 262k, 1m
        l0: small, medium, big (controls sparsity)

The output format is sharded files containing:
    - token_indices: global token index for each activation
    - latent_indices: which latent fired
    - activations: the activation value
    - token_ids: vocabulary index of the token
    - contexts: (doc_id, position) for context lookup

Also generates latent_labels.json with the most common token for each latent.
"""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor

# Gemma 3 uses a specific model class for the VLM
from transformers import Gemma3ForConditionalGeneration

# SAELens for loading Gemma Scope 2 SAEs
from sae_lens import SAE


@dataclass
class HarvestConfig:
    """Configuration for activation harvesting."""
    model_name: str = "google/gemma-3-27b-pt"
    sae_release: str = "gemma-scope-2-27b-pt-res"
    layer: int = 31
    sae_width: str = "65k"
    sae_l0: str = "medium"
    output_dir: Path = None
    num_tokens: int = 10_000_000
    batch_size: int = 2  # Smaller for 27B model
    seq_len: int = 1024  # Match Gemma Scope 2 eval context
    shard_size: int = 1_000_000  # activations per shard file
    activation_threshold: float = 0.0  # only store activations above this
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    @property
    def sae_id(self) -> str:
        """Construct SAE ID from components."""
        return f"layer_{self.layer}_width_{self.sae_width}_l0_{self.sae_l0}"


class ActivationHarvester:
    """
    Harvests SAE latent activations from Gemma 3.

    Uses hooks to capture residual stream activations at the target layer,
    then encodes through the SAE.
    """

    def __init__(self, config: HarvestConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Load tokenizer (Gemma 3 uses a processor but we only need tokenizer for text)
        print(f"Loading tokenizer for {config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Ensure we have a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        print(f"Loading model {config.model_name}...")
        # Use specific device if provided (e.g., cuda:5), otherwise use auto
        if config.device.startswith("cuda:"):
            device_map = {"": config.device}
        else:
            device_map = "auto"
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            config.model_name,
            torch_dtype=config.dtype,
            device_map=device_map,
        )
        self.model.eval()

        # Print model structure for debugging
        print(f"Model type: {type(self.model)}")
        print(f"Model has {len(self.model.model.layers)} layers")

        # Load SAE via SAELens
        print(f"Loading SAE: {config.sae_release} / {config.sae_id}...")
        self.sae, self.sae_cfg, self.sae_sparsity = SAE.from_pretrained(
            release=config.sae_release,
            sae_id=config.sae_id,
        )
        self.sae = self.sae.to(config.device).to(config.dtype)
        self.sae.eval()

        print(f"SAE loaded:")
        print(f"  d_in: {self.sae.cfg.d_in}")
        print(f"  d_sae: {self.sae.cfg.d_sae}")
        print(f"  hook_name: {self.sae.cfg.hook_name}")

        # Storage for captured activations
        self.captured_activations = None

        # Register hook at the correct location
        self._register_hook()

        # Get special token IDs to skip (BOS, EOS, PAD, etc.)
        self.special_token_ids = set(self.tokenizer.all_special_ids)
        print(f"Will skip {len(self.special_token_ids)} special tokens: {self.special_token_ids}")

        # Shard tracking
        self.current_shard = 0
        self.shard_data = {
            "token_indices": [],
            "latent_indices": [],
            "activations": [],
            "token_ids": [],
            "doc_ids": [],
            "positions": [],
        }
        self.activations_in_shard = 0

        # Track token counts per latent for computing labels
        # latent_id -> Counter[token_id] -> count
        self.latent_token_counts: dict[int, Counter] = defaultdict(Counter)

    def _register_hook(self):
        """
        Register forward hook to capture residual stream activations.

        For resid_post at layer N, we hook after the layer's forward pass.
        Gemma 3 model structure: model.model.layers[N]
        """
        target_layer = self.config.layer

        # Get the target layer module
        # Gemma3ForConditionalGeneration has: model.model.layers[N]
        layer_module = self.model.model.layers[target_layer]

        def hook_fn(module, input, output):
            # For Gemma decoder layers, output is typically:
            # (hidden_states, self_attn_weights, present_key_value) or just hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self.captured_activations = hidden.detach()

        self.hook_handle = layer_module.register_forward_hook(hook_fn)
        print(f"Registered hook at layer {target_layer} (resid_post)")

    def load_corpus(self) -> Iterator[dict]:
        """Load and iterate over the corpus."""
        print("Loading pile-uncopyrighted dataset...")
        dataset = load_dataset(
            "monology/pile-uncopyrighted",
            split="train",
            streaming=True,
        )
        return iter(dataset)

    def tokenize_batch(self, texts: list[str]) -> dict:
        """Tokenize a batch of texts."""
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.seq_len,
        )

    def process_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        doc_ids: list[int],
        global_token_offset: int,
    ) -> int:
        """
        Process a batch through the model and SAE.

        Returns the number of tokens processed.
        """
        batch_size, seq_len = input_ids.shape

        # Forward pass (hook captures activations)
        with torch.no_grad():
            # For text-only, we just need input_ids and attention_mask
            self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Get captured activations and encode through SAE
        hidden = self.captured_activations  # (batch, seq, d_model)

        # SAELens encode: use the SAE's encode method
        # This handles the SAE-specific encoding (JumpReLU, etc.)
        latent_acts = self.sae.encode(hidden)  # (batch, seq, d_sae)

        # Process each token
        tokens_processed = 0
        for batch_idx in range(batch_size):
            for pos in range(seq_len):
                # Skip padding tokens
                if attention_mask[batch_idx, pos] == 0:
                    continue

                token_id = input_ids[batch_idx, pos].item()

                # Skip special tokens (BOS, EOS, PAD, etc.)
                if token_id in self.special_token_ids:
                    continue

                global_token_idx = global_token_offset + tokens_processed

                # Get non-zero latent activations
                acts = latent_acts[batch_idx, pos]  # (d_sae,)
                nonzero_mask = acts > self.config.activation_threshold
                nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]

                # Store each non-zero activation
                for latent_idx in nonzero_indices:
                    latent_idx_int = latent_idx.item()
                    act_value = acts[latent_idx].item()

                    self.shard_data["token_indices"].append(global_token_idx)
                    self.shard_data["latent_indices"].append(latent_idx_int)
                    self.shard_data["activations"].append(act_value)
                    self.shard_data["token_ids"].append(token_id)
                    self.shard_data["doc_ids"].append(doc_ids[batch_idx])
                    self.shard_data["positions"].append(pos)

                    # Track token counts for this latent (for labels)
                    self.latent_token_counts[latent_idx_int][token_id] += 1

                    self.activations_in_shard += 1

                    # Save shard if full
                    if self.activations_in_shard >= self.config.shard_size:
                        self._save_shard()

                tokens_processed += 1

        return tokens_processed

    def _save_shard(self):
        """Save current shard to disk and reset."""
        if not self.shard_data["token_indices"]:
            return

        shard_path = self.config.output_dir / f"shard_{self.current_shard:05d}.npz"

        np.savez_compressed(
            shard_path,
            token_indices=np.array(self.shard_data["token_indices"], dtype=np.int64),
            latent_indices=np.array(self.shard_data["latent_indices"], dtype=np.int32),
            activations=np.array(self.shard_data["activations"], dtype=np.float16),
            token_ids=np.array(self.shard_data["token_ids"], dtype=np.int32),
            doc_ids=np.array(self.shard_data["doc_ids"], dtype=np.int32),
            positions=np.array(self.shard_data["positions"], dtype=np.int16),
        )

        print(f"Saved shard {self.current_shard} with {self.activations_in_shard} activations")

        # Reset
        self.current_shard += 1
        self.shard_data = {k: [] for k in self.shard_data}
        self.activations_in_shard = 0

    def _save_decoder(self):
        """Save SAE decoder matrix for later reconstruction."""
        decoder_path = self.config.output_dir / "decoder.npy"

        # SAELens SAE has W_dec attribute
        # Shape: (d_sae, d_in) - each row is a feature direction
        decoder = self.sae.W_dec.detach().cpu().float().numpy()
        np.save(decoder_path, decoder)
        print(f"Saved decoder matrix: shape {decoder.shape}")

    def _save_latent_labels(self):
        """
        Compute and save labels for each latent.

        For each latent, the label is the most frequently occurring token
        (mode of the token distribution when that latent fires).
        """
        labels_path = self.config.output_dir / "latent_labels.json"

        print("Computing latent labels (most frequent token per latent)...")
        labels = {}

        for latent_id in tqdm(range(self.sae.cfg.d_sae), desc="Computing labels"):
            if latent_id in self.latent_token_counts:
                token_counter = self.latent_token_counts[latent_id]

                if token_counter:
                    # Get most common token
                    most_common_token_id, count = token_counter.most_common(1)[0]

                    # Decode token to text
                    token_text = self.tokenizer.decode([most_common_token_id])

                    labels[str(latent_id)] = {
                        "token": token_text,
                        "token_id": int(most_common_token_id),
                        "count": int(count),
                        "total_firings": sum(token_counter.values()),
                    }
                else:
                    # No activations for this latent
                    labels[str(latent_id)] = {
                        "token": None,
                        "token_id": None,
                        "count": 0,
                        "total_firings": 0,
                    }
            else:
                # Latent never fired
                labels[str(latent_id)] = {
                    "token": None,
                    "token_id": None,
                    "count": 0,
                    "total_firings": 0,
                }

        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=2)

        # Print some stats
        active_latents = sum(1 for l in labels.values() if l["total_firings"] > 0)
        print(f"Saved latent labels to {labels_path}")
        print(f"  {active_latents}/{len(labels)} latents had at least one activation")

    def harvest(self):
        """Main harvesting loop."""
        corpus = self.load_corpus()

        global_token_idx = 0
        doc_id = 0
        batch_texts = []
        batch_doc_ids = []

        pbar = tqdm(total=self.config.num_tokens, desc="Harvesting activations")

        try:
            for doc in corpus:
                text = doc["text"]

                # Skip empty documents
                if not text or not text.strip():
                    continue

                batch_texts.append(text)
                batch_doc_ids.append(doc_id)
                doc_id += 1

                if len(batch_texts) >= self.config.batch_size:
                    # Tokenize and process batch
                    tokens = self.tokenize_batch(batch_texts)
                    input_ids = tokens["input_ids"].to(self.config.device)
                    attention_mask = tokens["attention_mask"].to(self.config.device)

                    n_processed = self.process_batch(
                        input_ids, attention_mask, batch_doc_ids, global_token_idx
                    )

                    global_token_idx += n_processed
                    pbar.update(n_processed)

                    batch_texts = []
                    batch_doc_ids = []

                    if global_token_idx >= self.config.num_tokens:
                        break

        finally:
            # Save any remaining activations
            self._save_shard()

            # Save decoder matrix for later reconstruction
            self._save_decoder()

            # Compute and save latent labels
            self._save_latent_labels()

            # Save metadata
            metadata = {
                "model_name": self.config.model_name,
                "sae_release": self.config.sae_release,
                "sae_id": self.config.sae_id,
                "layer": self.config.layer,
                "num_tokens": global_token_idx,
                "n_latents": self.sae.cfg.d_sae,
                "d_model": self.sae.cfg.d_in,
                "num_shards": self.current_shard,
                "sae_config": {
                    "d_in": self.sae.cfg.d_in,
                    "d_sae": self.sae.cfg.d_sae,
                    "hook_name": self.sae.cfg.hook_name,
                },
            }
            with open(self.config.output_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata to {self.config.output_dir / 'metadata.json'}")

            # Cleanup
            self.hook_handle.remove()
            pbar.close()

        print(f"Harvesting complete! Processed {global_token_idx} tokens, {self.current_shard} shards")


def main():
    parser = argparse.ArgumentParser(description="Harvest SAE activations from Gemma 3 27B")
    parser.add_argument("--layer", type=int, default=31, help="Layer to capture (default: 31)")
    parser.add_argument("--sae-width", type=str, default="65k",
                        choices=["16k", "65k", "262k", "1m"],
                        help="SAE width (default: 65k)")
    parser.add_argument("--sae-l0", type=str, default="medium",
                        choices=["small", "medium", "big"],
                        help="SAE L0 variant (controls sparsity, default: medium)")
    parser.add_argument("--output-dir", type=str,
                        default="/remote/ericjm/feature-manifold-interface/data/raw_activations",
                        help="Output directory")
    parser.add_argument("--num-tokens", type=int, default=10_000_000, help="Tokens to process (default: 10M)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (smaller for 27B)")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length (default: 1024, matches Gemma Scope 2 evals)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for SAE and tensors (e.g., cuda:5)")

    args = parser.parse_args()

    config = HarvestConfig(
        layer=args.layer,
        sae_width=args.sae_width,
        sae_l0=args.sae_l0,
        output_dir=Path(args.output_dir),
        num_tokens=args.num_tokens,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=args.device,
    )

    print(f"Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  SAE: {config.sae_release} / {config.sae_id}")
    print(f"  Layer: {config.layer}")
    print(f"  Tokens: {config.num_tokens:,}")

    harvester = ActivationHarvester(config)
    harvester.harvest()


if __name__ == "__main__":
    main()
