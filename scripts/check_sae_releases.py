"""Check what SAE releases are available in SAELens."""

from sae_lens import SAE

# Try to see what releases are available
try:
    # This might work if SAELens has a method to list releases
    if hasattr(SAE, 'list_releases'):
        print("Available releases:")
        print(SAE.list_releases())
except Exception as e:
    print(f"Couldn't list releases: {e}")

# Try loading with different release name variations
release_variants = [
    "gemma-scope-2-27b-pt-resid_post",
    "gemma-scope-2-27b-pt",
    "google/gemma-scope-2-27b-pt-resid_post",
    "google/gemma-scope-2-27b-pt",
]

sae_id = "layer_31_width_64k_l0_medium"

for release in release_variants:
    try:
        print(f"\nTrying release: {release}")
        sae, cfg, sparsity = SAE.from_pretrained(release=release, sae_id=sae_id)
        print(f"  ✓ SUCCESS with release: {release}")
        print(f"    SAE config: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")
        break
    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:100]}")

# Also check SAELens version
import sae_lens
print(f"\nSAELens version: {sae_lens.__version__}")
