#!/usr/bin/env python3
"""
Upload trained model weights to HuggingFace Spaces.

Usage:
    python notebooks/upload_models_to_hf.py --token hf_YOUR_TOKEN_HERE

Get your token from: https://huggingface.co/settings/tokens
(needs write access)
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True, help="HuggingFace write token")
parser.add_argument("--space", default="aqibdev01/workflow360-ai-server", help="HF Space repo ID")
args = parser.parse_args()

ROOT = Path(__file__).parent.parent
MODEL_WEIGHTS = ROOT / "model_weights"

if not MODEL_WEIGHTS.exists():
    print("ERROR: model_weights/ directory not found. Run run_training.py first.")
    sys.exit(1)

api = HfApi(token=args.token)

print(f"Uploading model_weights/ to HF Space: {args.space}")
print("Files to upload:")
for f in sorted(MODEL_WEIGHTS.rglob("*")):
    if f.is_file() and not f.suffix == ".png":
        rel = f.relative_to(ROOT)
        size_kb = f.stat().st_size / 1024
        print(f"  {rel}  ({size_kb:.1f} KB)")

print("\nUploading...")

for f in sorted(MODEL_WEIGHTS.rglob("*")):
    if f.is_file() and f.suffix != ".png":
        path_in_repo = str(f.relative_to(ROOT)).replace("\\", "/")
        print(f"  Uploading {path_in_repo}...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=path_in_repo,
            repo_id=args.space,
            repo_type="space",
            token=args.token,
        )
        print("done")

print("\nAll model files uploaded successfully!")
print(f"HF Space will rebuild automatically: https://huggingface.co/spaces/{args.space}")
