#!/usr/bin/env python3
"""
Upload trained model weights to HuggingFace Spaces.

Usage:
    python notebooks/upload_models_to_hf.py --token hf_YOUR_TOKEN_HERE

Get your token from: https://huggingface.co/settings/tokens
(needs Write access)
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
    print("ERROR: model_weights/ directory not found.")
    sys.exit(1)

# Checkpoint dirs are training artifacts — not needed for inference, skip them
SKIP_DIRS = {"checkpoint-200", "checkpoint-400", "checkpoint-600", "checkpoint-800",
             "checkpoint-1000", "checkpoint-1200", "checkpoint-1400", "checkpoint-1600"}
SKIP_SUFFIXES = {".png", ".jpg"}

def should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in SKIP_DIRS or part.startswith("checkpoint-") or part.startswith("."):
            return True
    return path.suffix in SKIP_SUFFIXES

api = HfApi(token=args.token)

files_to_upload = []
for f in sorted(MODEL_WEIGHTS.rglob("*")):
    if f.is_file() and not should_skip(f):
        files_to_upload.append(f)

print(f"Uploading to HF Space: {args.space}")
print(f"\nFiles ({len(files_to_upload)} total):")
for f in files_to_upload:
    rel = f.relative_to(ROOT)
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {str(rel):<60}  {size_mb:.1f} MB")

total_mb = sum(f.stat().st_size for f in files_to_upload) / (1024 * 1024)
print(f"\nTotal upload size: {total_mb:.1f} MB")
print("\nStarting upload (large files like model.safetensors will take a few minutes)...\n")

for i, f in enumerate(files_to_upload, 1):
    path_in_repo = str(f.relative_to(ROOT)).replace("\\", "/")
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"[{i}/{len(files_to_upload)}] {path_in_repo} ({size_mb:.1f} MB)...", end=" ", flush=True)
    api.upload_file(
        path_or_fileobj=str(f),
        path_in_repo=path_in_repo,
        repo_id=args.space,
        repo_type="space",
        token=args.token,
    )
    print("done")

print("\nAll model files uploaded!")
print(f"Space will rebuild in ~2-3 minutes: https://huggingface.co/spaces/{args.space}")
print("\nAfter rebuild, model versions in API responses will be:")
print("  Decomposition -> flan-t5-pm-v1  (or similar)")
print("  Assigner      -> assigner-v1")
print("  Optimizer     -> optimizer-v1")
