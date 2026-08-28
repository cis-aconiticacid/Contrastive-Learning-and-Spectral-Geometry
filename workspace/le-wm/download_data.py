"""Data download helper for AutoDL bootstrap.

Supports Push-T (LeWM upstream) and PointMaze (best-effort).
Falls back to HuggingFace mirror if direct download fails.

Usage:
    python download_data.py --dataset pusht --out /workspace/stablewm_home/pusht_expert_train.h5
    python download_data.py --dataset pointmaze --out /workspace/stablewm_home/pointmaze_train.h5
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path


# Best-known URLs (verify these from the LeWM upstream README before depending)
DATA_URLS = {
    "pusht": [
        # Primary: LeWM upstream HF dataset (placeholder URL — verify on lucas-maes/le-wm)
        "https://huggingface.co/datasets/lucas-maes/lewm-pusht/resolve/main/pusht_expert_train.h5",
        # Fallback HF mirror (Chinese mainland mirror)
        "https://hf-mirror.com/datasets/lucas-maes/lewm-pusht/resolve/main/pusht_expert_train.h5",
    ],
    "pointmaze": [
        # PointMaze isn't in LeWM upstream; try generic locations
        "https://hf-mirror.com/datasets/farama-foundation/PointMaze-Medium/resolve/main/data.h5",
    ],
}


def download_with_progress(url, out_path):
    """Download a URL to out_path with simple progress indication."""
    def reporthook(blocknum, blocksize, totalsize):
        if totalsize > 0:
            pct = min(100, blocknum * blocksize * 100 // totalsize)
            print(f"\r  {pct}% ({blocknum * blocksize / 1e9:.2f} GB / {totalsize / 1e9:.2f} GB)", end="")
    try:
        urllib.request.urlretrieve(url, out_path, reporthook=reporthook)
        print()
        return True
    except Exception as e:
        print(f"\n  FAILED: {type(e).__name__}: {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["pusht", "pointmaze"])
    p.add_argument("--out", required=True, help="output path for the .h5 file")
    args = p.parse_args()

    if args.dataset not in DATA_URLS:
        print(f"FATAL: unknown dataset {args.dataset}")
        sys.exit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"[download] {out} already exists ({out.stat().st_size / 1e9:.2f} GB). Skipping.")
        return

    urls = DATA_URLS[args.dataset]
    for i, url in enumerate(urls):
        print(f"[download] [{i+1}/{len(urls)}] trying {url}")
        if download_with_progress(url, str(out)):
            print(f"[download] saved to {out} ({out.stat().st_size / 1e9:.2f} GB)")
            return

    print(f"[download] FATAL: all {len(urls)} URLs failed for {args.dataset}")
    print("  Manual fallback:")
    print("    (a) scp the h5 file from your local machine")
    print("    (b) for PointMaze: substitute TwoRoom by using data=tworoom in hydra overrides")
    sys.exit(1)


if __name__ == "__main__":
    main()
