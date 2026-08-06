"""Find and characterise rows whose image file is missing on disk.

Motivation
----------
build_dataloaders() now drops rows with missing images up front (previously a
silent fallback in the Dataset walked to a neighbouring row instead, which
desynchronised features from labels -- see mimic_clip/data.py). That fix
surfaced that ~30% of both splits are missing their image file. Before
trusting any retrieval numbers, it's worth knowing WHY:

  - Missing files scattered randomly across subjects/views -> likely benign,
    just document the true candidate-pool size.
  - Missing files clustered in specific path prefixes, or entire parent
    directories absent -> likely an incomplete download or a path/mount
    misconfiguration, and current results may be biased by whichever subset
    of the data happens to be present.

This script writes the missing rows to a CSV per split and prints a few
breakdowns to distinguish those cases quickly.

Usage
-----
    python find_missing_images.py --split val
    python find_missing_images.py --split train
    python find_missing_images.py --split both --sample 15
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import pandas as pd

from mimic_clip.config import IMAGE_DIR, TRAIN_CSV_PATH, VAL_CSV_PATH


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=("train", "val", "both"), default="both")
    p.add_argument("--sample", type=int, default=10,
                   help="How many example missing paths to print.")
    p.add_argument("--depth", type=int, default=2,
                   help="How many leading path components to group by when "
                        "looking for clustering (e.g. 2 -> 'files/p10').")
    p.add_argument("--out-dir", default=".",
                   help="Where to write missing_images_{split}.csv.")
    return p.parse_args()


def _prefix(rel_path: str, depth: int) -> str:
    parts = str(rel_path).replace("\\", "/").split("/")
    return "/".join(parts[:depth]) if len(parts) >= depth else str(rel_path)


def analyze_split(csv_path: str, name: str, depth: int, sample: int, out_dir: str) -> None:
    if not os.path.exists(csv_path):
        print(f"[{name}] processed CSV not found at {csv_path}, skipping.")
        return

    df = pd.read_csv(csv_path).fillna("")
    if "image" not in df.columns:
        raise ValueError(f"[{name}] CSV has no 'image' column: {list(df.columns)}")

    full_paths = df["image"].apply(lambda p: os.path.join(IMAGE_DIR, str(p)))
    exists = full_paths.apply(os.path.exists)
    missing = df[~exists].copy()
    missing["full_path"] = full_paths[~exists]
    missing["parent_dir_exists"] = missing["full_path"].apply(
        lambda p: os.path.isdir(os.path.dirname(p))
    )
    missing["prefix"] = missing["image"].apply(lambda p: _prefix(p, depth))

    n_total, n_missing = len(df), len(missing)
    pct = 100 * n_missing / max(n_total, 1)

    print("\n" + "=" * 88)
    print(f"{name.upper()}  ({csv_path})")
    print("=" * 88)
    print(f"IMAGE_DIR       : {IMAGE_DIR}")
    print(f"total rows      : {n_total:,}")
    print(f"missing rows    : {n_missing:,}  ({pct:.1f}%)")

    if n_missing == 0:
        print("No missing files. Nothing further to report.")
        return

    # Directory present but file absent (naming/extension mismatch) vs whole
    # directory absent (incomplete download / wrong mount).
    dir_missing = int((~missing["parent_dir_exists"]).sum())
    file_only_missing = n_missing - dir_missing
    print(f"\n  parent directory absent entirely : {dir_missing:,} "
          f"({100*dir_missing/n_missing:.1f}% of missing)")
    print(f"    -> suggests an incomplete download, a shard never fetched, "
          f"or IMAGE_DIR pointing at the wrong location.")
    print(f"  parent directory exists, file doesn't : {file_only_missing:,} "
          f"({100*file_only_missing/n_missing:.1f}% of missing)")
    print(f"    -> suggests a filename/extension/case mismatch between the "
          f"CSV and what's on disk.")

    # Is missingness concentrated in specific path prefixes, or spread evenly?
    total_by_prefix = Counter(df["image"].apply(lambda p: _prefix(p, depth)))
    missing_by_prefix = Counter(missing["prefix"])
    print(f"\n  Missingness by path prefix (depth={depth}), top 15 by missing count:")
    print(f"  {'prefix':<40} {'missing':>8} {'of total in prefix':>10} {'missing %':>10}")
    for prefix, n_miss in missing_by_prefix.most_common(15):
        n_tot = total_by_prefix[prefix]
        print(f"  {prefix:<40} {n_miss:>8,} {n_tot:>18,} {100*n_miss/n_tot:>9.1f}%")
    n_prefixes_all_missing = sum(
        1 for pfx, n_tot in total_by_prefix.items() if missing_by_prefix.get(pfx, 0) == n_tot
    )
    print(f"\n  Prefixes that are 100% missing: {n_prefixes_all_missing} "
          f"of {len(total_by_prefix)} distinct prefixes at this depth.")
    print("    A handful of prefixes at or near 100% missing (rest near 0%) means")
    print("    specific shards/folders are absent -- a download/mount problem.")
    print("    Missingness spread thinly across most prefixes instead means it's")
    print("    closer to random per-file gaps.")

    if "view" in missing.columns:
        vc_missing = missing["view"].value_counts()
        vc_total = df["view"].value_counts()
        print("\n  Missingness by view:")
        for view in vc_total.index:
            m = vc_missing.get(view, 0)
            print(f"    {str(view):<12} {m:>7,} / {vc_total[view]:>7,} missing "
                  f"({100*m/vc_total[view]:.1f}%)")

    if sample > 0:
        print(f"\n  Example missing paths (up to {sample}):")
        for p in missing["full_path"].head(sample):
            print(f"    {p}")

    out_path = os.path.join(out_dir, f"missing_images_{name}.csv")
    cols = [c for c in ["subject_id", "study_id", "image", "full_path",
                        "parent_dir_exists", "view"] if c in missing.columns]
    missing[cols].to_csv(out_path, index=False)
    print(f"\n  Wrote {n_missing:,} rows to {out_path}")


def main() -> None:
    args = parse_args()
    splits = {"train": TRAIN_CSV_PATH, "val": VAL_CSV_PATH}
    targets = splits if args.split == "both" else {args.split: splits[args.split]}
    for name, path in targets.items():
        analyze_split(path, name, args.depth, args.sample, args.out_dir)


if __name__ == "__main__":
    main()
