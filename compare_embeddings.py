"""Intrinsically score embedding variants against lexical clinical equivalence.

Why this exists
---------------
Choosing an embedding source by training a Soft-CLIP model per variant is slow
and confounded: each training run adds optimisation noise, and the retrieval
metric it produces is several causal steps away from the thing being compared.

The soft targets use exactly ONE property of an embedding: does it assign high
similarity to reports that mean the same thing? That is measurable directly,
with no training, against ground truth already present in the data -- reports
whose text normalises to the same string are clinically equivalent by definition.

Ground truth uses EXACT normalised match, not token overlap. Radiology reports
differ by negation and laterality, which barely move token-overlap scores:
"small left pneumothorax" vs "NO small left pneumothorax" has Jaccard 0.71,
while the genuinely equivalent "No acute cardiopulmonary process" vs "No acute
cardiopulmonary abnormality is seen" scores 0.43. The two populations overlap,
so no overlap threshold separates them. Exact match has false negatives (misses
paraphrase) but essentially no false positives, which is the right trade here:
it still ranks variants correctly, it just compresses the AUROC range.

Usage
-----
    python compare_embeddings.py --tags biomedvlp_text_projection biomedvlp_text_cls
    python compare_embeddings.py --tags $(...) --out embedding_comparison.csv
"""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd
import torch

from mimic_clip.config import BASE_DATA_DIR, VAL_CSV_PATH, TRAIN_CSV_PATH, DEFAULT_BATCH_SIZE
from mimic_clip.metrics import build_text_group_ids

KNOWN_MODELS = ("bioclinicalbert", "gemma_embed", "biomedvlp")
KNOWN_FIELDS = ("findings_clean", "impression_clean", "text")
KNOWN_POOLING = ("projection", "native", "mean", "cls")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--sample-size", type=int, default=5000,
                   help="Rows to subsample. The SAME rows are used for every tag.")
    p.add_argument("--group-field", default="text",
                   help="Column whose near-duplicates define clinical equivalence. "
                        "Held fixed across all tags.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help="Training batch size, used to preview what the soft targets "
                        "would actually look like.")
    p.add_argument("--n-boot", type=int, default=20,
                   help="Bootstrap resamples for the AUROC confidence interval. 0 to skip.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    return p.parse_args()


# --------------------------------------------------------------------------
# Tag parsing -- lets the script find comparisons that differ in ONE factor.
# --------------------------------------------------------------------------
def parse_tag(tag: str) -> dict:
    rest = tag
    sent = rest.endswith("_sent")
    if sent:
        rest = rest[: -len("_sent")]
    model = next((m for m in KNOWN_MODELS if rest.startswith(m)), None)
    pooling = next((p for p in KNOWN_POOLING if rest.endswith(p)), None)
    field = None
    if model and pooling:
        middle = rest[len(model) + 1 : len(rest) - len(pooling) - 1]
        field = next((f for f in KNOWN_FIELDS if middle == f), middle or None)
    return {"model": model, "field": field, "pooling": pooling,
            "sent": "yes" if sent else "no"}


def _auroc(same: np.ndarray, diff: np.ndarray) -> float:
    """Mann-Whitney U -- exact, and cheaper than sorting the full score vector."""
    allv = np.concatenate([same, diff])
    ranks = allv.argsort().argsort().astype(np.float64) + 1
    n1, n2 = same.size, diff.size
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def _sparkline(values: np.ndarray, lo: float, hi: float, width: int = 28) -> str:
    blocks = " ▁▂▃▄▅▆▇█"
    counts, _ = np.histogram(values, bins=width, range=(lo, hi))
    if counts.max() == 0:
        return " " * width
    idx = np.ceil(counts / counts.max() * (len(blocks) - 1)).astype(int)
    return "".join(blocks[i] for i in idx)


def verdict(auroc: float) -> str:
    if auroc >= 0.90: return "strong"
    if auroc >= 0.80: return "good"
    if auroc >= 0.65: return "weak"
    if auroc >= 0.55: return "very weak"
    return "NO SIGNAL"


def soft_target_preview(emb: np.ndarray, batch_size: int, rng, n_batches: int = 20) -> dict:
    """What the soft targets would look like at training time.

    diag_mass is the mean probability the target puts on the TRUE pair. It is the
    quantity that makes alpha interpretable, and it depends on batch size, so it
    can only be previewed here, not read off the AUROC.
    """
    n = emb.shape[0]
    b = min(batch_size, n)
    masses_at_01, temps_for_half = [], []
    for _ in range(n_batches):
        idx = rng.choice(n, b, replace=False)
        sub = emb[idx]
        sim = sub @ sub.T

        e = np.exp((sim - sim.max(1, keepdims=True)) / 0.1)
        masses_at_01.append(np.mean(np.diag(e / e.sum(1, keepdims=True))))

        lo, hi = 1e-3, 2.0
        for _ in range(30):
            mid = (lo + hi) / 2
            e = np.exp((sim - sim.max(1, keepdims=True)) / mid)
            m = np.mean(np.diag(e / e.sum(1, keepdims=True)))
            if m < 0.5: hi = mid
            else: lo = mid
        temps_for_half.append((lo + hi) / 2)
    return {"diag_mass_at_t0.1": float(np.mean(masses_at_01)),
            "temp_for_diag_0.5": float(np.mean(temps_for_half))}


def score_embedding(emb: torch.Tensor, groups: np.ndarray, rng,
                    n_boot: int, batch_size: int) -> dict:
    e = torch.nn.functional.normalize(emb.float(), dim=-1).numpy()
    sim = e @ e.T
    n = sim.shape[0]

    off = ~np.eye(n, dtype=bool)
    same_m = (groups[:, None] == groups[None, :]) & off
    same_vals, diff_vals = sim[same_m], sim[off & ~same_m]
    if same_vals.size == 0:
        raise ValueError("No same-meaning pairs in the sample -- increase --sample-size.")

    auroc = _auroc(same_vals, diff_vals)

    # Bootstrap over ROWS (not pairs), so the dependency between pairs sharing a
    # row is respected. The similarity matrix is computed once and re-indexed.
    lo = hi = float("nan")
    if n_boot > 0:
        boots = []
        for _ in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            s = sim[np.ix_(idx, idx)]
            g = groups[idx]
            o = ~np.eye(len(idx), dtype=bool)
            sm = (g[:, None] == g[None, :]) & o
            if sm.sum() == 0:
                continue
            boots.append(_auroc(s[sm], s[o & ~sm]))
        if boots:
            lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    has_partner = same_m.any(axis=1)
    sim_nd = sim.copy()
    np.fill_diagonal(sim_nd, -np.inf)
    nn = sim_nd.argmax(axis=1)
    p1 = same_m[np.arange(n), nn][has_partner].mean()

    pooled_sd = np.sqrt((same_vals.var() + diff_vals.var()) / 2)
    out = {
        "pair_auroc": auroc, "ci_lo": lo, "ci_hi": hi,
        "p_at_1": float(p1),
        "separation": float(same_vals.mean() - diff_vals.mean()),
        "bimodality": float((same_vals.mean() - diff_vals.mean()) / max(pooled_sd, 1e-9)),
        "mean_same": float(same_vals.mean()), "mean_diff": float(diff_vals.mean()),
        "std_diff": float(diff_vals.std()), "dim": int(e.shape[1]),
        "verdict": verdict(auroc),
    }
    out.update(soft_target_preview(e, batch_size, rng))
    out["_same"] = same_vals
    out["_diff"] = diff_vals[rng.choice(diff_vals.size, min(200_000, diff_vals.size), replace=False)]
    return out


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    csv_path = VAL_CSV_PATH if args.split == "val" else TRAIN_CSV_PATH
    df = pd.read_csv(csv_path).fillna("")
    idx = (np.sort(rng.choice(len(df), args.sample_size, replace=False))
           if len(df) > args.sample_size else np.arange(len(df)))
    groups = build_text_group_ids(df[args.group_field])[idx]

    sizes = pd.Series(groups).value_counts()
    n_pairs = int((sizes * (sizes - 1) / 2).sum())

    print("=" * 96)
    print("EMBEDDING SCREEN -- intrinsic quality vs. lexical clinical equivalence")
    print("=" * 96)
    print(f"split            : {args.split}  ({csv_path})")
    print(f"rows scored      : {len(idx):,} of {len(df):,}")
    print(f"ground truth     : exact normalised match on {args.group_field!r}")
    print(f"groups           : {len(np.unique(groups)):,}  "
          f"({int((sizes > 1).sum()):,} with >1 member, largest = {int(sizes.max()):,})")
    print(f"same-meaning prs : {n_pairs:,}")
    if n_pairs < 100:
        print("\n  ** WARNING: too few same-meaning pairs. AUROC will be unstable.")
        print("     Rerun with --split train, or raise --sample-size. **")
    print()

    rows = []
    for tag in args.tags:
        path = os.path.join(BASE_DATA_DIR, f"{args.split}_{tag}_embeddings.pt")
        if not os.path.exists(path):
            print(f"  [skip] {tag}: not found"); continue
        emb = torch.load(path, map_location="cpu", weights_only=True)
        if emb.shape[0] != len(df):
            print(f"  [skip] {tag}: {emb.shape[0]:,} rows vs CSV {len(df):,} -- regenerate"); continue
        rows.append({"tag": tag, **parse_tag(tag),
                     **score_embedding(emb[idx], groups, rng, args.n_boot, args.batch_size)})
        print(f"  [ok]   {tag}")
    if not rows:
        raise SystemExit("No embeddings could be scored.")

    out = pd.DataFrame(rows).sort_values("pair_auroc", ascending=False).reset_index(drop=True)

    # ---- 1. main table ----------------------------------------------------
    print("\n" + "=" * 96)
    print("1. RANKING   (pair_auroc = P(same-meaning pair scores above different-meaning pair))")
    print("=" * 96)
    hdr = f"{'tag':<38} {'AUROC':>6} {'95% CI':>15} {'P@1':>6} {'bimod':>6}  verdict"
    print(hdr); print("-" * len(hdr))
    for _, r in out.iterrows():
        ci = (f"[{r.ci_lo:.3f},{r.ci_hi:.3f}]" if np.isfinite(r.ci_lo) else "--")
        print(f"{r.tag:<38} {r.pair_auroc:>6.3f} {ci:>15} {r.p_at_1:>6.3f} "
              f"{r.bimodality:>6.2f}  {r.verdict}")
    print("\n  0.50 = cannot distinguish same-meaning from different-meaning reports.")
    print("  Overlapping CIs mean the difference is not resolvable at this sample size.")

    # ---- 2. distribution shape -------------------------------------------
    print("\n" + "=" * 96)
    print("2. SIMILARITY DISTRIBUTIONS   (want: same-meaning shifted right of different-meaning)")
    print("=" * 96)
    for _, r in out.iterrows():
        print(f"\n  {r.tag}")
        print(f"    same-meaning |{_sparkline(r._same, -1, 1)}| mean {r.mean_same:+.3f}")
        print(f"    different    |{_sparkline(r._diff, -1, 1)}| mean {r.mean_diff:+.3f} "
              f"(sd {r.std_diff:.3f})")
    print("\n  Scale is cosine -1 .. +1. A single overlapping blob = little usable structure.")

    # ---- 3. controlled one-factor comparisons -----------------------------
    print("\n" + "=" * 96)
    print("3. CONTROLLED COMPARISONS   (pairs differing in exactly ONE factor)")
    print("=" * 96)
    factors = ["model", "field", "pooling", "sent"]
    found = False
    for a, b in itertools.combinations(out.itertuples(), 2):
        differ = [f for f in factors if getattr(a, f) != getattr(b, f)]
        if len(differ) != 1:
            continue
        found = True
        f = differ[0]
        hi_, lo_ = (a, b) if a.pair_auroc >= b.pair_auroc else (b, a)
        overlap = (np.isfinite(hi_.ci_lo) and hi_.ci_lo <= lo_.ci_hi)
        print(f"\n  {f}: {getattr(hi_, f)} vs {getattr(lo_, f)}   "
              f"(everything else held fixed)")
        print(f"    {hi_.tag:<40} {hi_.pair_auroc:.3f}")
        print(f"    {lo_.tag:<40} {lo_.pair_auroc:.3f}")
        print(f"    delta = {hi_.pair_auroc - lo_.pair_auroc:+.3f}"
              f"{'   (CIs overlap -- not resolvable)' if overlap else ''}")
    if not found:
        print("\n  None. Every pair of tags differs in more than one factor, so no")
        print("  comparison here is controlled. Generate the missing cells before")
        print("  attributing any difference to a single cause.")

    # ---- 4. what this means for training ----------------------------------
    print("\n" + "=" * 96)
    print(f"4. SOFT-TARGET PREVIEW at batch size {args.batch_size}")
    print("=" * 96)
    hdr2 = f"{'tag':<38} {'diag_mass @ t=0.1':>18} {'t for diag_mass 0.5':>21}"
    print(hdr2); print("-" * len(hdr2))
    for _, r in out.iterrows():
        print(f"{r.tag:<38} {r['diag_mass_at_t0.1']:>18.3f} {r['temp_for_diag_0.5']:>21.4f}")
    print("\n  diag_mass = probability the soft target puts on the TRUE pair.")
    print("  At t=0.1 (the original setting) this is typically ~0.2, i.e. 80% of the")
    print("  target mass sits on other reports. Pass --calibrate-temp 0.5 to")
    print("  train_soft_clip.py, or set --soft-temp to the value in the last column.")

    # ---- 5. recommendation ------------------------------------------------
    best = out.iloc[0]
    print("\n" + "=" * 96)
    print("5. WHAT TO DO NEXT")
    print("=" * 96)
    if best.pair_auroc < 0.55:
        print(f"  Best variant is {best.tag} at AUROC {best.pair_auroc:.3f} -- no signal.")
        print("  No choice of embedding, alpha or temperature will make these soft")
        print("  targets informative. This IS the result: report it, show section 2's")
        print("  distributions as evidence, and skip the sweep.")
    else:
        print(f"  Winner: {best.tag}   (AUROC {best.pair_auroc:.3f}, {best.verdict})")
        print(f"    ./bas.sh promote {best.tag}")
        print(f"    ./bas.sh baseline")
        print(f"    ./bas.sh alpha0 --embeddings-tag {best.tag}")
        print(f"        ^ must reproduce baseline exactly before you sweep anything")
        print(f"    ./bas.sh soft --embeddings-tag {best.tag} --alpha 0.1 --calibrate-temp 0.5")
        worst = out.iloc[-1]
        if len(out) > 1 and worst.pair_auroc < best.pair_auroc - 0.05:
            print(f"\n  Also train {worst.tag} (AUROC {worst.pair_auroc:.3f}) as a contrast:")
            print("  if intrinsic quality predicts downstream retrieval, that is a result;")
            print("  if it does not, that is a more interesting one.")

    if args.out:
        out.drop(columns=["_same", "_diff"]).to_csv(args.out, index=False)
        print(f"\n  Wrote {args.out}")


if __name__ == "__main__":
    main()
