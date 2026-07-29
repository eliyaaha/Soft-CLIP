# MIMIC-CXR CLIP Ablations

Fine-tune CLIP on MIMIC-CXR with two training objectives — a hard study-level
contrastive loss and a soft KL-supervised hybrid loss — and run ablations
over text fields, embedding models, and soft-loss hyperparameters.

## Repo layout

```
.
├── preprocess.py            # raw MIMIC CSVs → processed CSVs   (step 1)
├── create_embeddings.py     # processed CSVs → embedding .pt files   (step 2)
├── train_baseline.py        # hard CLIP fine-tuning             (step 3a)
├── train_soft_clip.py       # soft-CLIP fine-tuning             (step 3b)
└── mimic_clip/              # shared library used by both train scripts
    ├── config.py            # paths + ExperimentConfig
    ├── data.py              # unified MimicCLIPDataset + dataloaders
    ├── clip_utils.py        # load_clip + clip_features
    ├── losses.py            # hard study-level + soft hybrid loss
    ├── metrics.py           # shared retrieval evaluation
    └── trainer.py           # train/val loop + early stopping
```

## Pipeline order

The pipeline is strictly sequential — each step writes outputs the next step
reads:

```
preprocess.py
     │   produces  mimic_cxr_processed_{train,validate}.csv
     ▼
create_embeddings.py            (only needed for soft-CLIP)
     │   produces  {train,val}_{model}_{field}_embeddings.pt
     ▼
train_baseline.py   /   train_soft_clip.py
     │   produces  checkpoints/{hard|soft}/{run_name}/
     ▼
train_*.py --mode eval
     │   prints Recall@1/5/10, Median Rank, MRR
```

- Preprocess **once**; the train and embedding scripts both depend on its
  output CSVs.
- After preprocess you can run `create_embeddings.py` with any combination
  of `--model` / `--field`; files accumulate side-by-side.
- Hard baseline does not need embeddings; soft-CLIP does.

## 1. Preprocessing — `preprocess.py`

No CLI args. Reads raw augmented CSVs from `BASE_DATA_DIR`, groups by
`study_id`, explodes to one row per image, extracts `findings_clean` and
`impression_clean` from the report text, drops empty rows.

```bash
python preprocess.py
```

Outputs: `mimic_cxr_processed_train.csv`, `mimic_cxr_processed_validate.csv`.

## 2. Text embeddings — `create_embeddings.py`

Embeds the chosen text column with a BERT-family or Gemma-family model and
saves normalized embeddings as a `.pt` tensor. **Each (model, field)
combination writes its own file** — nothing is overwritten silently, so you
can build a library of embeddings for soft-CLIP ablations.

```bash
python create_embeddings.py [--model {biomedvlp,bioclinicalbert,gemma_embed}]
                             [--field {text,findings_clean,impression_clean}]
                             [--batch-size N] [--max-length N]
                             [--overwrite] [--list]
```

Flag | Default | Notes
---|---|---
`--model` | `biomedvlp` | `biomedvlp` → `microsoft/BiomedVLP-CXR-BERT-specialized`; `bioclinicalbert` → `emilyalsentzer/Bio_ClinicalBERT`; `gemma_embed` → `google/embeddinggemma-300m` (general-purpose embedding model, no clinical fine-tuning)
`--field` | `text` | Raw report text. Other options embed just the findings or impression section.
`--pooling` | `auto` | `auto` = `projection` for biomedvlp, `mean` otherwise. `cls` reproduces the original (incorrect) behaviour for comparison. Ignored for `gemma_embed`.
`--sentence-level` | off | Embed each sentence separately and average, instead of truncating the whole report. Recommended for `--field text`.
`--tag-suffix` | `""` | Appended to output filenames, so pooling variants can sit side by side.
`--batch-size` | `64` | Tokenizer/forward batch.
`--max-length` | `128` | Tokenizer/sequence truncation length.
`--overwrite` | off | Recompute even if the output `.pt` exists.
`--list` | off | Print existing `*_embeddings.pt` files under `BASE_DATA_DIR` and exit.

> `gemma_embed` uses `sentence-transformers`'s `encode_document`, which handles tokenization, left-padding, causal pooling, and normalization internally — a different code path from the two BERT models. Requires the `sentence-transformers` package, and an `HF_TOKEN` in `.env` if the model is gated on Hugging Face.

> **Pooling matters more than the model choice.** Raw `last_hidden_state[:, 0, :]`
> is the wrong representation for both BERT encoders: CXR-BERT ships a projection
> head trained specifically to make CLS embeddings comparable, and
> Bio_ClinicalBERT has no sentence-level objective at all — its CLS vector is
> anisotropic, so every pair lands in a narrow similarity band regardless of
> content. **Any `.pt` file generated before this change used raw CLS; regenerate
> with `--overwrite`.**
>
> Afterwards run `analyze_similarity_threshold.py --tag <tag>` and look at the
> histogram. A working clinical signal on MIMIC-CXR is **bimodal** — a dense
> cluster of near-identical normal studies plus a long tail of distinct pathology.
> A wide unimodal blob means the soft targets built from these embeddings are
> close to noise, and no amount of `--alpha` tuning will fix that.

> **Pooling is part of the filename, on purpose.** Tags are
> `{model}_{field}_{pooling}[_sent]`. `--pooling auto` picks each model's *native*
> representation (`projection` for CXR-BERT, `mean` for Bio_ClinicalBERT), which
> answers "which pipeline works best" — but it **confounds a model-vs-model
> comparison**, because pooling varies alongside the model. For a controlled
> comparison, pass the same `--pooling` to every model. Encoding it in the tag
> means you cannot accidentally compare across pooling without seeing it.

> **CXR-BERT is a sentence encoder.** Its model card documents it for "radiological
> sentence embeddings", and its reported inputs average ~58 tokens. A full
> findings+impression report is well outside that regime and gets silently
> truncated at `--max-length`. `--sentence-level` embeds each sentence inside the
> length the encoder was trained for and averages, covering the whole report.
> Worth generating both and comparing histograms:
>
> ```bash
> python create_embeddings.py --model biomedvlp --field text --overwrite
> python create_embeddings.py --model biomedvlp --field text --sentence-level \
>     --tag-suffix sent --overwrite
> python analyze_similarity_threshold.py --tag biomedvlp_text
> python analyze_similarity_threshold.py --tag biomedvlp_text_sent
> ```

Output file names:

```
train_{model_slug}_{field}_embeddings.pt
val_{model_slug}_{field}_embeddings.pt
```

The string `{model_slug}_{field}` is the **embeddings tag** that
`train_soft_clip.py` consumes via `--embeddings-tag`.

Examples:

```bash
# Default — BiomedVLP over the raw `text` column
python create_embeddings.py

# Findings-only ablation, same model
python create_embeddings.py --field findings_clean

# Compare BERT models
python create_embeddings.py --model bioclinicalbert --field impression_clean

# Gemma embeddings (general-purpose, no clinical fine-tuning)
python create_embeddings.py --model gemma_embed --field text

# List what you already have
python create_embeddings.py --list
```

## 2b. Choosing an embedding source — `compare_embeddings.py`

Selecting an embedding by training one Soft-CLIP model per variant is slow and
confounded: each run adds optimisation noise, and the retrieval metric it
produces is several causal steps away from the thing being compared.

The soft targets use exactly one property of an embedding — *does it assign high
similarity to reports that mean the same thing?* That is measurable directly, with
no training, against ground truth already present in the data: reports whose text
normalises to the same string are clinically equivalent by definition.

```bash
# Controlled — pooling held fixed, only the model varies
python compare_embeddings.py --tags \
    biomedvlp_text_mean bioclinicalbert_text_mean gemma_embed_text_native

# Controlled — model held fixed, only pooling varies
python compare_embeddings.py --tags \
    biomedvlp_text_cls biomedvlp_text_mean biomedvlp_text_projection \
    biomedvlp_text_projection_sent
```

Metric | Meaning
---|---
`pair_auroc` | P(a same-meaning pair scores above a different-meaning pair). **0.5 = no signal.**
`p_at_1` | Fraction of anchors whose nearest neighbour means the same thing.
`separation` | `mean(same) − mean(different)`, in cosine units.
`bimodality` | Standardised separation (Cohen's *d*) — how distinct the two populations are.

Every tag is scored on the **same subsampled rows** against the **same ground
truth**, so the comparison is paired and training-free. Run the grid here, then
train Soft-CLIP only with the winner (plus one loser, as a contrast worth
reporting).

If the best variant sits near `pair_auroc` 0.5, the soft targets are noise no
matter which one you pick — and no `alpha` will fix that. That is a publishable
finding on its own, and it costs no GPU time to establish.

## 3a. Hard baseline — `train_baseline.py`

Fine-tunes CLIP with a study-level supervised contrastive loss (any
text/image from the same `study_id` is a valid positive in the batch). No
precomputed embeddings required.

```bash
python train_baseline.py [--mode {train,eval,both}] [--checkpoint PATH]
                          [--text-field {text,findings_clean,impression_clean}]
                          [--batch-size N] [--lr LR] [--weight-decay WD]
                          [--epochs N] [--patience N] [--num-workers N]
                          [--run-name NAME]
```

| Flag | Default | Notes |
|------|---------|-------|
| `--mode` | `both` | `train`, `eval`, or `both`. |
| `--checkpoint` | – | Required for `--mode eval`. Loaded with `from_pretrained`. |
| `--text-field` | `text` | CLIP text input column. |
| `--batch-size` | `256` | From `mimic_clip.config.DEFAULT_BATCH_SIZE` — shared with soft-CLIP. |
| `--lr` | `5e-6` | From `mimic_clip.config.DEFAULT_LR` — shared with soft-CLIP. |
| `--weight-decay` | `0.2` | |
| `--epochs` | `10` | |
| `--patience` | `2` | Early stopping, on validation **hard** loss. |
| `--seed` | `42` | Run ≥3 seeds and report mean ± std. |
| `--run-name` | auto | Auto name encodes `{loss}_{field}` (plus `_s{seed}` if non-default). |

Checkpoints go to `checkpoints/hard/{run_name}/`, together with a
`run_config.json` recording exactly what the run was trained with.

> **Optimisation defaults are shared between the two arms** via
> `mimic_clip/config.py`. They previously differed (soft-CLIP defaulted to
> batch 128 / lr 1e-6 against the baseline's 256 / 5e-6, and no run command
> overrode them), which meant every soft-CLIP result was produced with half the
> in-batch negatives and a 5× smaller step than the baseline it was compared
> against. Change them in one place or not at all.

Examples:

```bash
# Default fine-tune + eval on text
python train_baseline.py --mode both

# Same loss, different text input
python train_baseline.py --mode train --text-field findings_clean --run-name hard_find

# Eval a saved checkpoint
python train_baseline.py --mode eval --checkpoint checkpoints/hard/hard_impression_clean
```

## 3b. Soft-CLIP — `train_soft_clip.py`

Hybrid loss: hard diagonal cross-entropy + soft KL targets built from
precomputed similarities. **The choice of embeddings is independent
of the CLIP training hyperparameters**, so you can compare embedding
sources without re-embedding or re-training in lock-step.

Two mutually exclusive pseudo-positive selection strategies are available: a **fixed top-K** (`--soft-top-k`), and a **similarity threshold** (`--soft-threshold`) where the number of retained neighbors varies per row based on local similarity density. If `--soft-threshold` is set, `--soft-top-k` is ignored.

```bash
python train_soft_clip.py [--mode {train,eval,both}] [--checkpoint PATH]
                           [--text-field {text,findings_clean,impression_clean}]
                           [--embeddings-tag TAG]
                           [--train-embeddings PATH] [--val-embeddings PATH]
                           [--alpha A] [--soft-temp T]
                           [--soft-top-k K]
                           [--soft-threshold TAU] [--text-similarity-weight W]
                           [--batch-size N] [--lr LR] [--weight-decay WD]
                           [--epochs N] [--patience N] [--num-workers N]
                           [--run-name NAME]
```

Flag | Default | Notes
---|---|---
`--mode` | `both` |
`--checkpoint` | – | Required for `--mode eval`.
`--text-field` | `text` | Text column CLIP itself sees.
`--embeddings-tag` | `biomedvlp_text` | Looks up `{train,val}_{tag}_embeddings.pt` under `BASE_DATA_DIR`.
`--train-embeddings`, `--val-embeddings` | – | Override the tag with explicit paths.
`--alpha` | `0.5` | Soft-loss weight; `(1 - alpha)` weights the hard term.
`--soft-temp` | `0.1` | Temperature for softmax over semantic similarities (top-K mode only).
`--soft-top-k` | `None` | If set, only the K largest similarities per row form the soft target distribution (fixed-K neighbor ablation). Ignored if `--soft-threshold` is set.
`--soft-threshold` | `None` | If set, keeps only pairs whose combined similarity exceeds this value, rescales as `(s - threshold)/(1 - threshold)`, and row-normalizes — a data-dependent (dynamic-K) alternative to `--soft-top-k`. Must be in `[-1, 1)`.
`--text-similarity-weight` | `1.0` | Only used in threshold mode. Weight on text similarity when combining with image similarity: `combined = w * text_sim + (1-w) * image_sim`. Defaults to pure text — mixing in the model's own image similarity is self-reinforcing.
`--calibrate-temp` | `None` | Solve for the temperature that puts this much target mass on the true pair (e.g. `0.5`) instead of using `--soft-temp` directly. See the note below.
`--shuffle-embeddings` | off | **Control run.** Permutes the semantic embeddings so soft targets carry no image–text correspondence.
`--batch-size` | `256` | Shared with the baseline via `mimic_clip.config`.
`--lr` | `5e-6` | Shared with the baseline via `mimic_clip.config`.
`--epochs` / `--patience` | `10` / `2` |
`--seed` | `42` | Run ≥3 seeds and report mean ± std.
`--early-stop-metric` | `hard` | Which validation quantity early stopping monitors. `hard` is defined identically for both arms, so checkpoints stay comparable across `alpha`. `total` restores the old behaviour of monitoring the blended loss.
`--run-name` | auto | Auto name encodes `loss_field_tag_alpha_temp[_k][_thr][_cal][_shuffled][_seed]`.

Startup validates that the resolved CSV and `.pt` files exist; missing files
raise `FileNotFoundError` with the exact command to run.

Checkpoints go to `checkpoints/soft/{run_name}/`, together with a
`run_config.json` that evaluation reads back.

### Three things worth knowing before sweeping `alpha`

**1. `soft_temp` matters more than `alpha`, and `0.1` is close to the worst
setting.** With L2-normalised embeddings the diagonal is exactly 1.0, so for a
similarity distribution with mean ≈ 0.48 / std ≈ 0.19 at batch size 128:

| `soft_temp` | mass the soft target puts on the true pair |
|---|---|
| 0.03 | 0.66 |
| 0.05 | 0.50 |
| **0.10** | **0.21** ← the original setting |
| 0.20 | 0.06 |

At 0.10 the true pair is barely preferred over an arbitrary other report, so
sweeping `alpha` at that temperature really just sweeps *how much label
corruption to apply*. `--calibrate-temp 0.5` fixes the smoothing to something
interpretable so `alpha` means the same thing across embedding sources and batch
sizes. Every run prints `diag_mass` per epoch — report it alongside your results.

**2. `--alpha 0.0` is a strict control.** The hybrid loss reduces exactly to
`study_level_contrastive_loss`, so an `alpha=0` run must reproduce
`train_baseline.py` at the same seed, batch size and lr. If it doesn't, there's a
pipeline bug and no soft-CLIP number is attributable. Include `alpha=0` as the
left endpoint of the sweep, and sample the low end densely
(`{0, 0.05, 0.1, 0.2, 0.35, 0.5}`) — an interior optimum at 0.1 is invisible on a
`{0.3, 0.5, 0.7}` grid.

**3. `--shuffle-embeddings` is the decisive experiment, and costs one run.**
If the shuffled control scores the same as the real run, the soft targets carry
no clinical information and any gain is generic label smoothing rather than
semantics. Without it, a small win at low `alpha` cannot be attributed.

Examples:

```bash
# Default — soft-CLIP with BiomedVLP text embeddings
python train_soft_clip.py --mode both

# Compare embedding models with same loss params
python train_soft_clip.py --mode train --embeddings-tag biomedvlp_text \
    --run-name soft_biomedvlp_text
python train_soft_clip.py --mode train --embeddings-tag bioclinicalbert_text \
    --run-name soft_bioclin_text
python train_soft_clip.py --mode train --embeddings-tag gemma_embed_text \
    --run-name soft_gemma_text

# K-neighbor ablation (fixed top-K)
python train_soft_clip.py --mode train --embeddings-tag biomedvlp_text \
    --alpha 0.3 --soft-top-k 5 --soft-temp 0.1 \
    --run-name soft_biomedvlp_a03_k5
 
# Threshold-based pseudo-positive selection (dynamic K), text-only similarity
python train_soft_clip.py --mode train --embeddings-tag biomedvlp_text \
    --soft-threshold 0.83 --text-similarity-weight 1.0 \
    --run-name soft_biomedvlp_thresh083

# Custom paths (e.g. embeddings in a different directory)
python train_soft_clip.py --mode train \
    --train-embeddings /path/to/train_xxx_embeddings.pt \
    --val-embeddings   /path/to/val_xxx_embeddings.pt

# Eval a saved checkpoint
python train_soft_clip.py --mode eval --checkpoint checkpoints/soft/soft_biomedvlp_text
```

## Evaluation metrics

Both training scripts use the same retrieval evaluation — Recall@1/5/10, Median
Rank and MRR, in both directions — reported under **four groupings** of the same
similarity matrix:

Grouping | A retrieval counts as correct when… | Purpose
---|---|---
`exact` | the retrieved item shares the query's `study_id` | the original metric
`clinical` | the retrieved report is a near-duplicate of the query's | clinical equivalence, no labels needed
`exact_dedup` | `study_id` matches, against a pool with duplicate reports removed | removes the redundancy ceiling
`subject` | the retrieved item shares the query's `subject_id` | nuisance probe: how much patient identity survives

**Why `exact` alone is not enough.** MIMIC-CXR contains thousands of studies whose
reports are verbatim identical ("No acute cardiopulmonary abnormality"). Under
`exact`, retrieving one of those for another is scored as an error even though it
is clinically correct. The only way to win on such pairs is to encode nuisance
variation — anatomy, positioning, exposure — which is exactly what soft
supervision is designed to suppress. So `exact` cannot distinguish "the soft
targets are noise" from "the soft targets worked as intended".

`clinical` groups are built by normalising report text (lowercase, strip
punctuation, collapse whitespace) and matching exactly. Two studies whose reports
normalise to the same string are clinically equivalent *by definition* — this is
lexical ground truth, not a model output, so using it to score the models under
comparison is not circular. It needs no CheXpert labels and no PhysioNet access.

**The signature of a method that works as intended: `exact` down, `clinical` up,
`subject` down.** Report all four; any one alone is uninterpretable.

The implementation lives in
[`mimic_clip/metrics.py`](mimic_clip/metrics.py) (`run_retrieval_eval`).
Run `train_*.py --mode eval --checkpoint <dir>` at any time to get a fresh
report on a saved checkpoint without retraining.

## Typical ablation workflow

```bash
# Step 1 — once
python preprocess.py

# Step 2 — generate every embedding variant you want to compare
python create_embeddings.py --model biomedvlp        --field text
python create_embeddings.py --model biomedvlp        --field findings_clean
python create_embeddings.py --model biomedvlp        --field impression_clean
python create_embeddings.py --model bioclinicalbert  --field text
python create_embeddings.py --model gemma_embed      --field text

# Step 3 — train soft-CLIP with each, holding loss params fixed
python train_soft_clip.py --mode train --embeddings-tag biomedvlp_text         --run-name cmp_biomedvlp_text
python train_soft_clip.py --mode train --embeddings-tag biomedvlp_findings_clean --run-name cmp_biomedvlp_find
python train_soft_clip.py --mode train --embeddings-tag bioclinicalbert_text   --run-name cmp_bioclin_text
python train_soft_clip.py --mode train --embeddings-tag gemma_embed_text        --run-name cmp_gemma_text

# Step 4 — re-eval anytime
python train_soft_clip.py --mode eval --checkpoint checkpoints/soft/cmp_biomedvlp_text
```

Loss-parameter ablations with the same embeddings:

```bash
for ALPHA in 0.3 0.5 0.7; do
  for K in 3 5 10; do
    python train_soft_clip.py --mode train \
        --embeddings-tag biomedvlp_text \
        --alpha $ALPHA --soft-top-k $K \
        --run-name soft_a${ALPHA}_k${K}
  done
done

# Threshold sweep
for TAU in 0.74 0.83 0.89; do
  python train_soft_clip.py --mode train \
      --embeddings-tag biomedvlp_text \
      --soft-threshold $TAU --text-similarity-weight 1.0 \
      --run-name soft_thresh${TAU}
done
```

## Paths and environment

* `BASE_DATA_DIR` is defined in `preprocess.py` and reused everywhere via `mimic_clip.config`. Override it there if your data lives elsewhere.
* `IMAGE_DIR` and the processed CSV paths derive from `BASE_DATA_DIR`.
* `HF_TOKEN` is read from `.env` (used by `create_embeddings.py`, required for gated Hugging Face models).
* `sentence-transformers` is required for `--model gemma_embed` in `create_embeddings.py`.
* Checkpoints land under `./checkpoints/{hard|soft}/{run_name}/`.
