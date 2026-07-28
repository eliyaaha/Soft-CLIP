# MIMIC-CXR CLIP Ablations

Fine-tune CLIP on MIMIC-CXR with two training objectives — a hard study-level
contrastive loss and a soft KL-supervised hybrid loss — and run ablations
over text fields, BERT embedding models, and soft-loss hyperparameters.

## Repo layout

```
.
├── preprocess.py            # raw MIMIC CSVs → processed CSVs   (step 1)
├── create_embeddings.py     # processed CSVs → BERT .pt files   (step 2)
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

Tokenizes the chosen text column with a BERT-family model and saves
L2-normalized CLS embeddings as a `.pt` tensor. **Each (model, field)
combination writes its own file** — nothing is overwritten silently, so you
can build a library of embeddings for soft-CLIP ablations.

```bash
python create_embeddings.py [--model {biomedvlp,bioclinicalbert}]
                             [--field {text,findings_clean,impression_clean}]
                             [--batch-size N] [--max-length N]
                             [--overwrite] [--list]
```

Flag | Default | Notes
---|---|---
`--model` | `biomedvlp` | `biomedvlp` → `microsoft/BiomedVLP-CXR-BERT-specialized`; `bioclinicalbert` → `emilyalsentzer/Bio_ClinicalBERT`; `gemma_embed` → `google/embeddinggemma-300m` (general-purpose embedding model, no clinical fine-tuning)
`--field` | `text` | Raw report text. Other options embed just the findings or impression section.
`--batch-size` | `64` | Tokenizer/forward batch.
`--max-length` | `128` | Tokenizer/sequence truncation length.
`--overwrite` | off | Recompute even if the output `.pt` exists.
`--list` | off | Print existing `*_embeddings.pt` files under `BASE_DATA_DIR` and exit.

> `gemma_embed` uses `sentence-transformers`'s `encode_document`, which handles tokenization, left-padding, causal pooling, and normalization internally — a different code path from the CLS-token pooling used for the two BERT models. Requires the `sentence-transformers` package, and an `HF_TOKEN` in `.env` if the model is gated on Hugging Face.

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
| `--batch-size` | `256` | |
| `--lr` | `5e-6` | |
| `--weight-decay` | `0.2` | |
| `--epochs` | `10` | |
| `--patience` | `2` | Early stopping. |
| `--run-name` | auto | Auto name encodes `{loss}_{field}` when omitted. |

Checkpoints go to `checkpoints/hard/{run_name}/`.

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
precomputed BERT similarities. **The choice of embeddings is independent
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
`--text-similarity-weight` | `0.5` | Only used in threshold mode. Weight on text similarity when combining it with image similarity: `combined = w * text_sim + (1-w) * image_sim`. Set to `1.0` to threshold on text similarity alone.
`--batch-size` | `128` |
`--lr` | `1e-6` |
`--epochs` / `--patience` | `10` / `2` |
`--run-name` | auto | Auto name encodes `loss_field_tag_alpha_temp[_k]`.

Startup validates that the resolved CSV and `.pt` files exist; missing files
raise `FileNotFoundError` with the exact command to run.

Checkpoints go to `checkpoints/soft/{run_name}/`.

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

Both training scripts use the same retrieval evaluation:

- **Image → Text** and **Text → Image** at the study level
- Recall@1, Recall@5, Recall@10
- Median Rank
- MRR (Mean Reciprocal Rank)

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
