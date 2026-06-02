# -*- coding: utf-8 -*-


import os
import ast
import re
import time
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# Define paths for the SLURM environment
BASE_DATA_DIR = "/groups/orentsur_group/work/omertole/mimic_data"
IMAGE_DIR = os.path.join(BASE_DATA_DIR, "official_data_iccv_final")

TRAIN_CSV_PATH = os.path.join(BASE_DATA_DIR, "mimic_cxr_aug_train.csv")
VAL_CSV_PATH = os.path.join(BASE_DATA_DIR, "mimic_cxr_aug_validate.csv")

def safe_literal_eval(val):
    """Safely convert string representations of lists to actual Python lists."""
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val

def extract_sections(text):
    """Extract Findings and Impression sections from raw text using regex."""
    if not isinstance(text, str):
        return "", ""

    findings_match = re.search(r'Findings:\s*(.*?)(?=Impression:|$)', text, re.DOTALL | re.IGNORECASE)
    impression_match = re.search(r'Impression:\s*(.*)', text, re.DOTALL | re.IGNORECASE)

    findings = findings_match.group(1).strip() if findings_match else ""
    impression = impression_match.group(1).strip() if impression_match else text

    return findings, impression

def prepare_dataframe(csv_path):
    """Load, convert, explode, and clean the dataset split."""
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Convert string lists to Python lists
    df['image'] = df['image'].apply(safe_literal_eval)
    df['text'] = df['text'].apply(safe_literal_eval)
    df['text_augment'] = df['text_augment'].apply(safe_literal_eval)

    # Flatten dataframe so each row contains exactly one image path
    df_flat = df.explode('image').reset_index(drop=True)

    # Extract the primary text report
    df_flat['text'] = df_flat['text'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)

    # Extract clean text sub-sections
    df_flat[['findings_clean', 'impression_clean']] = df_flat['text'].apply(
        lambda x: pd.Series(extract_sections(x))
    )

    print(f"Processing complete. Total flattened rows: {len(df_flat):,}")
    return df_flat

# Run processing for both splits
df_train_flat = prepare_dataframe(TRAIN_CSV_PATH)
df_val_flat = prepare_dataframe(VAL_CSV_PATH)

class MimicCLIPDataset(Dataset):
    def __init__(self, dataframe, base_image_dir):
        self.df = dataframe
        self.base_dir = base_image_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        current_idx = idx

        while True:
            img_path_rel = self.df.iloc[current_idx]['image']
            img_path_full = os.path.join(self.base_dir, img_path_rel)
            text_signal = self.df.iloc[current_idx]['impression_clean']

            if os.path.exists(img_path_full):
                try:
                    image = Image.open(img_path_full).convert('RGB')
                    image = self.transform(image)

                    # Extract study_id directly from the string path (e.g., 's50414267')
                    match = re.search(r'/s(\d+)/', img_path_rel)
                    study_id = match.group(1) if match else "unknown"

                    # Return all 3 elements
                    return image, text_signal, study_id

                except Exception:
                    pass # Skip to fallback if file is corrupted

            # Fallback strategy: try previous index
            current_idx = (current_idx - 1) % len(self.df)

BATCH_SIZE = 256
NUM_WORKERS = 4

train_dataset = MimicCLIPDataset(df_train_flat, IMAGE_DIR)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

val_dataset = MimicCLIPDataset(df_val_flat, IMAGE_DIR)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Load official pretrained components
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"DataLoaders & Model initialized. Device: {device}")

model_name = "openai/clip-vit-base-patch32"

# Load official pretrained components
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"CLIP Model loaded and mapped to device: {device}")

def study_level_contrastive_loss(image_features, text_features, logit_scale, study_ids):
    """
    Computes Supervised Contrastive Loss at the study level.
    Treats any text/image from the same study_id in the batch as a valid positive pair.
    """
    device = image_features.device

    # L2 Normalize embeddings to unit vectors
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate scaled cosine similarities
    scale = logit_scale.exp()
    logits_per_image = scale * torch.matmul(image_features, text_features.t())
    logits_per_text = logits_per_image.t()

    # Create boolean mask for positive pairs based on identical study_ids
    study_ids_np = np.array(study_ids)
    mask_np = (study_ids_np[:, None] == study_ids_np[None, :])
    mask = torch.tensor(mask_np, dtype=torch.float32, device=device)

    # Calculate log-softmax for both directions
    log_probs_img = F.log_softmax(logits_per_image, dim=1)
    log_probs_txt = F.log_softmax(logits_per_text, dim=1)

    # Average the log-probabilities of all positive pairs per row
    positives_per_row = mask.sum(dim=1)
    loss_img = -(log_probs_img * mask).sum(dim=1) / positives_per_row
    loss_txt = -(log_probs_txt * mask).sum(dim=1) / positives_per_row

    # Return symmetric average
    return (loss_img.mean() + loss_txt.mean()) / 2

LEARNING_RATE = 5e-6
WEIGHT_DECAY = 0.2
EPOCHS = 3

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print("Beginning execution of fine-tuning loop...")

for epoch in range(EPOCHS):
    # --- TRAINING PHASE ---
    model.train()
    train_loss = 0.0
    start_time = time.time()

    # Unpack 3 items: images, texts, and study_ids
    for batch_idx, (images, texts, study_ids) in enumerate(train_loader):
        optimizer.zero_grad()

        text_inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True, max_length=77)
        input_ids = text_inputs['input_ids'].to(device)
        attention_mask = text_inputs['attention_mask'].to(device)
        pixel_values = images.to(device)

        image_outputs = model.get_image_features(pixel_values=pixel_values)
        text_outputs = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        image_features = image_outputs.pooler_output if hasattr(image_outputs, 'pooler_output') else image_outputs
        text_features = text_outputs.pooler_output if hasattr(text_outputs, 'pooler_output') else text_outputs

        # Pass study_ids to the custom loss function
        loss = study_level_contrastive_loss(image_features, text_features, model.logit_scale, study_ids)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if batch_idx % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Step [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)

    # --- VALIDATION PHASE ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, texts, study_ids) in enumerate(val_loader):
            text_inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True, max_length=77)
            input_ids = text_inputs['input_ids'].to(device)
            attention_mask = text_inputs['attention_mask'].to(device)
            pixel_values = images.to(device)

            image_outputs = model.get_image_features(pixel_values=pixel_values)
            text_outputs = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

            image_features = image_outputs.pooler_output if hasattr(image_outputs, 'pooler_output') else image_outputs
            text_features = text_outputs.pooler_output if hasattr(text_outputs, 'pooler_output') else text_outputs

            loss = study_level_contrastive_loss(image_features, text_features, model.logit_scale, study_ids)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    elapsed_time = time.time() - start_time

    print(f"\n=======================================================")
    print(f"Epoch {epoch+1} Metrics:")
    print(f"-> Average Train Loss: {avg_train_loss:.4f}")
    print(f"-> Average Val Loss  : {avg_val_loss:.4f}")
    print(f"-> Time Taken        : {elapsed_time:.2f}s")
    print(f"=======================================================\n")

# Save Baseline Model
output_model_dir = os.path.join(BASE_DATA_DIR, "hard_clip_study_baseline")
os.makedirs(output_model_dir, exist_ok=True)
model.save_pretrained(output_model_dir)
processor.save_pretrained(output_model_dir)
print(f"Model saved to {output_model_dir}")

# Ensure model is in evaluation mode
model.eval()

all_image_features = []
all_text_features = []

print("Extracting features for the Validation Set for Evaluation...")
with torch.no_grad():
    for images, texts, _ in tqdm(val_loader, desc="Validation Batches"):
        text_inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        pixel_values = images.to(device)

        image_outputs = model.get_image_features(pixel_values=pixel_values)
        text_outputs = model.get_text_features(**text_inputs)

        img_feats = image_outputs.pooler_output if hasattr(image_outputs, 'pooler_output') else image_outputs
        txt_feats = text_outputs.pooler_output if hasattr(text_outputs, 'pooler_output') else text_outputs

        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

        all_image_features.append(img_feats.cpu())
        all_text_features.append(txt_feats.cpu())

all_image_features = torch.cat(all_image_features, dim=0).to(device)
all_text_features = torch.cat(all_text_features, dim=0).to(device)

print("\n--- Calculating Retrieval Metrics (Study-Level) ---")
sim_matrix = torch.matmul(all_image_features, all_text_features.t())

# Extract study_ids directly to an array
def get_study_id(image_path):
    match = re.search(r'/s(\d+)/', image_path)
    return match.group(1) if match else "unknown"

df_val_flat['study_id'] = df_val_flat['image'].apply(get_study_id)
study_ids = df_val_flat['study_id'].values

def calculate_study_level_metrics(sim_matrix, study_ids):
    sim_matrix_np = sim_matrix.cpu().numpy()
    r1, r5, r10, mrr = 0.0, 0.0, 0.0, 0.0
    ranks = []
    num_queries = sim_matrix_np.shape[0]

    for i in range(num_queries):
        query_study = study_ids[i]
        sorted_indices = np.argsort(-sim_matrix_np[i])
        retrieved_studies = study_ids[sorted_indices]
        matches = (retrieved_studies == query_study)

        first_match_rank = np.where(matches)[0][0] + 1
        ranks.append(first_match_rank)

        if first_match_rank <= 1: r1 += 1
        if first_match_rank <= 5: r5 += 1
        if first_match_rank <= 10: r10 += 1
        mrr += 1.0 / first_match_rank

    return (r1/num_queries)*100, (r5/num_queries)*100, (r10/num_queries)*100, np.median(ranks), mrr/num_queries

i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_mrr = calculate_study_level_metrics(sim_matrix, study_ids)
t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_mrr = calculate_study_level_metrics(sim_matrix.t(), study_ids)

print(f"\n[Image-to-Text Retrieval (Study-Level)]")
print(f"Recall@1 : {i2t_r1:.2f}% | Recall@5 : {i2t_r5:.2f}% | Recall@10: {i2t_r10:.2f}% | Median R : {i2t_medr:.1f} | MRR: {i2t_mrr:.4f}")

print(f"\n[Text-to-Image Retrieval (Study-Level)]")
print(f"Recall@1 : {t2i_r1:.2f}% | Recall@5 : {t2i_r5:.2f}% | Recall@10: {t2i_r10:.2f}% | Median R : {t2i_medr:.1f} | MRR: {t2i_mrr:.4f}")
print("==========================================\n")