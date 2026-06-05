#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import ast
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# Define paths
BASE_DATA_DIR = "/groups/orentsur_group/work/omertole/mimic_data"
IMAGE_DIR = os.path.join(BASE_DATA_DIR, "official_data_iccv_final")
TRAIN_CSV_PATH = os.path.join(BASE_DATA_DIR, "mimic_cxr_processed_train.csv")
VAL_CSV_PATH = os.path.join(BASE_DATA_DIR, "mimic_cxr_processed_validate.csv")
# Path to the .pt file you generated in the previous step
BERT_EMBEDDINGS_PATH_TRAIN = os.path.join(BASE_DATA_DIR, "train_bioclinicalbert_embeddings.pt")
BERT_EMBEDDINGS_PATH_VAL = os.path.join(BASE_DATA_DIR, "val_bioclinicalbert_embeddings.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


class MimicSoftCLIPDataset(Dataset):
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
                    match = re.search(r'/s(\d+)/', img_path_rel)
                    study_id = match.group(1) if match else "unknown"

                    # Return image, text, study_id, and the original dataset index
                    return image, text_signal, study_id, current_idx
                except Exception:
                    pass
            current_idx = (current_idx - 1) % len(self.df)


# In[3]:


def soft_clip_hybrid_loss(image_features, text_features, logit_scale, batch_semantic_embeddings, alpha=0.5, soft_temp=0.1):
    # L2 Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate CLIP scaled logits
    scale = logit_scale.exp()
    logits_per_image = scale * torch.matmul(image_features, text_features.t())
    logits_per_text = logits_per_image.t()

    # 1. Hard Loss
    batch_size = image_features.size(0)
    hard_targets = torch.arange(batch_size, device=image_features.device)
    hard_loss = (F.cross_entropy(logits_per_image, hard_targets) + F.cross_entropy(logits_per_text, hard_targets)) / 2

    # 2. Soft Loss (KL Divergence)
    # Cosine similarity of semantic embeddings
    semantic_sim_matrix = torch.matmul(batch_semantic_embeddings, batch_semantic_embeddings.t())
    soft_targets_dist = F.softmax(semantic_sim_matrix / soft_temp, dim=1)

    log_preds_img = F.log_softmax(logits_per_image, dim=1)
    log_preds_txt = F.log_softmax(logits_per_text, dim=1)

    soft_loss = (F.kl_div(log_preds_img, soft_targets_dist, reduction='batchmean') + 
                 F.kl_div(log_preds_txt, soft_targets_dist.t(), reduction='batchmean')) / 2

    return (1 - alpha) * hard_loss + alpha * soft_loss


# In[ ]:


# Setup
# Load separate semantic embeddings for train and validation sets
train_semantic_embeddings = torch.load(BERT_EMBEDDINGS_PATH_TRAIN, map_location='cpu')
val_semantic_embeddings = torch.load(BERT_EMBEDDINGS_PATH_VAL, map_location='cpu')

model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.2)

# Loaders
df_train_flat = pd.read_csv(TRAIN_CSV_PATH).fillna("")
df_val_flat = pd.read_csv(VAL_CSV_PATH).fillna("")

train_loader = DataLoader(dataset=MimicSoftCLIPDataset(df_train_flat, IMAGE_DIR), batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=MimicSoftCLIPDataset(df_val_flat, IMAGE_DIR), batch_size=128, shuffle=False, num_workers=4)

# Early stopping configuration
EPOCHS = 10
PATIENCE = 2
OUTPUT_DIR = os.path.join("home/omertole/checkpoints/soft_clip_baseline")

best_val_loss = float('inf')
epochs_without_improvement = 0

print("Starting Soft-CLIP Training...")

for epoch in range(EPOCHS):
    # --- TRAINING PHASE ---
    model.train()
    for batch_idx, (images, texts, study_ids, indices) in enumerate(train_loader):
        optimizer.zero_grad()

        # CRITICAL FIX: Explicitly cast all elements to string to prevent float/NaN tokenizer crashes
        clean_texts = [str(t) for t in texts]

        # Get embeddings
        text_inputs = processor(text=clean_texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        pixel_values = images.to(device)

        # Fetch from train semantic embeddings using batch indices
        batch_semantic_embeddings = train_semantic_embeddings[indices].to(device)

        # Forward pass through CLIP
        image_outputs = model.get_image_features(pixel_values=pixel_values)
        text_outputs = model.get_text_features(**text_inputs)

        # Extract raw tensors from Hugging Face output objects
        image_features = image_outputs.pooler_output if hasattr(image_outputs, 'pooler_output') else image_outputs
        text_features = text_outputs.pooler_output if hasattr(text_outputs, 'pooler_output') else text_outputs

        # Hybrid Loss
        loss = soft_clip_hybrid_loss(image_features, text_features, model.logit_scale, batch_semantic_embeddings, alpha=0.5)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Step {batch_idx} | Loss: {loss.item():.4f}")

    # --- VALIDATION PHASE ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, texts, study_ids, indices) in enumerate(val_loader):
            # CRITICAL FIX: Explicitly cast all elements to string to prevent float/NaN tokenizer crashes
            clean_texts = [str(t) for t in texts]

            text_inputs = processor(text=clean_texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            pixel_values = images.to(device)

            # Fetch from validation semantic embeddings using batch indices
            batch_semantic_embeddings = val_semantic_embeddings[indices].to(device)

            # Forward pass through CLIP
            image_outputs = model.get_image_features(pixel_values=pixel_values)
            text_outputs = model.get_text_features(**text_inputs)

            # Extract raw tensors from Hugging Face output objects
            image_features = image_outputs.pooler_output if hasattr(image_outputs, 'pooler_output') else image_outputs
            text_features = text_outputs.pooler_output if hasattr(text_outputs, 'pooler_output') else text_outputs

            loss = soft_clip_hybrid_loss(image_features, text_features, model.logit_scale, batch_semantic_embeddings, alpha=0.5)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Average Val Loss: {avg_val_loss:.4f}")

    # --- EARLY STOPPING CHECK ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0

        # Save the current best model weights and processor configuration
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        print(f"--> Saved new best checkpoint to {OUTPUT_DIR} with Validation Loss: {best_val_loss:.4f}\n")
    else:
        epochs_without_improvement += 1
        print(f"--> Validation loss did not improve. Counter: {epochs_without_improvement}/{PATIENCE}\n")

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered. Terminating training loop at Epoch {epoch+1}.")
            break


# In[ ]:


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

