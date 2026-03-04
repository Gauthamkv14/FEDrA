"""
scripts/extract_visual_embeddings.py
======================================
Extracts visual embeddings from screenshots using a pretrained MobileNetV2 
frozen feature extractor as per Step 5 schema.
"""

import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST    = os.path.join(BASE_DIR, "Dataset", "manifest.csv")
OUT_DIR     = os.path.join(BASE_DIR, "Dataset", "features")
OUT_NPY     = os.path.join(OUT_DIR, "visual_embeddings.npy")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Model & Transforms ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pretrained MobileNetV2 and use only the feature extractor part
# The original features output represents [Batch, 1280, 7, 7]
weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
model = models.mobilenet_v2(weights=weights)
model = model.features
model.eval()
model.to(device)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Helpers ─────────────────────────────────────────────────────────────
def extract_embedding(img_path: str) -> np.ndarray:
    """Extracts a 1280-dim feature vector for an image."""
    if not img_path or not isinstance(img_path, str) or not os.path.isfile(img_path):
        return np.zeros(1280, dtype=np.float32)
        
    try:
        with Image.open(img_path) as img:
            # Add RGB conversion for safety (handles RGBA/L/etc)
            img = img.convert('RGB')
            tensor = preprocess(img).unsqueeze(0).to(device)
            
        with torch.no_grad():
            features = model(tensor)  # (1, 1280, 7, 7)
            # Global Average Pooling (simulating the classifier head's pooling)
            embedding = features.mean([2, 3]).squeeze(0)  # (1280,)
            
        return embedding.cpu().numpy()
        
    except Exception as e:
        print(f"  [WARN] Failed to process {img_path}: {e}")
        return np.zeros(1280, dtype=np.float32)

# ── Main ─────────────────────────────────────────────────────────────────
def run():
    print(f"Reading {MANIFEST}...")
    df = pd.read_csv(MANIFEST)
    
    embeddings = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Visual embeddings"):
        emb = extract_embedding(row['screenshot_path'])
        embeddings.append(emb)
        
    # Stack into shape (N, 1280)
    embeddings_array = np.stack(embeddings)
    
    # Save directly as .npy
    np.save(OUT_NPY, embeddings_array)
    print(f"Saved {embeddings_array.shape} array to {OUT_NPY}")

if __name__ == "__main__":
    run()
