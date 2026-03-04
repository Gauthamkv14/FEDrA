"""
scripts/train_baselines.py
==========================
Step 6 of the FEDrA pipeline.

Trains three separate baseline models (URL-only, HTML-only, Image-only).
- 80/20 train/test split (stratified by label)
- Address class imbalance using class_weight='balanced'
- Reports Precision, Recall, F1, AUC
- Saves trained models to models/ folder using sklearn/joblib
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# ── Config ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(BASE_DIR, "Dataset", "features")
MANIFEST = os.path.join(BASE_DIR, "Dataset", "manifest.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)
import json
RESULTS = {}

# ── Evaluator ───────────────────────────────────────────────────────────
def evaluate_and_save(clf, X_train, X_test, y_train, y_test, scaler, name):
    print(f"\n[{name.upper()}] Training...")
    
    # Scale data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    clf.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else y_pred
    
    # Metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    acc = clf.score(X_test_scaled, y_test)
    
    RESULTS[name] = {
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "AUC": round(auc, 4)
    }
    
    # Save Model + Scaler together
    model_path = os.path.join(MODELS_DIR, f"{name}_baseline.pkl")
    joblib.dump({"model": clf, "scaler": scaler}, model_path)
    print(f"[{name.upper()}] Acc: {acc:.4f} F1: {f1:.4f} AUC: {auc:.4f} -> {model_path}")

# ── Main ─────────────────────────────────────────────────────────────────
def run():
    print("Loading Manifest...")
    manifest = pd.read_csv(MANIFEST)
    # Ensure order matches features
    manifest = manifest.sort_values(by="sample_id").reset_index(drop=True)
    labels = manifest["label"].values

    print(f"Total samples: {len(labels)}, Phishing: {sum(labels)}, Legit: {len(labels)-sum(labels)}")
    
    # We will use the same stratified split indices for all 3 modalities to keep test sets consistent
    indices = np.arange(len(labels))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, labels, test_size=0.20, stratify=labels, random_state=42
    )
    
    print(f"Train set: {len(y_train)} samples\nTest set: {len(y_test)} samples")
    
    # ── 1. URL Model ──
    url_df = pd.read_csv(os.path.join(FEAT_DIR, "url_features.csv"))
    url_df = url_df.sort_values(by="sample_id").reset_index(drop=True)
    
    # Handle categorical columns (like tld_type) via one-hot encoding
    url_df = pd.get_dummies(url_df, drop_first=True)
    
    X_url = url_df.drop(columns=["sample_id"]).values
    
    url_clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    evaluate_and_save(url_clf, X_url[idx_train], X_url[idx_test], y_train, y_test, StandardScaler(), "url")

    # ── 2. HTML Model ──
    html_df = pd.read_csv(os.path.join(FEAT_DIR, "html_features.csv"))
    html_df = html_df.sort_values(by="sample_id").reset_index(drop=True)
    X_html = html_df.drop(columns=["sample_id"]).values
    
    html_clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    evaluate_and_save(html_clf, X_html[idx_train], X_html[idx_test], y_train, y_test, StandardScaler(), "html")

    # ── 3. Image Model ──
    X_img = np.load(os.path.join(FEAT_DIR, "visual_embeddings.npy"))
    # The embeddings are ordered by manifest traversal.
    # To be safe against index mismatches from the numpy array, we trust they align 1:1 with manifest index
    
    img_clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    evaluate_and_save(img_clf, X_img[idx_train], X_img[idx_test], y_train, y_test, StandardScaler(), "image")

    with open(os.path.join(MODELS_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("Metrics saved to models/baseline_metrics.json")

if __name__ == "__main__":
    run()
