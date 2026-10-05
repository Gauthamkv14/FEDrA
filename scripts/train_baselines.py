"""
scripts/train_baselines.py
==========================
Step 6 of the FEDrA pipeline.

Trains three separate baseline models (URL-only, HTML-only, Image-only).
- 80/20 train/test split (stratified by label, random_state=42)
- Address class imbalance using class_weight='balanced'
- Reports Accuracy, Precision, Recall, F1, AUC
- Saves trained models to models/ folder using sklearn/joblib

NOTE: Models saved with joblib.dump() — always load with joblib.load()
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

# Ensure scripts directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from url_features import (
    CANONICAL_URL_FEATURE_NAMES,
    URL_SCHEMA_VERSION,
    URL_FEATURE_DIM,
)

FEAT_DIR = os.path.join(BASE_DIR, "Dataset", "features")
MANIFEST = os.path.join(BASE_DIR, "Dataset", "manifest.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)
RESULTS = {}


# ── Evaluator ───────────────────────────────────────────────────────────
def evaluate_and_save(clf, X_train, X_test, y_train, y_test, scaler, name, extra_meta=None):
    print(f"\n[{name.upper()}] Training (input dim: {X_train.shape[1]})...")
    
    # Scale data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    clf.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else y_pred
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    
    RESULTS[name] = {
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1": round(float(f1), 4),
        "AUC": round(float(auc), 4),
        "input_dim": int(X_train.shape[1]),
    }
    
    # Save Model + Scaler + Metadata together
    bundle = {
        "model": clf,
        "scaler": scaler,
        "input_dim": int(X_train.shape[1]),
    }
    if extra_meta:
        bundle.update(extra_meta)
        
    model_path = os.path.join(MODELS_DIR, f"{name}_baseline.pkl")
    joblib.dump(bundle, model_path)
    print(f"[{name.upper()}] Acc: {acc:.4f} Prec: {precision:.4f} Rec: {recall:.4f} F1: {f1:.4f} AUC: {auc:.4f} -> {model_path}")


# ── Main ─────────────────────────────────────────────────────────────────
def run():
    print("Loading Manifest...")
    manifest = pd.read_csv(MANIFEST)
    manifest = manifest.sort_values(by="sample_id").reset_index(drop=True)
    labels = manifest["label"].values

    print(f"Total samples: {len(labels)}, Phishing: {sum(labels)}, Legit: {len(labels)-sum(labels)}")
    
    indices = np.arange(len(labels))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, labels, test_size=0.20, stratify=labels, random_state=42
    )
    
    print(f"Train set: {len(y_train)} samples\nTest set:  {len(y_test)} samples")
    
    # ── 1. URL Model (Canonical 22 features) ──
    url_df = pd.read_csv(os.path.join(FEAT_DIR, "url_features.csv"))
    url_df = url_df.sort_values(by="sample_id").reset_index(drop=True)
    
    # Select exact canonical columns
    feature_cols = [c for c in CANONICAL_URL_FEATURE_NAMES if c in url_df.columns]
    assert len(feature_cols) == URL_FEATURE_DIM, f"Expected {URL_FEATURE_DIM} URL features, found {len(feature_cols)}"
    
    X_url = url_df[feature_cols].values.astype(np.float64)
    print(f"\nURL feature matrix shape: {X_url.shape}")
    
    url_clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    url_meta = {
        "schema_version": URL_SCHEMA_VERSION,
        "feature_names": CANONICAL_URL_FEATURE_NAMES,
        "url_feature_dim": URL_FEATURE_DIM,
    }
    evaluate_and_save(url_clf, X_url[idx_train], X_url[idx_test], y_train, y_test, StandardScaler(), "url", url_meta)

    # ── 2. HTML Model (12 features) ──
    html_df = pd.read_csv(os.path.join(FEAT_DIR, "html_features.csv"))
    html_df = html_df.sort_values(by="sample_id").reset_index(drop=True)
    X_html = html_df.drop(columns=["sample_id"]).values.astype(np.float64)
    print(f"HTML feature matrix shape: {X_html.shape}")
    
    html_clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    evaluate_and_save(html_clf, X_html[idx_train], X_html[idx_test], y_train, y_test, StandardScaler(), "html")

    # ── 3. Image Model (1280 features) ──
    X_img = np.load(os.path.join(FEAT_DIR, "visual_embeddings.npy")).astype(np.float64)
    print(f"Image feature matrix shape: {X_img.shape}")
    
    img_clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    evaluate_and_save(img_clf, X_img[idx_train], X_img[idx_test], y_train, y_test, StandardScaler(), "image")

    metrics_path = os.path.join(MODELS_DIR, "baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\n[OK] Fresh baseline metrics saved to {metrics_path}")


if __name__ == "__main__":
    run()
