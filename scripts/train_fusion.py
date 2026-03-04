"""
scripts/train_fusion.py
=======================
Step 7 of the FEDrA pipeline.

Trains a late-fusion ensemble that combines the three baseline model
probability outputs (URL, HTML, Image) into a single phishing detector.

Strategy: Logistic Regression meta-learner on the stacked probabilities.

NOTE: Models saved with joblib.dump() — always load with joblib.load()

Usage:
    python scripts/train_fusion.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib  # ← always use joblib.load() for .pkl files in this project

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR   = os.path.join(BASE_DIR, "Dataset", "features")
MANIFEST   = os.path.join(BASE_DIR, "Dataset", "manifest.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

BASELINE_MODELS = {
    "url":   os.path.join(MODELS_DIR, "url_baseline.pkl"),
    "html":  os.path.join(MODELS_DIR, "html_baseline.pkl"),
    "image": os.path.join(MODELS_DIR, "image_baseline.pkl"),
}

FUSION_MODEL_PATH   = os.path.join(MODELS_DIR, "fusion_model.pkl")
FUSION_METRICS_PATH = os.path.join(MODELS_DIR, "fusion_metrics.json")

os.makedirs(MODELS_DIR, exist_ok=True)


# ── Load baseline models ───────────────────────────────────────────────────────
def load_baselines() -> dict:
    """
    Load all baseline model bundles.
    NOTE: Models saved with joblib.dump() — always load with joblib.load()
    """
    bundles = {}
    for name, path in BASELINE_MODELS.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Baseline model not found: {path}\n"
                "Run scripts/train_baselines.py first."
            )
        # NOTE: Models saved with joblib.dump() — always load with joblib.load()
        bundles[name] = joblib.load(path)
        print(f"  [✓] Loaded {name}_baseline ({type(bundles[name]['model']).__name__})")
    return bundles


# ── Build meta-features ────────────────────────────────────────────────────────
def build_meta_features(bundles: dict, manifest: pd.DataFrame) -> np.ndarray:
    """
    For every sample, collect the phishing probability from each baseline.
    Returns shape (N, 3) — one column per modality.
    """
    manifest = manifest.sort_values("sample_id").reset_index(drop=True)

    # Load raw features
    url_df = pd.read_csv(os.path.join(FEAT_DIR, "url_features.csv"))
    url_df = url_df.sort_values("sample_id").reset_index(drop=True)
    url_df = pd.get_dummies(url_df, drop_first=True)
    X_url  = url_df.drop(columns=["sample_id"]).values

    html_df = pd.read_csv(os.path.join(FEAT_DIR, "html_features.csv"))
    html_df = html_df.sort_values("sample_id").reset_index(drop=True)
    X_html  = html_df.drop(columns=["sample_id"]).values

    X_img = np.load(os.path.join(FEAT_DIR, "visual_embeddings.npy"))

    feature_sets = {"url": X_url, "html": X_html, "image": X_img}

    meta_cols = []
    for modality, X in feature_sets.items():
        bundle  = bundles[modality]
        model   = bundle["model"]
        scaler  = bundle["scaler"]

        # Align feature count (handles one-hot expansion differences)
        expected = scaler.n_features_in_
        current  = X.shape[1]
        if current < expected:
            X = np.hstack([X, np.zeros((X.shape[0], expected - current))])
        elif current > expected:
            X = X[:, :expected]

        X_scaled = scaler.transform(X)
        proba    = model.predict_proba(X_scaled)[:, 1]   # phishing probability
        meta_cols.append(proba)
        print(f"  [✓] {modality.upper():5s} meta-features built  shape=({len(proba)},)")

    return np.column_stack(meta_cols)   # (N, 3)


# ── Train fusion ───────────────────────────────────────────────────────────────
def run():
    print("\n" + "=" * 60)
    print("  FEDrA Step 7 — Fusion Model Training")
    print("=" * 60)

    # Load manifest and labels
    manifest = pd.read_csv(MANIFEST)
    manifest = manifest.sort_values("sample_id").reset_index(drop=True)
    labels   = manifest["label"].values
    print(f"\nSamples: {len(labels)}  |  Phishing: {sum(labels)}  |  Legit: {len(labels)-sum(labels)}")

    # ── Step 1: Load baseline models ──
    print("\n[Step 1] Loading baseline models (joblib.load)...")
    bundles = load_baselines()

    # ── Step 2: Build meta-features ──
    print("\n[Step 2] Building meta-features from baseline probabilities...")
    X_meta = build_meta_features(bundles, manifest)
    print(f"  Meta-feature matrix shape: {X_meta.shape}")

    # ── Step 3: Stratified split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_meta, labels, test_size=0.20, stratify=labels, random_state=42
    )
    print(f"\n[Step 3] Split → Train: {len(y_train)}  Test: {len(y_test)}")

    # ── Step 4: Train meta-learner ──
    print("\n[Step 4] Training Logistic Regression meta-learner...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_train_s, y_train)

    # ── Step 5: Evaluate ──
    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    metrics = {
        "Accuracy":  round(accuracy_score(y_test, y_pred),           4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0),    4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0),        4),
        "AUC":       round(roc_auc_score(y_test, y_prob),             4),
    }

    print("\n[Step 5] Fusion Model Metrics:")
    print(f"  {'Accuracy':<12}: {metrics['Accuracy']}")
    print(f"  {'Precision':<12}: {metrics['Precision']}")
    print(f"  {'Recall':<12}: {metrics['Recall']}")
    print(f"  {'F1':<12}: {metrics['F1']}")
    print(f"  {'AUC':<12}: {metrics['AUC']}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Legit", "Phishing"]))

    # ── Step 6: Save fusion model ──
    # NOTE: Models saved with joblib.dump() — always load with joblib.load()
    joblib.dump({"model": clf, "scaler": scaler}, FUSION_MODEL_PATH)
    print(f"[Step 6] Fusion model saved → {FUSION_MODEL_PATH}")

    with open(FUSION_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"         Metrics saved   → {FUSION_METRICS_PATH}")

    print("\n" + "=" * 60)
    print("  Fusion training complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run()
