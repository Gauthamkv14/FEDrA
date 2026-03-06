"""
scripts/train_fusion.py
=======================
Step 7 of the FEDrA pipeline — FUSION MODEL (Feature-Level Concatenation)

Strategy:
  1. Load raw features for all 3 modalities:
       url_features.csv, html_features.csv, visual_embeddings.npy
  2. Normalize each modality independently (separate StandardScaler per modality)
  3. Apply modality weights BEFORE concatenation:
       URL = 0.4  (strongest signal)
       HTML = 0.3
       Visual = 0.3
  4. Train a single MLP classifier (sigmoid output) on the weighted concat vector
  5. 80/20 stratified split + class_weight='balanced'
  6. Report Precision, Recall, F1, AUC — compared against Step 6 baselines
  7. Save fusion model to models/fusion_model.pkl (joblib)

NOTE: This file is completely standalone.  Do NOT mix with test_single_url.py.
NOTE: All models are saved/loaded with joblib (not pickle).

Usage:
    /opt/anaconda3/envs/fedra/bin/python scripts/train_fusion.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR   = os.path.join(BASE_DIR, "Dataset", "features")
MANIFEST   = os.path.join(BASE_DIR, "Dataset", "manifest.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FUSION_MODEL_PATH   = os.path.join(MODELS_DIR, "fusion_model.pkl")
FUSION_METRICS_PATH = os.path.join(MODELS_DIR, "fusion_metrics.json")
BASELINE_METRICS    = os.path.join(MODELS_DIR, "baseline_metrics.json")

os.makedirs(MODELS_DIR, exist_ok=True)

# Modality weights (must sum to 1.0)
WEIGHTS = {"url": 0.4, "html": 0.3, "visual": 0.3}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_features(manifest: pd.DataFrame):
    """
    Load all three modality feature matrices, aligned to manifest order.
    Returns X_url, X_html, X_visual and the label array y.
    """
    # Labels
    manifest = manifest.sort_values("sample_id").reset_index(drop=True)
    y = manifest["label"].values

    # --- URL features ---
    url_df = pd.read_csv(os.path.join(FEAT_DIR, "url_features.csv"))
    url_df = url_df.sort_values("sample_id").reset_index(drop=True)
    url_df = pd.get_dummies(url_df, drop_first=True)   # handle tld_type etc.
    X_url = url_df.drop(columns=["sample_id"]).values.astype(np.float64)

    # --- HTML features ---
    html_df = pd.read_csv(os.path.join(FEAT_DIR, "html_features.csv"))
    html_df = html_df.sort_values("sample_id").reset_index(drop=True)
    X_html = html_df.drop(columns=["sample_id"]).values.astype(np.float64)

    # --- Visual embeddings ---
    X_visual = np.load(os.path.join(FEAT_DIR, "visual_embeddings.npy")).astype(np.float64)

    assert len(y) == X_url.shape[0] == X_html.shape[0] == X_visual.shape[0], (
        f"Sample count mismatch! labels={len(y)}, url={X_url.shape[0]}, "
        f"html={X_html.shape[0]}, visual={X_visual.shape[0]}"
    )

    print(f"  URL features   : {X_url.shape}")
    print(f"  HTML features  : {X_html.shape}")
    print(f"  Visual embed.  : {X_visual.shape}")
    return X_url, X_html, X_visual, y


# ── Build weighted-concatenated feature vector ────────────────────────────────
def build_fusion_features(X_url, X_html, X_visual, idx_train, idx_test):
    """
    For each modality:
      1. Fit a StandardScaler on the TRAINING slice only
      2. Transform both train + test
      3. Multiply by the modality weight
    Then concatenate all weighted modalities horizontally.

    Returns:
        X_train_fused, X_test_fused  (shape: [N, url_dim+html_dim+visual_dim])
        scalers dict  (kept for inference at test time)
    """
    scalers = {}
    train_parts = []
    test_parts  = []

    for name, X, w in [
        ("url",    X_url,    WEIGHTS["url"]),
        ("html",   X_html,   WEIGHTS["html"]),
        ("visual", X_visual, WEIGHTS["visual"]),
    ]:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[idx_train]) * w
        X_te = sc.transform(X[idx_test])      * w
        scalers[name] = sc
        train_parts.append(X_tr)
        test_parts.append(X_te)
        print(f"  [✓] {name.upper():<6} normalized + weighted (w={w})  "
              f"dim={X.shape[1]}")

    X_train_fused = np.hstack(train_parts)
    X_test_fused  = np.hstack(test_parts)
    print(f"\n  Fused train shape : {X_train_fused.shape}")
    print(f"  Fused test  shape : {X_test_fused.shape}")
    return X_train_fused, X_test_fused, scalers


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print("\n" + "=" * 65)
    print("  FEDrA Step 7 — Fusion Model (Weighted Feature Concatenation MLP)")
    print("=" * 65)

    # ── Step 1: Load manifest + features ──
    print("\n[Step 1] Loading features...")
    manifest = pd.read_csv(MANIFEST)
    X_url, X_html, X_visual, y = load_features(manifest)
    print(f"\n  Total samples : {len(y)}")
    print(f"  Phishing      : {int(y.sum())}  |  Legit : {int((y == 0).sum())}")

    # ── Step 2: Stratified 80/20 split (index-based — same split for all modalities) ──
    print("\n[Step 2] Stratified 80/20 split...")
    indices = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"  Train: {len(y_train)}  |  Test: {len(y_test)}")

    # ── Step 3: Normalize each modality separately, apply weights, concatenate ──
    print("\n[Step 3] Per-modality normalization + weighting + concatenation...")
    print(f"  Weights → URL={WEIGHTS['url']}  HTML={WEIGHTS['html']}  "
          f"Visual={WEIGHTS['visual']}")
    X_train_fused, X_test_fused, scalers = build_fusion_features(
        X_url, X_html, X_visual, idx_train, idx_test
    )

    # ── Step 4: Train MLP classifier ──
    # Note: sklearn MLPClassifier does not support class_weight directly.
    # We use sample_weight to achieve balanced class training.
    print("\n[Step 4] Training MLP classifier (sigmoid output, balanced weights)...")
    from sklearn.utils.class_weight import compute_sample_weight
    sw = compute_sample_weight("balanced", y_train)

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_train_fused, y_train, sample_weight=sw)
    print(f"  Training stopped after {mlp.n_iter_} iterations")

    # ── Step 5: Evaluate ──
    print("\n[Step 5] Evaluating fusion model on held-out test set...")
    y_pred = mlp.predict(X_test_fused)
    y_prob = mlp.predict_proba(X_test_fused)[:, 1]   # sigmoid-like probability

    fusion_metrics = {
        "Accuracy":  round(float(accuracy_score(y_test, y_pred)),              4),
        "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "Recall":    round(float(recall_score(y_test, y_pred, zero_division=0)),    4),
        "F1":        round(float(f1_score(y_test, y_pred, zero_division=0)),        4),
        "AUC":       round(float(roc_auc_score(y_test, y_prob)),               4),
    }

    # ── Step 6: Print comparison table ──
    print("\n" + "=" * 65)
    print("  PERFORMANCE COMPARISON — Fusion vs. Step 6 Baselines")
    print("=" * 65)
    header = f"  {'Model':<22} {'Precision':>10} {'Recall':>10} {'F1':>8} {'AUC':>8}"
    sep    = "  " + "-" * 61
    print(header)
    print(sep)

    # Load baseline metrics if available
    if os.path.isfile(BASELINE_METRICS):
        with open(BASELINE_METRICS) as f:
            baseline = json.load(f)
        for model_name, m in baseline.items():
            label = f"Step-6 {model_name.upper()} (baseline)"
            print(f"  {label:<22} {m['Precision']:>10.4f} {m['Recall']:>10.4f} "
                  f"{m['F1']:>8.4f} {m['AUC']:>8.4f}")
    else:
        print("  (baseline_metrics.json not found — run train_baselines.py first)")

    print(sep)
    fm = fusion_metrics
    print(f"  {'Step-7 FUSION (MLP)':<22} {fm['Precision']:>10.4f} {fm['Recall']:>10.4f} "
          f"{fm['F1']:>8.4f} {fm['AUC']:>8.4f}")
    print("=" * 65)

    print("\n  Full classification report (Fusion):")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Phishing"]))

    # ── Step 7: Save fusion model ──
    bundle = {
        "model":   mlp,
        "scalers": scalers,          # dict with keys: url, html, visual
        "weights": WEIGHTS,
        "url_columns": None,         # placeholder — populated below if needed
    }
    joblib.dump(bundle, FUSION_MODEL_PATH)
    print(f"[Step 7] Fusion model saved  → {FUSION_MODEL_PATH}")

    with open(FUSION_METRICS_PATH, "w") as f:
        json.dump(fusion_metrics, f, indent=2)
    print(f"         Fusion metrics saved → {FUSION_METRICS_PATH}")

    print("\n" + "=" * 65)
    print("  Step 7 complete!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run()
