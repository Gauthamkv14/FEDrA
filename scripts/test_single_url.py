"""
scripts/test_single_url.py
==========================
Run a single-URL prediction using the three baseline models.

Usage:
    python scripts/test_single_url.py --url "https://example.com"

NOTE: Models saved with joblib.dump() — always load with joblib.load()
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib  # ← always use joblib.load() for .pkl files in this project

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILES = {
    "url":   os.path.join(MODELS_DIR, "url_baseline.pkl"),
    "html":  os.path.join(MODELS_DIR, "html_baseline.pkl"),
    "image": os.path.join(MODELS_DIR, "image_baseline.pkl"),
}


# ── Feature extractors (lightweight, inline) ─────────────────────────────────
def extract_url_features(url: str) -> np.ndarray:
    """
    Extract the same URL features produced by scripts/extract_url_features.py.
    Returns a 2-D array of shape (1, n_url_features).
    """
    import re
    from urllib.parse import urlparse

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path     = parsed.path or ""

    feats = {
        "url_length":        len(url),
        "num_dots":          url.count("."),
        "num_hyphens":       url.count("-"),
        "num_slashes":       url.count("/"),
        "num_at":            url.count("@"),
        "num_digits":        sum(c.isdigit() for c in url),
        "num_special":       sum(not c.isalnum() for c in url),
        "has_https":         int(parsed.scheme == "https"),
        "hostname_length":   len(hostname),
        "path_length":       len(path),
        "num_subdomains":    hostname.count("."),
        "has_ip":            int(bool(re.fullmatch(r"\d+\.\d+\.\d+\.\d+", hostname))),
    }
    return np.array(list(feats.values()), dtype=float).reshape(1, -1)


def extract_html_features_dummy() -> np.ndarray:
    """
    Placeholder: returns a zero vector matching the HTML model's expected shape.
    For a real pipeline, render the page and extract features.
    """
    try:
        bundle = joblib.load(MODEL_FILES["html"])  # NOTE: joblib.load()
        n = bundle["scaler"].n_features_in_
    except Exception:
        n = 12
    print(f"  [html]  Using zero-vector dummy ({n} features) — "
          "replace with real HTML extraction for production use.")
    return np.zeros((1, n))


def extract_visual_features_dummy() -> np.ndarray:
    """
    Placeholder: returns a zero vector matching the image model's expected shape.
    For a real pipeline, capture a screenshot and run MobileNetV2.
    """
    try:
        bundle = joblib.load(MODEL_FILES["image"])  # NOTE: joblib.load()
        n = bundle["scaler"].n_features_in_
    except Exception:
        n = 1280
    print(f"  [image] Using zero-vector dummy ({n} features) — "
          "replace with real visual extraction for production use.")
    return np.zeros((1, n))


# ── Predict ──────────────────────────────────────────────────────────────────
def predict_url(url: str) -> dict:
    """
    Load all three baseline models (via joblib.load) and return predictions.

    NOTE: Models saved with joblib.dump() — always load with joblib.load()
    """
    results = {}

    feature_map = {
        "url":   extract_url_features(url),
        "html":  extract_html_features_dummy(),
        "image": extract_visual_features_dummy(),
    }

    for modality, X in feature_map.items():
        path = MODEL_FILES[modality]
        if not os.path.isfile(path):
            results[modality] = {"error": f"Model file not found: {path}"}
            continue
        try:
            # NOTE: Models saved with joblib.dump() — always load with joblib.load()
            bundle  = joblib.load(path)
            model   = bundle["model"]
            scaler  = bundle["scaler"]

            # Pad / truncate X to match expected feature count
            expected = scaler.n_features_in_
            current  = X.shape[1]
            if current < expected:
                X = np.hstack([X, np.zeros((1, expected - current))])
            elif current > expected:
                X = X[:, :expected]

            X_scaled = scaler.transform(X)
            pred     = int(model.predict(X_scaled)[0])
            prob     = float(model.predict_proba(X_scaled)[0][1])
            results[modality] = {
                "prediction": pred,
                "label":      "PHISHING" if pred == 1 else "LEGITIMATE",
                "phishing_probability": round(prob, 4),
            }
        except Exception as e:
            results[modality] = {"error": str(e)}

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Test a single URL against FEDrA baseline models."
    )
    parser.add_argument("--url", required=True, help="URL to classify")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  FEDrA — Single URL Prediction")
    print(f"{'='*60}")
    print(f"  URL: {args.url}\n")

    results = predict_url(args.url)

    votes = []
    for modality, res in results.items():
        if "error" in res:
            print(f"  [{modality.upper():5s}] ❌ {res['error']}")
        else:
            icon = "🚨" if res["prediction"] == 1 else "✅"
            print(f"  [{modality.upper():5s}] {icon}  {res['label']:<12}"
                  f"  (phishing prob: {res['phishing_probability']:.4f})")
            votes.append(res["prediction"])

    if votes:
        majority = int(sum(votes) > len(votes) / 2)
        verdict  = "🚨 PHISHING" if majority == 1 else "✅ LEGITIMATE"
        print(f"\n  Majority vote → {verdict}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
