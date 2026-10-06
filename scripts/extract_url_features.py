"""
scripts/extract_url_features.py
================================
Extracts canonical 22 URL-level features from Dataset/manifest.csv as per FEDrA v1 schema.
Single source of truth imported from scripts.url_features.
"""

import os
import sys
import pandas as pd
from tqdm import tqdm

# Ensure scripts directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from url_features import (
    extract_canonical_url_features_dict,
    CANONICAL_URL_FEATURE_NAMES,
    URL_SCHEMA_VERSION,
    URL_FEATURE_DIM,
)

# ── Config ──────────────────────────────────────────────────────────────
MANIFEST = os.path.join(BASE_DIR, "Dataset", "manifest.csv")
OUT_DIR = os.path.join(BASE_DIR, "Dataset", "features")
OUT_CSV = os.path.join(OUT_DIR, "url_features.csv")

os.makedirs(OUT_DIR, exist_ok=True)


# ── Main ─────────────────────────────────────────────────────────────────
def run():
    print(f"Reading manifest from {MANIFEST}...")
    df = pd.read_csv(MANIFEST)

    features_list = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting canonical URL features"):
        raw_dict = extract_canonical_url_features_dict(row["url"])
        feat = {k: raw_dict[k] for k in CANONICAL_URL_FEATURE_NAMES}
        feat["sample_id"] = row["sample_id"]
        features_list.append(feat)

    out_df = pd.DataFrame(features_list)
    cols = ["sample_id"] + CANONICAL_URL_FEATURE_NAMES
    out_df = out_df[cols]

    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] Saved {len(out_df)} records to {OUT_CSV}")
    print(f"     Schema Version: {URL_SCHEMA_VERSION}")
    print(f"     Feature Dimensions: {URL_FEATURE_DIM} (Total columns: {out_df.shape[1]})")
    print(f"     Columns: {list(out_df.columns)}")


if __name__ == "__main__":
    run()
