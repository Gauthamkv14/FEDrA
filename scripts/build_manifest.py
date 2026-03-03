"""
scripts/build_manifest.py
=========================
Step 2 of the FEDrA pipeline.

Walks legit_dataset/ (label=0) and phised_dataset/ (label=1),
extracts the URL from each folder's metadata.txt, records absolute
paths to page.html and screenshot.png, and writes Dataset/manifest.csv.

IMPORTANT: folder names are NEVER used as features – they are only
used as directory paths to locate the files inside.
"""

import os
import re
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

SOURCES = [
    (os.path.join(DATASET_DIR, "legit_dataset"),   0),
    (os.path.join(DATASET_DIR, "phised_dataset"),  1),
]

OUTPUT_CSV = os.path.join(DATASET_DIR, "manifest.csv")

# ── Helpers ──────────────────────────────────────────────────────────────
def extract_url_from_metadata(metadata_path: str) -> str:
    """
    Parse the metadata.txt file and return the best URL string found.

    The file may contain lines like:
        Original URL: github.com
        URL: https://github.com
        Label: 0
        Status: Success
    Priority: 'URL:' line (full URL with scheme) > 'Original URL:' line.
    Returns empty string if nothing is found.
    """
    url = ""
    original_url = ""

    try:
        with open(metadata_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                # Prefer the 'URL:' field (has scheme like https://)
                m = re.match(r"^URL:\s*(.+)$", line, re.IGNORECASE)
                if m:
                    url = m.group(1).strip()
                # Fallback to 'Original URL:'
                m2 = re.match(r"^Original URL:\s*(.+)$", line, re.IGNORECASE)
                if m2:
                    original_url = m2.group(1).strip()
    except Exception as e:
        print(f"  [WARN] Could not read {metadata_path}: {e}")

    return url if url else original_url


# ── Main ─────────────────────────────────────────────────────────────────
def build_manifest() -> pd.DataFrame:
    rows = []
    sample_id = 0

    for dataset_path, label in SOURCES:
        if not os.path.isdir(dataset_path):
            print(f"[ERROR] Directory not found: {dataset_path}")
            continue

        # Sort for reproducibility
        folders = sorted(os.listdir(dataset_path))

        for folder_name in folders:
            folder_path = os.path.join(dataset_path, folder_name)
            if not os.path.isdir(folder_path):
                continue  # skip stray files

            metadata_path    = os.path.join(folder_path, "metadata.txt")
            html_path        = os.path.join(folder_path, "page.html")
            screenshot_path  = os.path.join(folder_path, "screenshot.png")

            # Skip if metadata is missing (no URL to extract)
            if not os.path.isfile(metadata_path):
                print(f"  [SKIP] No metadata.txt in {folder_path}")
                continue

            url = extract_url_from_metadata(metadata_path)
            if not url:
                print(f"  [WARN] Could not extract URL from {metadata_path}")

            rows.append({
                "sample_id":       sample_id,
                "label":           label,
                "url":             url,
                "html_path":       html_path if os.path.isfile(html_path) else "",
                "screenshot_path": screenshot_path if os.path.isfile(screenshot_path) else "",
            })
            sample_id += 1

    df = pd.DataFrame(rows, columns=[
        "sample_id", "label", "url", "html_path", "screenshot_path"
    ])
    return df


if __name__ == "__main__":
    print("Building manifest.csv …")
    df = build_manifest()

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {OUTPUT_CSV}")
    print(f"Shape : {df.shape}  ({df['label'].value_counts().to_dict()})")
    print("\nFirst 5 rows:")
    pd.set_option("display.max_colwidth", 60)
    print(df.head(5).to_string(index=False))
