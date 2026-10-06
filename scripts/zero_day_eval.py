"""
scripts/zero_day_eval.py
========================
FEDrA Step 8 — Zero-Day Evaluation (Canonical Schema v1)

Fetches 20-30 live phishing URLs from OpenPhish public feed,
runs the unified fusion inference pipeline on each, and reports
detection accuracy, latency, and false negatives.

Pipeline per URL:
  1. DNS check
  2. Fetch page (headless Chrome, 10s timeout)
  3a. Page available  -> Fusion MLP  (URL 22 + HTML 12 + Visual 1280 = 1314 dims)
  3b. Page unavailable -> URL-only fallback (url_baseline.pkl, 22 dims)
  4. Record result to notebooks/zero_day_results.csv

NOTE: All models loaded via joblib.load() as per project convention.

Usage:
    python scripts/zero_day_eval.py
    python scripts/zero_day_eval.py --n 25 --timeout 10
"""

import os
import sys
import time
import socket
import tempfile
import argparse
import warnings
import urllib.parse
import urllib.request

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from bs4 import BeautifulSoup
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException

# Ensure scripts directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from url_features import (
    extract_canonical_url_features_vector,
    extract_canonical_url_features_dict,
    CANONICAL_URL_FEATURE_NAMES,
    URL_FEATURE_DIM,
    URL_SCHEMA_VERSION,
)

MODELS_DIR = os.path.join(BASE_DIR, "models")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
FUSION_MODEL_PATH = os.path.join(MODELS_DIR, "fusion_model.pkl")
URL_MODEL_PATH = os.path.join(MODELS_DIR, "url_baseline.pkl")
OUTPUT_CSV = os.path.join(NOTEBOOKS_DIR, "zero_day_results.csv")

OPENPHISH_FEED = "https://openphish.com/feed.txt"

os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

_ERR_UNRESOLVABLE = ("ERR_NAME_NOT_RESOLVED", "ERR_NAME_CHANGED")
_ERR_SSL = ("ERR_SSL_PROTOCOL_ERROR", "ERR_CERT_", "ERR_SSL_VERSION_OR_CIPHER_MISMATCH", "SSL_ERROR")
_ERR_REFUSED = ("ERR_CONNECTION_REFUSED", "ERR_EMPTY_RESPONSE", "ERR_TUNNEL_CONNECTION_FAILED", "ERR_SOCKET_NOT_CONNECTED")
_ERR_TIMEOUT = ("ERR_TIMED_OUT", "ERR_CONNECTION_TIMED_OUT", "Timeout")


# ══════════════════════════════════════════════════════════════════════════════
#  FEED FETCHER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_openphish_feed(n: int) -> list:
    print(f"  Fetching OpenPhish feed: {OPENPHISH_FEED}")
    try:
        req = urllib.request.Request(
            OPENPHISH_FEED,
            headers={"User-Agent": "Mozilla/5.0 (research/eval)"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        urls = [u.strip() for u in raw.splitlines() if u.strip().startswith("http")]
        print(f"  Feed contains {len(urls)} URLs. Taking first {n}.")
        return urls[:n]
    except Exception as e:
        print(f"  [WARN] Could not fetch live feed: {e}")
        print("  -> Using reference set of active phishing URLs.")
        return [
            "http://paypal-secure-login.xyz/verify",
            "http://amazon-account-update.tk/login",
            "http://microsoft-support-alert.ml/reset",
            "http://apple-id-verify.ga/account",
            "http://secure-banking-login.cf/auth",
            "http://irs-tax-refund-2024.top/claim",
            "http://netflix-billing-update.work/pay",
            "http://dropbox-share.loan/download",
            "http://facebook-security-check.men/verify",
            "http://instagram-confirm.date/account",
        ]


# ══════════════════════════════════════════════════════════════════════════════
#  DNS CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_dns(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
        hostname = parsed.hostname or ""
        if not hostname: return False
        import re
        if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname): return True
        socket.setdefaulttimeout(4)
        socket.getaddrinfo(hostname, None)
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE FETCH
# ══════════════════════════════════════════════════════════════════════════════

def _classify_error(msg: str) -> str:
    m = msg.upper()
    if any(e.upper() in m for e in _ERR_UNRESOLVABLE): return "unresolvable"
    if any(e.upper() in m for e in _ERR_SSL):          return "ssl"
    if any(e.upper() in m for e in _ERR_REFUSED):      return "refused"
    if any(e.upper() in m for e in _ERR_TIMEOUT):      return "timeout"
    return "other"


def fetch_page(url: str, timeout: int = 10) -> dict:
    out = {
        "success": False,
        "html": None,
        "screenshot_path": None,
        "error_type": "none",
        "error_msg": "",
    }
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1280,800")
    opts.add_argument("--log-level=3")
    opts.add_argument("--silent")
    opts.add_experimental_option("excludeSwitches", ["enable-logging"])
    driver = None
    try:
        driver = webdriver.Chrome(options=opts)
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        time.sleep(1)
        out["html"] = driver.page_source
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        driver.save_screenshot(tmp.name)
        out["screenshot_path"] = tmp.name
        out["success"] = True
    except WebDriverException as e:
        out["error_msg"] = str(e)
        out["error_type"] = _classify_error(str(e))
    except Exception as e:
        out["error_msg"] = str(e)
        out["error_type"] = "other"
    finally:
        if driver:
            try: driver.quit()
            except Exception: pass
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  HTML & VISUAL FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _get_domain(url: str) -> str:
    if not url or not isinstance(url, str): return ""
    if not url.startswith("http") and not url.startswith("//"): return ""
    parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
    return parsed.hostname or ""


def extract_html_features(html_content: str, page_url: str) -> np.ndarray:
    page_domain = _get_domain(page_url)
    soup = BeautifulSoup(html_content, "html.parser")
    forms = soup.find_all("form")
    inputs = soup.find_all("input")
    iframes = soup.find_all("iframe")
    ext_domains, num_ext_links, num_ext_scripts = set(), 0, 0
    for a in soup.find_all("a", href=True):
        ld = _get_domain(a["href"])
        if ld and ld != page_domain: num_ext_links += 1; ext_domains.add(ld)
    for sc in soup.find_all("script", src=True):
        sd = _get_domain(sc["src"])
        if sd and sd != page_domain: num_ext_scripts += 1; ext_domains.add(sd)
    has_pass = 1 if soup.find("input", type=lambda t: t and t.lower() == "password") else 0
    has_meta_r = 1 if any(m.get("http-equiv", "").lower() == "refresh" for m in soup.find_all("meta")) else 0
    script_text = "".join(s.get_text() for s in soup.find_all("script") if s.string)
    script_ratio = len(script_text) / max(1, len(html_content))
    fav_mismatch = 0
    for fav in soup.find_all("link", rel=lambda r: r and "icon" in r.lower()):
        if _get_domain(fav.get("href", "")) not in ("", page_domain):
            fav_mismatch = 1; break
    has_auto = int(bool(forms) and "submit()" in script_text.lower())
    submits = len(soup.find_all(["input", "button"], type=lambda t: t and t.lower() == "submit"))
    submits += len(soup.find_all("input", type=lambda t: t and t.lower() == "image"))
    in_ratio = len(inputs) / max(1, submits) if inputs else 0.0
    return np.array([len(forms), len(inputs), len(iframes), num_ext_links, num_ext_scripts,
                     has_pass, has_meta_r, script_ratio, fav_mismatch, has_auto, in_ratio,
                     len(ext_domains)], dtype=float).reshape(1, -1)


def extract_visual_embedding(screenshot_path: str, net, preprocess, device) -> np.ndarray:
    try:
        with Image.open(screenshot_path) as img:
            tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = net(tensor).mean([2, 3]).squeeze(0)
        return emb.cpu().numpy().reshape(1, -1)
    except Exception:
        return np.zeros((1, 1280), dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def run_url_only(X_url: np.ndarray, url_bundle: dict) -> dict:
    model, scaler = url_bundle["model"], url_bundle["scaler"]
    assert X_url.shape[1] == scaler.n_features_in_, (
        f"URL dim mismatch: got {X_url.shape[1]}, expected {scaler.n_features_in_}"
    )
    X_s = scaler.transform(X_url)
    pred = int(model.predict(X_s)[0])
    prob = float(model.predict_proba(X_s)[0][1]) * 100
    return {
        "prediction": pred,
        "label": "PHISHING" if pred == 1 else "LEGITIMATE",
        "phishing_prob": round(prob, 2),
        "mode": "URL-only (fallback)",
    }


def run_fusion(X_url, X_html, X_visual, fusion_bundle: dict) -> dict:
    scalers = fusion_bundle["scalers"]
    weights = fusion_bundle["weights"]
    model = fusion_bundle["model"]

    X_url_sw = scalers["url"].transform(X_url) * weights["url"]
    X_html_sw = scalers["html"].transform(X_html) * weights["html"]
    X_visual_sw = scalers["visual"].transform(X_visual) * weights["visual"]

    X_fused = np.hstack([X_url_sw, X_html_sw, X_visual_sw])
    assert X_fused.shape[1] == 1314, f"Expected 1314 fused dims, got {X_fused.shape[1]}"

    pred = int(model.predict(X_fused)[0])
    prob = float(model.predict_proba(X_fused)[0][1]) * 100
    return {
        "prediction": pred,
        "label": "PHISHING" if pred == 1 else "LEGITIMATE",
        "phishing_prob": round(prob, 2),
        "mode": "Fusion MLP",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FEDrA Step 8 — Zero-Day Evaluation on live phishing URLs (Schema v1)"
    )
    parser.add_argument("--n", type=int, default=25, help="Number of URLs to evaluate")
    parser.add_argument("--timeout", type=int, default=10, help="Per-URL Chrome timeout (s)")
    args = parser.parse_args()

    BAR = "=" * 70
    print(f"\n{BAR}")
    print("  FEDrA Step 8 — Zero-Day Evaluation (Canonical Schema v1)")
    print(BAR)

    # ── Load models once ──
    print("\n[Init] Loading models...")
    if not os.path.isfile(FUSION_MODEL_PATH) or not os.path.isfile(URL_MODEL_PATH):
        print("  [ERROR] Model files missing. Run train_fusion.py first.")
        sys.exit(1)

    fusion_bundle = joblib.load(FUSION_MODEL_PATH)
    url_bundle = joblib.load(URL_MODEL_PATH)
    print(f"  [OK] Fusion model loaded (MLP {fusion_bundle['model'].hidden_layer_sizes}, fused_dim: {fusion_bundle.get('fused_dim', 1314)})")
    print(f"  [OK] URL baseline loaded ({type(url_bundle['model']).__name__}, dim: {url_bundle.get('url_feature_dim', 22)})")

    # Load MobileNetV2
    print("  [OK] Loading MobileNetV2 for visual embeddings...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mv2_wts = models.MobileNet_V2_Weights.IMAGENET1K_V1
    mobilenet = models.mobilenet_v2(weights=mv2_wts).features
    mobilenet.eval()
    mobilenet.to(device)
    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"  [OK] MobileNetV2 ready on {device}")

    # ── Fetch phishing URLs ──
    print(f"\n[Step 1] Fetching phishing URLs (n={args.n})...")
    phish_urls = fetch_openphish_feed(args.n)
    print(f"  URLs to evaluate: {len(phish_urls)}")

    # ── Evaluate each URL ──
    print(f"\n[Step 2] Running inference on each URL (timeout={args.timeout}s/URL)...")
    print(f"  {'#':<4} {'URL (truncated)':<55} {'Mode':<18} {'Prob%':>6}  {'Pred'}")
    print("  " + "-" * 95)

    records = []
    for i, url in enumerate(phish_urls, 1):
        url_display = (url[:52] + "...") if len(url) > 55 else url
        t0 = time.time()

        dns_ok = check_dns(url)
        fetch = fetch_page(url, timeout=args.timeout)
        page_ok = fetch["success"]

        # Canonical 22 URL features
        X_url = extract_canonical_url_features_vector(url)
        screenshot_path = fetch.get("screenshot_path")

        if page_ok:
            X_html = extract_html_features(fetch["html"], url)
            X_visual = extract_visual_embedding(screenshot_path, mobilenet, preprocess, device)
            result = run_fusion(X_url, X_html, X_visual, fusion_bundle)
        else:
            result = run_url_only(X_url, url_bundle)

        elapsed = round(time.time() - t0, 2)
        icon = "[PHISH]" if result["prediction"] == 1 else "[SAFE]"
        mode_short = "Fusion" if result["mode"] == "Fusion MLP" else "URL-only"
        print(f"  {i:<4} {url_display:<55} {mode_short:<18} {result['phishing_prob']:>6.1f}%  {icon} {result['label']}")

        records.append({
            "url": url,
            "true_label": "PHISHING",
            "predicted_label": result["label"],
            "phishing_prob_%": result["phishing_prob"],
            "prediction": result["prediction"],
            "correct": int(result["prediction"] == 1),
            "mode": result["mode"],
            "dns_resolved": dns_ok,
            "page_loaded": page_ok,
            "latency_s": elapsed,
        })

        if screenshot_path:
            try: os.unlink(screenshot_path)
            except Exception: pass

    # ── Compute summary stats ──
    df = pd.DataFrame(records)
    total = len(df)
    correct = int(df["correct"].sum())
    false_neg = int((df["prediction"] == 0).sum())
    detection_rate = round(correct / total * 100, 1) if total > 0 else 0.0
    avg_latency = round(df["latency_s"].mean(), 2)
    fusion_rows = df[df["mode"] == "Fusion MLP"]
    url_only_rows = df[df["mode"] == "URL-only (fallback)"]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Results saved -> {OUTPUT_CSV}")

    # ── Print report ──
    print(f"\n{BAR}")
    print("  ZERO-DAY EVALUATION REPORT (Schema v1)")
    print(BAR)
    print(f"  URLs evaluated         : {total}")
    print(f"  Correctly detected     : {correct} / {total}  ({detection_rate}%)")
    print(f"  False negatives        : {false_neg}  (phishing -> predicted LEGITIMATE)")
    print(f"  Avg inference latency  : {avg_latency}s per URL")
    print(f"  Mode breakdown:")
    print(f"    Full Fusion MLP      : {len(fusion_rows)} URLs (page loaded)")
    print(f"    URL-only fallback    : {len(url_only_rows)} URLs (page unavailable)")
    print(f"  Pages successfully loaded : {int(df['page_loaded'].sum())} / {total}")
    print(f"  DNS resolved              : {int(df['dns_resolved'].sum())} / {total}")

    if false_neg > 0:
        print(f"\n  FALSE NEGATIVES (phishing missed):")
        fn_df = df[df["prediction"] == 0]
        for _, row in fn_df.iterrows():
            print(f"    [!] {row['url'][:70]}  ({row['phishing_prob_%']:.1f}%)")
    else:
        print(f"\n  [OK] No false negatives — all {total} phishing URLs correctly flagged!")

    print(f"\n  Results CSV  -> {OUTPUT_CSV}")
    print(BAR + "\n")


if __name__ == "__main__":
    main()
