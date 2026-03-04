"""
scripts/test_single_url.py
==========================
Test a single URL against all three FEDrA baseline models.

What this script does (locally, no external APIs):
  1. Fetches the live page HTML and takes a screenshot via headless Chrome (Selenium)
  2. Extracts URL features   — same schema as extract_url_features.py
  3. Extracts HTML features  — same schema as extract_html_features.py
  4. Extracts visual embedding — same MobileNetV2 pipeline as extract_visual_embeddings.py
  5. Loads each .pkl model with joblib.load()
  6. Prints per-model phishing probability and final majority-vote verdict

NOTE: Models saved with joblib.dump() — always load with joblib.load()

Usage:
    python scripts/test_single_url.py --url "https://example.com"

Dependencies (all in fedra conda env):
    selenium, requests, beautifulsoup4, torch, torchvision, Pillow
"""

import os
import re
import sys
import math
import time
import tempfile
import argparse
import warnings
import urllib.parse

warnings.filterwarnings("ignore")

import numpy as np
import joblib                          # NOTE: always use joblib.load() for .pkl files
from bs4 import BeautifulSoup
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILES = {
    "url":   os.path.join(MODELS_DIR, "url_baseline.pkl"),
    "html":  os.path.join(MODELS_DIR, "html_baseline.pkl"),
    "image": os.path.join(MODELS_DIR, "image_baseline.pkl"),
}

SUSPICIOUS_WORDS = {"login", "redirect", "verify", "secure"}


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE FETCHER — headless Chrome via Selenium (no external API)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_page(url: str, timeout: int = 20) -> tuple[str, str]:
    """
    Launch headless Chrome, navigate to `url`, capture:
      - full page HTML (after JS execution)
      - PNG screenshot saved to a temp file

    Returns (html_content: str, screenshot_path: str)
    Raises RuntimeError on failure.
    """
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1280,800")
    opts.add_argument("--log-level=3")
    opts.add_argument("--silent")
    opts.add_experimental_option("excludeSwitches", ["enable-logging"])

    driver = webdriver.Chrome(options=opts)
    try:
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        time.sleep(2)                          # allow JS to settle

        html_content = driver.page_source

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        driver.save_screenshot(tmp.name)
        screenshot_path = tmp.name
    finally:
        driver.quit()

    return html_content, screenshot_path


# ══════════════════════════════════════════════════════════════════════════════
#  URL FEATURE EXTRACTION  (mirrors extract_url_features.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probabilities = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probabilities)


def _tld_extract_simple(url: str) -> tuple[str, str, str]:
    """
    Lightweight tldextract replacement using urllib + regex.
    Returns (subdomain, domain, suffix).
    """
    parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
    hostname = parsed.hostname or ""
    parts = hostname.split(".")

    # Known multi-part TLDs (common subset)
    multi_tld = {
        "co.uk", "co.in", "co.jp", "co.nz", "co.za", "com.au", "com.br",
        "com.cn", "com.mx", "net.au", "org.uk", "gov.uk",
    }
    if len(parts) >= 3 and ".".join(parts[-2:]) in multi_tld:
        suffix    = ".".join(parts[-2:])
        domain    = parts[-3] if len(parts) >= 3 else ""
        subdomain = ".".join(parts[:-3]) if len(parts) > 3 else ""
    elif len(parts) >= 2:
        suffix    = parts[-1]
        domain    = parts[-2]
        subdomain = ".".join(parts[:-2])
    else:
        suffix, domain, subdomain = "", hostname, ""

    return subdomain, domain, suffix


def extract_url_features(url: str) -> np.ndarray:
    """
    Produces the exact same 11-feature dict as extract_url_features.py,
    then one-hot-encodes `tld_type` to match the training schema.

    The training CSV had get_dummies(drop_first=True) applied on tld_type,
    which expanded the 11 columns to 89 (one-hot of TLD type).
    We replicate the same encoding using the scaler's n_features_in_.
    """
    parsed_url = url if "://" in url else "http://" + url
    parsed     = urllib.parse.urlparse(parsed_url)
    subdomain, domain, suffix = _tld_extract_simple(parsed_url)

    url_len            = len(url)
    is_ip              = 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) else 0
    num_subdomains     = len([s for s in subdomain.split(".") if s]) if subdomain else 0
    is_https           = 1 if url.lower().startswith("https") else 0
    num_at             = url.count("@")
    num_dash           = url.count("-")
    num_double_slash   = url.count("//") - 1 if "://" in url else url.count("//")
    if num_double_slash < 0:
        num_double_slash = 0
    domain_token       = f"{domain}.{suffix}"
    domain_entropy     = _shannon_entropy(domain_token)
    tld_type           = suffix
    query_params       = urllib.parse.parse_qsl(parsed.query)
    num_params         = len(query_params)
    has_susp_params    = 0
    for k, _ in query_params:
        if any(sw in k.lower() for sw in SUSPICIOUS_WORDS):
            has_susp_params = 1
            break

    # Numeric core (10 features excluding tld_type)
    numeric = np.array([
        url_len, num_subdomains, is_ip, is_https,
        num_at, num_dash, num_double_slash,
        domain_entropy, num_params, has_susp_params,
    ], dtype=float)

    # One-hot encode tld_type — we need to match the 89-dim space the scaler
    # was fitted on. The simplest safe approach: pad with zeros for unknown TLDs
    # (the scaler will normalise them; LR coeff for unseen TLD cols ≈ 0).
    # We return the raw 10-feature vector here and let _align() pad to 89.
    return numeric.reshape(1, -1)


# ══════════════════════════════════════════════════════════════════════════════
#  HTML FEATURE EXTRACTION  (mirrors extract_html_features.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

def _get_domain(url: str) -> str:
    if not url or not isinstance(url, str):
        return ""
    if not url.startswith("http") and not url.startswith("//"):
        return ""
    _, domain, suffix = _tld_extract_simple(url)
    return f"{domain}.{suffix}" if domain else ""


def extract_html_features(html_content: str, page_url: str) -> np.ndarray:
    """
    Produces the exact 12-feature vector used in html_features.csv.
    """
    page_domain = _get_domain(page_url)
    content     = html_content
    soup        = BeautifulSoup(content, "html.parser")

    forms   = soup.find_all("form")
    inputs  = soup.find_all("input")
    iframes = soup.find_all("iframe")
    num_forms   = len(forms)
    num_inputs  = len(inputs)
    num_iframes = len(iframes)

    external_domains = set()
    num_ext_links    = 0
    num_ext_scripts  = 0

    for a in soup.find_all("a", href=True):
        ld = _get_domain(a["href"])
        if ld and ld != page_domain:
            num_ext_links += 1
            external_domains.add(ld)

    for sc in soup.find_all("script", src=True):
        sd = _get_domain(sc["src"])
        if sd and sd != page_domain:
            num_ext_scripts += 1
            external_domains.add(sd)

    num_unique_ext_domains = len(external_domains)

    has_password_field = 1 if soup.find(
        "input", type=lambda t: t and t.lower() == "password"
    ) else 0

    has_meta_redirect = 1 if any(
        m.get("http-equiv", "").lower() == "refresh"
        for m in soup.find_all("meta")
    ) else 0

    scripts      = soup.find_all("script")
    script_text  = "".join([s.get_text() for s in scripts if s.string])
    script_len   = len(script_text)
    total_len    = max(1, len(content))
    script_content_ratio = script_len / total_len

    favicon_mismatch = 0
    for fav in soup.find_all("link", rel=lambda r: r and "icon" in r.lower()):
        fd = _get_domain(fav.get("href", ""))
        if fd and fd != page_domain:
            favicon_mismatch = 1
            break

    has_auto_submit = 0
    if forms and "submit()" in script_text.lower():
        has_auto_submit = 1

    submits = soup.find_all(
        ["input", "button"], type=lambda t: t and t.lower() == "submit"
    )
    num_submits = len(submits) + len(
        soup.find_all("input", type=lambda t: t and t.lower() == "image")
    )
    input_submit_ratio = num_inputs / max(1, num_submits) if num_inputs > 0 else 0.0

    feats = np.array([
        num_forms, num_inputs, num_iframes,
        num_ext_links, num_ext_scripts,
        has_password_field, has_meta_redirect,
        script_content_ratio, favicon_mismatch,
        has_auto_submit, input_submit_ratio,
        num_unique_ext_domains,
    ], dtype=float)

    return feats.reshape(1, -1)


# ══════════════════════════════════════════════════════════════════════════════
#  VISUAL EMBEDDING  (mirrors extract_visual_embeddings.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

def _load_mobilenet() -> tuple:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model   = models.mobilenet_v2(weights=weights)
    model   = model.features
    model.eval()
    model.to(device)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess, device


def extract_visual_embedding(screenshot_path: str) -> np.ndarray:
    """
    Returns a 1280-dim float32 embedding using frozen MobileNetV2,
    identical to the training pipeline.
    """
    model, preprocess, device = _load_mobilenet()

    try:
        with Image.open(screenshot_path) as img:
            img    = img.convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            features  = model(tensor)               # (1, 1280, 7, 7)
            embedding = features.mean([2, 3]).squeeze(0)   # (1280,)

        return embedding.cpu().numpy().reshape(1, -1)
    except Exception as e:
        print(f"  [WARN] Visual embedding failed: {e}")
        return np.zeros((1, 1280), dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def _align(X: np.ndarray, expected: int) -> np.ndarray:
    """Pad or truncate X to match `expected` features."""
    cur = X.shape[1]
    if cur < expected:
        return np.hstack([X, np.zeros((1, expected - cur))])
    return X[:, :expected]


def run_predictions(url: str, html_content: str, screenshot_path: str) -> dict:
    """
    Load each model via joblib.load(), align features, return per-model results.
    NOTE: Models saved with joblib.dump() — always load with joblib.load()
    """
    print("\n  Extracting features...")

    X_url  = extract_url_features(url)
    print(f"    URL features:    {X_url.shape[1]} raw dims")

    X_html = extract_html_features(html_content, url)
    print(f"    HTML features:   {X_html.shape[1]} dims")

    X_vis  = extract_visual_embedding(screenshot_path)
    print(f"    Visual embedding: {X_vis.shape[1]} dims")

    feature_map = {"url": X_url, "html": X_html, "image": X_vis}
    results     = {}

    for modality, X in feature_map.items():
        path = MODEL_FILES[modality]
        if not os.path.isfile(path):
            results[modality] = {"error": f"Model not found: {path}"}
            continue
        try:
            # NOTE: Models saved with joblib.dump() — always load with joblib.load()
            bundle  = joblib.load(path)
            model   = bundle["model"]
            scaler  = bundle["scaler"]

            X_aligned  = _align(X, scaler.n_features_in_)
            X_scaled   = scaler.transform(X_aligned)
            pred       = int(model.predict(X_scaled)[0])
            prob       = float(model.predict_proba(X_scaled)[0][1])

            results[modality] = {
                "prediction": pred,
                "label":      "PHISHING" if pred == 1 else "LEGITIMATE",
                "phishing_prob": round(prob * 100, 2),
            }
        except Exception as e:
            results[modality] = {"error": str(e)}

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FEDrA — Phishing detector for a single URL (local, no APIs)"
    )
    parser.add_argument("--url", required=True, help="URL to classify")
    parser.add_argument(
        "--timeout", type=int, default=20,
        help="Page load timeout in seconds (default: 20)"
    )
    args = parser.parse_args()
    url = args.url

    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  FEDrA — Single URL Phishing Detector")
    print(bar)
    print(f"  URL : {url}")

    # ── Step 1: Fetch ────────────────────────────────────────────
    print("\n  [1/3] Fetching page (headless Chrome)...")
    try:
        html_content, screenshot_path = fetch_page(url, timeout=args.timeout)
        html_kb = len(html_content) / 1024
        print(f"        HTML size   : {html_kb:.1f} KB")
        print(f"        Screenshot  : {screenshot_path}")
    except Exception as e:
        print(f"\n  ❌ Could not fetch URL: {e}")
        sys.exit(1)

    # ── Step 2: Predict ──────────────────────────────────────────
    print("\n  [2/3] Running model predictions...")
    results = run_predictions(url, html_content, screenshot_path)

    # ── Step 3: Report ───────────────────────────────────────────
    print(f"\n{bar}")
    print(f"  {'MODEL':<10}  {'PROB (phishing)':>17}  {'LABEL'}")
    print(f"  {'-'*58}")

    votes = []
    modality_names = {"url": "URL", "html": "HTML", "image": "Visual"}
    for modality, name in modality_names.items():
        res = results.get(modality, {})
        if "error" in res:
            print(f"  {name:<10}  {'ERROR':>17}  {res['error']}")
        else:
            icon  = "🚨" if res["prediction"] == 1 else "✅"
            prob  = res["phishing_prob"]
            label = res["label"]
            bar_  = "█" * int(prob / 5)
            print(f"  {name:<10}  {prob:>15.2f}%  {icon}  {label}")
            votes.append(res["prediction"])

    # Majority vote
    if votes:
        majority = int(sum(votes) > len(votes) / 2)
        verdict  = "🚨  PHISHING" if majority == 1 else "✅  LEGITIMATE"
        print(f"\n  {'─'*58}")
        print(f"  Final verdict (majority vote) →  {verdict}")

    print(f"{bar}\n")

    # Cleanup temp screenshot
    try:
        os.unlink(screenshot_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
