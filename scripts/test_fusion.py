"""
scripts/test_fusion.py
======================
FEDrA Step 7 — Fusion Model Inference for a Single URL

Uses the canonical 22-dimensional URL feature schema (v1).
Uses ONLY the fusion model (models/fusion_model.pkl).

What it does:
  1. Extract canonical URL features     (22 features via scripts.url_features)
  2. Fetch page & extract HTML features (12 features via BeautifulSoup)
  3. Extract visual embedding           (1280-dim MobileNetV2 GAP)
  4. Apply per-modality scalers + weights (URL=0.4, HTML=0.3, Visual=0.3)
  5. Concatenate (1314 dims) and run through the fusion MLP
  6. Print a clean fusion-specific report

Usage:
    python scripts/test_fusion.py --url "https://www.google.com"
    python scripts/test_fusion.py --url "https://suspicious-login.xyz" --timeout 15

NOTE: All models are loaded with joblib.load() as per project convention.
"""

import os
import sys
import time
import socket
import tempfile
import argparse
import warnings
import urllib.parse

warnings.filterwarnings("ignore")

import numpy as np
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
FUSION_MODEL_PATH = os.path.join(MODELS_DIR, "fusion_model.pkl")
URL_MODEL_PATH = os.path.join(MODELS_DIR, "url_baseline.pkl")

_BAR = "=" * 65

_ERR_UNRESOLVABLE = ("ERR_NAME_NOT_RESOLVED", "ERR_NAME_CHANGED")
_ERR_SSL = ("ERR_SSL_PROTOCOL_ERROR", "ERR_CERT_", "ERR_SSL_VERSION_OR_CIPHER_MISMATCH", "SSL_ERROR")
_ERR_REFUSED = ("ERR_CONNECTION_REFUSED", "ERR_EMPTY_RESPONSE", "ERR_TUNNEL_CONNECTION_FAILED", "ERR_SOCKET_NOT_CONNECTED")
_ERR_TIMEOUT = ("ERR_TIMED_OUT", "ERR_CONNECTION_TIMED_OUT", "Timeout")


# ══════════════════════════════════════════════════════════════════════════════
#  DNS CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_dns(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
        hostname = parsed.hostname or ""
        if not hostname:
            return False
        import re
        if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname):
            return True
        socket.setdefaulttimeout(4)
        socket.getaddrinfo(hostname, None)
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE FETCH (headless Chrome)
# ══════════════════════════════════════════════════════════════════════════════

def _classify_error(msg: str) -> str:
    m = msg.upper()
    if any(e.upper() in m for e in _ERR_UNRESOLVABLE): return "unresolvable"
    if any(e.upper() in m for e in _ERR_SSL):          return "ssl"
    if any(e.upper() in m for e in _ERR_REFUSED):      return "refused"
    if any(e.upper() in m for e in _ERR_TIMEOUT):      return "timeout"
    return "other"


def fetch_page(url: str, timeout: int = 20) -> dict:
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
        time.sleep(2)
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
#  HTML FEATURE EXTRACTION (12 features)
# ══════════════════════════════════════════════════════════════════════════════

def _get_domain(url: str) -> str:
    if not url or not isinstance(url, str): return ""
    if not url.startswith("http") and not url.startswith("//"): return ""
    parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
    return parsed.hostname or ""


def extract_html_features(html_content: str, page_url: str) -> np.ndarray:
    page_domain = _get_domain(page_url)
    soup = BeautifulSoup(html_content, "html.parser")
    content = html_content

    forms = soup.find_all("form")
    inputs = soup.find_all("input")
    iframes = soup.find_all("iframe")
    num_forms, num_inputs, num_iframes = len(forms), len(inputs), len(iframes)

    ext_domains, num_ext_links, num_ext_scripts = set(), 0, 0
    for a in soup.find_all("a", href=True):
        ld = _get_domain(a["href"])
        if ld and ld != page_domain:
            num_ext_links += 1
            ext_domains.add(ld)
    for sc in soup.find_all("script", src=True):
        sd = _get_domain(sc["src"])
        if sd and sd != page_domain:
            num_ext_scripts += 1
            ext_domains.add(sd)

    num_unique_ext_domains = len(ext_domains)
    has_password_field = 1 if soup.find("input", type=lambda t: t and t.lower() == "password") else 0
    has_meta_redirect = 1 if any(
        m.get("http-equiv", "").lower() == "refresh" for m in soup.find_all("meta")
    ) else 0
    script_text = "".join(s.get_text() for s in soup.find_all("script") if s.string)
    script_content_ratio = len(script_text) / max(1, len(content))
    favicon_mismatch = 0
    for fav in soup.find_all("link", rel=lambda r: r and "icon" in r.lower()):
        fd = _get_domain(fav.get("href", ""))
        if fd and fd != page_domain:
            favicon_mismatch = 1
            break
    has_auto_submit = int(bool(forms) and "submit()" in script_text.lower())
    submits = len(soup.find_all(["input", "button"], type=lambda t: t and t.lower() == "submit"))
    submits += len(soup.find_all("input", type=lambda t: t and t.lower() == "image"))
    input_submit_ratio = num_inputs / max(1, submits) if num_inputs > 0 else 0.0

    return np.array([
        num_forms, num_inputs, num_iframes,
        num_ext_links, num_ext_scripts,
        has_password_field, has_meta_redirect,
        script_content_ratio, favicon_mismatch,
        has_auto_submit, input_submit_ratio,
        num_unique_ext_domains,
    ], dtype=float).reshape(1, -1)


# ══════════════════════════════════════════════════════════════════════════════
#  VISUAL EMBEDDING (1280-dim MobileNetV2)
# ══════════════════════════════════════════════════════════════════════════════

def extract_visual_embedding(screenshot_path: str) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    net = models.mobilenet_v2(weights=weights).features
    net.eval()
    net.to(device)

    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    try:
        with Image.open(screenshot_path) as img:
            tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = net(tensor).mean([2, 3]).squeeze(0)
        return emb.cpu().numpy().reshape(1, -1)
    except Exception as e:
        print(f"  [WARN] Visual embedding failed: {e}")
        return np.zeros((1, 1280), dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  URL-ONLY FALLBACK INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_url_only(X_url: np.ndarray) -> dict:
    """
    URL-only prediction using url_baseline.pkl.
    Direct 22-dimensional input with zero padding.
    """
    if not os.path.isfile(URL_MODEL_PATH):
        return {"error": f"URL model not found: {URL_MODEL_PATH}"}
    bundle = joblib.load(URL_MODEL_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]
    
    assert X_url.shape[1] == scaler.n_features_in_, (
        f"URL feature dimension mismatch: got {X_url.shape[1]}, expected {scaler.n_features_in_}"
    )
    
    X_s = scaler.transform(X_url)
    pred = int(model.predict(X_s)[0])
    prob = float(model.predict_proba(X_s)[0][1]) * 100
    return {
        "prediction": pred,
        "label": "PHISHING" if pred == 1 else "LEGITIMATE",
        "phishing_prob": round(prob, 2),
        "url_dim": X_url.shape[1],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FUSION INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_fusion(X_url: np.ndarray, X_html: np.ndarray, X_visual: np.ndarray,
               bundle: dict) -> dict:
    """
    Apply per-modality scalers + weights then pass through the fusion MLP.
    Direct concatenation: 22 + 12 + 1280 = 1314 dimensions.
    """
    scalers = bundle["scalers"]
    weights = bundle["weights"]
    model = bundle["model"]

    assert X_url.shape[1] == scalers["url"].n_features_in_, (
        f"URL dim mismatch: got {X_url.shape[1]}, expected {scalers['url'].n_features_in_}"
    )
    assert X_html.shape[1] == scalers["html"].n_features_in_, (
        f"HTML dim mismatch: got {X_html.shape[1]}, expected {scalers['html'].n_features_in_}"
    )
    assert X_visual.shape[1] == scalers["visual"].n_features_in_, (
        f"Visual dim mismatch: got {X_visual.shape[1]}, expected {scalers['visual'].n_features_in_}"
    )

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
        "fused_dim": X_fused.shape[1],
        "url_dim": X_url.shape[1],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FEDrA Step 7 — Fusion Model inference for a single URL"
    )
    parser.add_argument("--url", required=True, help="URL to classify")
    parser.add_argument("--timeout", type=int, default=20, help="Page load timeout in seconds")
    args = parser.parse_args()
    url = args.url

    print(f"\n{_BAR}")
    print("  FEDrA — Fusion Model (Step 7) — Single URL Inference (Schema v1)")
    print(_BAR)
    print(f"  URL     : {url}")
    print(f"  Model   : {FUSION_MODEL_PATH}")
    print(_BAR)

    # ── 0. Load fusion model ──
    if not os.path.isfile(FUSION_MODEL_PATH):
        print(f"\n[ERROR] Fusion model not found at {FUSION_MODEL_PATH}")
        print("        Run: python scripts/train_fusion.py first.")
        sys.exit(1)

    print("\n  [0/4] Loading fusion model bundle (joblib)...")
    bundle = joblib.load(FUSION_MODEL_PATH)
    weights = bundle["weights"]
    print(f"        Weights -> URL={weights['url']}  HTML={weights['html']}  Visual={weights['visual']}")
    print(f"        MLP architecture: {bundle['model'].hidden_layer_sizes}")
    print(f"        URL Schema Version: {bundle.get('schema_version', 'v1')} ({URL_FEATURE_DIM} features)")

    # ── 1. DNS check ──
    print("\n  [1/4] DNS resolution check...")
    dns_ok = check_dns(url)
    dns_icon = "OK (resolves)" if dns_ok else "UNRESOLVABLE (phishing signal)"
    print(f"        Domain -> {dns_icon}")

    # ── 2. Fetch page ──
    print(f"\n  [2/4] Fetching page (headless Chrome, timeout={args.timeout}s)...")
    fetch = fetch_page(url, timeout=args.timeout)

    page_available = False
    if fetch["success"]:
        html_kb = len(fetch["html"]) / 1024
        print(f"        [OK] Loaded — HTML: {html_kb:.1f} KB")
        page_available = True
    else:
        etype = fetch["error_type"]
        print(f"        [WARN] Fetch failed [{etype}]: {fetch['error_msg'][:80]}")

    # ── 3. URL feature extraction (Canonical 22 features, zero padding) ──
    print("\n  [3/4] Extracting canonical URL features...")
    X_url = extract_canonical_url_features_vector(url)
    print(f"        URL features : {X_url.shape[1]} dims (exact canonical schema v1)")

    # ── 4. Decide: Fusion or URL-only fallback ──
    if not page_available:
        print("\n  [4/4] Page unavailable -> switching to URL-only detection")
        print(f"        Loading URL baseline model: {URL_MODEL_PATH}")
        result = run_url_only(X_url)
        mode_used = "URL-only (fallback)"

        if "error" in result:
            print(f"        [ERROR] URL model error: {result['error']}")
            sys.exit(1)

        prob = result["phishing_prob"]
        pred = result["prediction"]
        label = result["label"]
        icon = "[PHISH]" if pred == 1 else "[SAFE]"

        if prob >= 80:   conf = "HIGH CONFIDENCE"
        elif prob >= 55: conf = "MEDIUM CONFIDENCE"
        elif prob <= 20: conf = "HIGH CONFIDENCE (likely safe)"
        elif prob <= 45: conf = "MEDIUM CONFIDENCE (likely safe)"
        else:            conf = "LOW CONFIDENCE — borderline"

        print(f"\n{_BAR}")
        print(f"  VERDICT  [Mode: {mode_used}]")
        print(_BAR)
        print(f"  {'Modality':<20} {'Dims':>6}  {'Status'}")
        print(f"  {'-'*45}")
        print(f"  {'URL':<20} {X_url.shape[1]:>6}  live canonical features")
        print(f"  {'HTML':<20} {'—':>6}  skipped (page unavailable)")
        print(f"  {'Visual':<20} {'—':>6}  skipped (page unavailable)")
        print(f"  {'-'*45}")
        print(f"\n  URL-only phishing probability : {prob:.2f}%")
        print(f"  Prediction                    : {icon}  {label}")
        print(f"  Confidence band               : {conf}")
        print(f"\n  DNS resolved : {'YES' if dns_ok else 'NO'}")
        print(f"  Page loaded  : NO  ->  fusion model NOT used")
        print(_BAR)

    else:
        print(f"\n        HTML features    : ", end="")
        X_html = extract_html_features(fetch["html"], url)
        print(f"{X_html.shape[1]} dims  [live HTML]")

        print(f"        Visual embedding : ", end="")
        X_visual = extract_visual_embedding(fetch["screenshot_path"])
        print(f"{X_visual.shape[1]} dims  [live screenshot]")

        print(f"\n  [4/4] Running Fusion MLP ({URL_FEATURE_DIM} + 12 + 1280 = 1314 dims)...")
        result = run_fusion(X_url, X_html, X_visual, bundle)
        mode_used = "Fusion MLP"
        prob = result["phishing_prob"]
        pred = result["prediction"]
        label = result["label"]
        icon = "[PHISH]" if pred == 1 else "[SAFE]"

        if prob >= 80:   conf = "HIGH CONFIDENCE"
        elif prob >= 55: conf = "MEDIUM CONFIDENCE"
        elif prob <= 20: conf = "HIGH CONFIDENCE (likely safe)"
        elif prob <= 45: conf = "MEDIUM CONFIDENCE (likely safe)"
        else:            conf = "LOW CONFIDENCE — borderline"

        print(f"\n{_BAR}")
        print(f"  VERDICT  [Mode: {mode_used}]")
        print(_BAR)
        print(f"  {'Modality':<20} {'Weight':>8}  {'Dims':>6}  {'Status'}")
        print(f"  {'-'*55}")
        print(f"  {'URL':<20} {weights['url']:>8}  {X_url.shape[1]:>6}  live canonical features")
        print(f"  {'HTML':<20} {weights['html']:>8}  {X_html.shape[1]:>6}  live HTML")
        print(f"  {'Visual':<20} {weights['visual']:>8}  {X_visual.shape[1]:>6}  live screenshot")
        print(f"  {'-'*55}")
        print(f"\n  Fusion phishing probability : {prob:.2f}%")
        print(f"  Prediction                  : {icon}  {label}")
        print(f"  Confidence band             : {conf}")
        print(f"  Total Fused Dimensions      : {result['fused_dim']}")
        print(f"\n  DNS resolved : YES")
        print(f"  Page loaded  : YES  ->  full fusion model used")
        print(_BAR)

    if fetch.get("screenshot_path"):
        try: os.unlink(fetch["screenshot_path"])
        except Exception: pass


if __name__ == "__main__":
    main()
