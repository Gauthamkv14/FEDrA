"""
scripts/test_single_url.py
==========================
FEDrA — Phishing detector for a single URL using baseline models.

Uses canonical 22-dimensional URL feature schema (v1).
Local only — no external APIs.

Usage:
    python scripts/test_single_url.py --url "https://example.com"
    python scripts/test_single_url.py --url "https://example.com" --timeout 15
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
MODEL_FILES = {
    "url": os.path.join(MODELS_DIR, "url_baseline.pkl"),
    "html": os.path.join(MODELS_DIR, "html_baseline.pkl"),
    "image": os.path.join(MODELS_DIR, "image_baseline.pkl"),
}

_ERR_UNRESOLVABLE = ("ERR_NAME_NOT_RESOLVED", "ERR_NAME_CHANGED")
_ERR_SSL = ("ERR_SSL_PROTOCOL_ERROR", "ERR_CERT_", "ERR_SSL_VERSION_OR_CIPHER_MISMATCH", "SSL_ERROR")
_ERR_REFUSED = ("ERR_CONNECTION_REFUSED", "ERR_EMPTY_RESPONSE", "ERR_TUNNEL_CONNECTION_FAILED", "ERR_SOCKET_NOT_CONNECTED")
_ERR_TIMEOUT = ("ERR_TIMED_OUT", "ERR_CONNECTION_TIMED_OUT", "Timeout")


# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN RESOLUTION CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_domain_resolves(url: str) -> bool:
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
#  FETCH ERROR CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

class FetchResult:
    def __init__(self):
        self.html: str | None = None
        self.screenshot: str | None = None
        self.success: bool = False
        self.error_type: str = "none"
        self.error_msg: str = ""

    @property
    def available(self) -> bool:
        return self.success and self.html is not None


def _classify_error(msg: str) -> str:
    m = msg.upper()
    if any(e.upper() in m for e in _ERR_UNRESOLVABLE): return "unresolvable"
    if any(e.upper() in m for e in _ERR_SSL):          return "ssl"
    if any(e.upper() in m for e in _ERR_REFUSED):      return "refused"
    if any(e.upper() in m for e in _ERR_TIMEOUT):      return "timeout"
    return "other"


def fetch_page(url: str, timeout: int = 20) -> FetchResult:
    result = FetchResult()
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

        result.html = driver.page_source
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        driver.save_screenshot(tmp.name)
        result.screenshot = tmp.name
        result.success = True

    except WebDriverException as e:
        result.error_msg = str(e)
        result.error_type = _classify_error(str(e))
    except Exception as e:
        result.error_msg = str(e)
        result.error_type = "other"
    finally:
        if driver:
            try: driver.quit()
            except Exception: pass

    return result


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
    content = html_content

    forms, inputs, iframes = soup.find_all("form"), soup.find_all("input"), soup.find_all("iframe")
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
#  PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _predict_model(modality: str, X: np.ndarray) -> dict:
    path = MODEL_FILES[modality]
    if not os.path.isfile(path):
        return {"error": f"Model not found: {path}"}
    try:
        bundle = joblib.load(path)
        model = bundle["model"]
        scaler = bundle["scaler"]
        assert X.shape[1] == scaler.n_features_in_, (
            f"{modality} dimension mismatch: got {X.shape[1]}, expected {scaler.n_features_in_}"
        )
        X_s = scaler.transform(X)
        pred = int(model.predict(X_s)[0])
        prob = float(model.predict_proba(X_s)[0][1])
        return {
            "prediction": pred,
            "label": "PHISHING" if pred == 1 else "LEGITIMATE",
            "phishing_prob": round(prob * 100, 2),
            "input_dim": X.shape[1],
        }
    except Exception as e:
        return {"error": str(e)}


def determine_verdict(
    url_prob: float,
    domain_resolved: bool,
    html_result: dict | None,
    vis_result: dict | None,
    fetch_result: FetchResult,
) -> tuple[str, str]:
    u = url_prob / 100.0

    if not domain_resolved:
        if u > 0.5:
            return "PHISHING", "HIGH CONFIDENCE — URL risk + domain unresolvable"
        else:
            return "SUSPICIOUS", "dead/parked domain, low URL risk — flag for review"

    if not fetch_result.success:
        if u > 0.5:
            return "PHISHING", "MEDIUM CONFIDENCE — URL risk (HTML/Visual unavailable)"
        else:
            return "SUSPICIOUS", "page unreachable, URL appears low risk — needs manual check"

    votes = []
    for r in (html_result, vis_result):
        if r and "prediction" in r:
            votes.append(r["prediction"])
    votes.append(1 if u > 0.5 else 0)

    majority = int(sum(votes) > len(votes) / 2)
    if majority == 1:
        return "PHISHING", "majority vote (URL + HTML + Visual)"
    else:
        return "LEGITIMATE", "majority vote (URL + HTML + Visual)"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

_BAR = "=" * 65

def main():
    parser = argparse.ArgumentParser(
        description="FEDrA — Phishing detector for a single URL using baseline models (Schema v1)"
    )
    parser.add_argument("--url", required=True, help="URL to classify")
    parser.add_argument("--timeout", type=int, default=20, help="Page load timeout (s)")
    args = parser.parse_args()
    url = args.url

    print(f"\n{_BAR}")
    print("  FEDrA — Single URL Phishing Detector (Canonical Schema v1)")
    print(_BAR)
    print(f"  URL : {url}\n")

    # ── 1. DNS check ──
    print("  [1/4] Checking DNS resolution...")
    domain_resolved = check_domain_resolves(url)
    dns_icon = "OK (resolves)" if domain_resolved else "UNRESOLVABLE"
    print(f"        Domain -> {dns_icon}")

    # ── 2. URL model (Canonical 22 features) ──
    print("\n  [2/4] Extracting canonical URL features & running URL model...")
    X_url = extract_canonical_url_features_vector(url)
    url_result = _predict_model("url", X_url)
    print(f"        Canonical URL features: {X_url.shape[1]} dims -> phishing prob = {url_result.get('phishing_prob', '?')}%")

    # ── 3. Fetch page ──
    print(f"\n  [3/4] Fetching page (headless Chrome, timeout={args.timeout}s)...")
    fetch = fetch_page(url, timeout=args.timeout)

    if fetch.success:
        html_kb = len(fetch.html) / 1024
        print(f"        [OK] Loaded — HTML: {html_kb:.1f} KB | screenshot: {fetch.screenshot}")
    else:
        etype = fetch.error_type
        print(f"        [WARN] Fetch failed [{etype}]: {fetch.error_msg[:80]}")
        print("        -> Continuing in URL-only mode")

    # ── 4. HTML + Visual ──
    html_result = None
    vis_result = None

    if fetch.available:
        print("\n  [4/4] Extracting HTML and Visual features...")
        X_html = extract_html_features(fetch.html, url)
        html_result = _predict_model("html", X_html)
        print(f"        HTML features: {X_html.shape[1]} dims")

        X_vis = extract_visual_embedding(fetch.screenshot)
        vis_result = _predict_model("image", X_vis)
        print(f"        Visual embedding: {X_vis.shape[1]} dims")
    else:
        print("\n  [4/4] Skipping HTML + Visual (page unavailable)")

    # ── 5. Verdict ──
    url_prob = url_result.get("phishing_prob", 0.0)
    verdict, confidence = determine_verdict(
        url_prob, domain_resolved, html_result, vis_result, fetch
    )

    # ── 6. Print report ──
    print("\nPrediction probability from each baseline model:")
    def _get_prob_str(res):
        if res is None or "error" in res: return "N/A"
        return f"{res['phishing_prob']:.2f}%"

    print(f"   - URL model ({URL_FEATURE_DIM} dims): {_get_prob_str(url_result)}")
    print(f"   - HTML model (12 dims): {_get_prob_str(html_result)}")
    print(f"   - Visual model (1280 dims): {_get_prob_str(vis_result)}")

    print(f"\nFinal label : {verdict}")
    print(f"Confidence  : {confidence}")
    print(f"{_BAR}\n")

    if fetch.screenshot:
        try: os.unlink(fetch.screenshot)
        except Exception: pass


if __name__ == "__main__":
    main()
