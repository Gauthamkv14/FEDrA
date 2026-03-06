"""
scripts/test_fusion.py
======================
FEDrA Step 7 — Fusion Model Inference for a Single URL

This script is COMPLETELY SEPARATE from test_single_url.py.
It uses ONLY the fusion model (models/fusion_model.pkl).

What it does:
  1. Extract URL features               (same schema as url_features.csv)
  2. Fetch page & extract HTML features (same schema as html_features.csv)
  3. Extract visual embedding           (same schema as visual_embeddings.npy)
  4. Apply per-modality scalers + weights (URL=0.4, HTML=0.3, Visual=0.3)
  5. Concatenate and run through the fusion MLP
  6. Print a clean fusion-specific report

Usage:
    /opt/anaconda3/envs/fedra/bin/python scripts/test_fusion.py --url "https://www.google.com"
    /opt/anaconda3/envs/fedra/bin/python scripts/test_fusion.py --url "https://suspicious-login.xyz" --timeout 15

NOTE: All models are loaded with joblib.load() as per project convention.
"""

import os
import re
import sys
import math
import time
import socket
import tempfile
import argparse
import warnings
import urllib.parse

warnings.filterwarnings("ignore")

import numpy as np
import joblib                            # always joblib.load() for .pkl files

from bs4 import BeautifulSoup
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR        = os.path.join(BASE_DIR, "models")
FUSION_MODEL_PATH = os.path.join(MODELS_DIR, "fusion_model.pkl")
URL_MODEL_PATH    = os.path.join(MODELS_DIR, "url_baseline.pkl")   # fallback

_BAR = "=" * 65

# ── Phishing heuristic lists (same as training pipeline) ──────────────────────
_SUSPICIOUS_TLDS = {
    "xyz", "tk", "ml", "ga", "cf", "gq", "top", "click",
    "work", "loan", "men", "date", "racing", "party", "trade",
    "kim", "country", "stream", "download", "gdn", "bid",
    "accountant", "faith", "review", "science", "win",
}
_FREE_HOSTING = {
    "000webhostapp.com", "weebly.com", "wixsite.com", "wordpress.com",
    "blogspot.com", "netlify.app", "github.io", "glitch.me",
    "firebaseapp.com", "web.app", "surge.sh", "pages.dev",
}
_URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "buff.ly", "is.gd", "short.io", "rebrand.ly",
}
_BRAND_KEYWORDS = [
    "paypal", "amazon", "apple", "google", "microsoft", "facebook",
    "instagram", "netflix", "dropbox", "linkedin", "twitter", "ebay",
    "wellsfargo", "chase", "citibank", "bankofamerica", "irs",
    "dhl", "fedex", "usps", "whatsapp", "telegram",
]
SUSPICIOUS_QUERY_WORDS = {"login", "redirect", "verify", "secure"}

_ERR_UNRESOLVABLE = ("ERR_NAME_NOT_RESOLVED", "ERR_NAME_CHANGED")
_ERR_SSL          = ("ERR_SSL_PROTOCOL_ERROR", "ERR_CERT_",
                     "ERR_SSL_VERSION_OR_CIPHER_MISMATCH", "SSL_ERROR")
_ERR_REFUSED      = ("ERR_CONNECTION_REFUSED", "ERR_EMPTY_RESPONSE",
                     "ERR_TUNNEL_CONNECTION_FAILED", "ERR_SOCKET_NOT_CONNECTED")
_ERR_TIMEOUT      = ("ERR_TIMED_OUT", "ERR_CONNECTION_TIMED_OUT", "Timeout")


# ══════════════════════════════════════════════════════════════════════════════
#  DNS CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_dns(url: str) -> bool:
    try:
        parsed   = urllib.parse.urlparse(url if "://" in url else "http://" + url)
        hostname = parsed.hostname or ""
        if not hostname:
            return False
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
    """
    Returns dict: {success, html, screenshot_path, error_type, error_msg}
    Never raises — always returns a dict.
    """
    out = {"success": False, "html": None, "screenshot_path": None,
           "error_type": "none", "error_msg": ""}

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
        out["error_msg"]  = str(e)
        out["error_type"] = _classify_error(str(e))
    except Exception as e:
        out["error_msg"]  = str(e)
        out["error_type"] = "other"
    finally:
        if driver:
            try: driver.quit()
            except Exception: pass

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  URL FEATURE EXTRACTION  (mirrors extract_url_features.py schema)
# ══════════════════════════════════════════════════════════════════════════════

def _shannon_entropy(s: str) -> float:
    if not s: return 0.0
    probs = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def _tld_extract(url: str) -> tuple:
    parsed   = urllib.parse.urlparse(url if "://" in url else "http://" + url)
    hostname = parsed.hostname or ""
    parts    = hostname.split(".")
    multi_tld = {
        "co.uk","co.in","co.jp","co.nz","co.za","com.au","com.br",
        "com.cn","com.mx","net.au","org.uk","gov.uk",
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


def extract_url_features(url: str, fetch_error_type: str = "none") -> np.ndarray:
    """26 features — same schema as url_features.csv training data."""
    parsed_url = url if "://" in url else "http://" + url
    parsed     = urllib.parse.urlparse(parsed_url)
    subdomain, domain, suffix = _tld_extract(parsed_url)
    hostname   = parsed.hostname or ""

    # Core 10 features
    url_len          = len(url)
    is_ip            = 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) else 0
    num_subdomains   = len([s for s in subdomain.split(".") if s]) if subdomain else 0
    is_https         = 1 if url.lower().startswith("https") else 0
    num_at           = url.count("@")
    num_dash         = url.count("-")
    num_double_slash = max(0, url.count("//") - 1) if "://" in url else url.count("//")
    domain_token     = f"{domain}.{suffix}"
    domain_entropy   = _shannon_entropy(domain_token)
    query_params     = urllib.parse.parse_qsl(parsed.query)
    num_params       = len(query_params)
    has_susp_params  = int(any(
        sw in k.lower() for k, _ in query_params for sw in SUSPICIOUS_QUERY_WORDS
    ))

    core = np.array([
        url_len, num_subdomains, is_ip, is_https,
        num_at, num_dash, num_double_slash,
        domain_entropy, num_params, has_susp_params,
    ], dtype=float)

    # Extended 12 features
    brand_in_domain    = int(any(b in domain.lower() and domain.lower() != b for b in _BRAND_KEYWORDS))
    brand_in_subdomain = int(any(b in subdomain.lower() for b in _BRAND_KEYWORDS))
    typo_patterns      = [("0","o"),("1","l"),("3","e"),("4","a"),("5","s")]
    has_typosquat      = int(any(
        any(b.replace(o, r) == domain.lower() for o, r in typo_patterns)
        for b in _BRAND_KEYWORDS
    ))
    has_punycode          = int("xn--" in hostname.lower())
    excessive_subdomains  = int(num_subdomains > 3)
    suspicious_tld        = int(suffix.lower() in _SUSPICIOUS_TLDS)
    full_host             = hostname.lower()
    free_hosting          = int(any(fh in full_host for fh in _FREE_HOSTING))
    is_shortener          = int(any(s in full_host for s in _URL_SHORTENERS))
    excessive_hyphens     = int(domain.count("-") >= 3)
    port                  = parsed.port
    has_nonstandard_port  = int(port is not None and port not in (80, 443, 8080))
    high_entropy          = int(domain_entropy > 3.8)
    long_url              = int(url_len > 100)

    extended = np.array([
        brand_in_domain, brand_in_subdomain, has_typosquat,
        has_punycode, excessive_subdomains, suspicious_tld,
        free_hosting, is_shortener, excessive_hyphens,
        has_nonstandard_port, high_entropy, long_url,
    ], dtype=float)

    # Fetch-failure signals (4 features)
    fail = np.array([
        float(fetch_error_type == "unresolvable"),
        float(fetch_error_type == "ssl"),
        float(fetch_error_type == "refused"),
        float(fetch_error_type not in ("none",)),
    ], dtype=float)

    return np.concatenate([core, extended, fail]).reshape(1, -1)


# ══════════════════════════════════════════════════════════════════════════════
#  HTML FEATURE EXTRACTION  (mirrors html_features.csv schema — 12 features)
# ══════════════════════════════════════════════════════════════════════════════

def _get_domain(url: str) -> str:
    if not url or not isinstance(url, str): return ""
    if not url.startswith("http") and not url.startswith("//"): return ""
    _, domain, suffix = _tld_extract(url)
    return f"{domain}.{suffix}" if domain else ""


def extract_html_features(html_content: str, page_url: str) -> np.ndarray:
    """12-feature vector — same schema as html_features.csv."""
    page_domain = _get_domain(page_url)
    soup        = BeautifulSoup(html_content, "html.parser")
    content     = html_content

    forms  = soup.find_all("form")
    inputs = soup.find_all("input")
    iframes = soup.find_all("iframe")
    num_forms, num_inputs, num_iframes = len(forms), len(inputs), len(iframes)

    ext_domains, num_ext_links, num_ext_scripts = set(), 0, 0
    for a in soup.find_all("a", href=True):
        ld = _get_domain(a["href"])
        if ld and ld != page_domain:
            num_ext_links += 1; ext_domains.add(ld)
    for sc in soup.find_all("script", src=True):
        sd = _get_domain(sc["src"])
        if sd and sd != page_domain:
            num_ext_scripts += 1; ext_domains.add(sd)

    num_unique_ext_domains = len(ext_domains)
    has_password_field     = 1 if soup.find("input", type=lambda t: t and t.lower() == "password") else 0
    has_meta_redirect      = 1 if any(
        m.get("http-equiv", "").lower() == "refresh" for m in soup.find_all("meta")
    ) else 0
    script_text          = "".join(s.get_text() for s in soup.find_all("script") if s.string)
    script_content_ratio = len(script_text) / max(1, len(content))
    favicon_mismatch     = 0
    for fav in soup.find_all("link", rel=lambda r: r and "icon" in r.lower()):
        fd = _get_domain(fav.get("href", ""))
        if fd and fd != page_domain:
            favicon_mismatch = 1; break
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


def html_zeros() -> np.ndarray:
    """Return a zero-filled 12-feature HTML vector when page is unavailable."""
    return np.zeros((1, 12), dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
#  VISUAL EMBEDDING  (mirrors extract_visual_embeddings.py — 1280-dim MobileNetV2)
# ══════════════════════════════════════════════════════════════════════════════

def extract_visual_embedding(screenshot_path: str) -> np.ndarray:
    """1280-dim MobileNetV2 embedding — same as training pipeline."""
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    net     = models.mobilenet_v2(weights=weights).features
    net.eval(); net.to(device)

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


def visual_zeros() -> np.ndarray:
    """Return a zero-filled 1280-feature visual vector when screenshot unavailable."""
    return np.zeros((1, 1280), dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
#  URL-ONLY FALLBACK INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_url_only(X_url: np.ndarray) -> dict:
    """
    Load url_baseline.pkl and return a prediction dict.
    Used as fallback when the page is unavailable (no zeros fed to fusion).
    NOTE: always joblib.load() for .pkl files.
    """
    if not os.path.isfile(URL_MODEL_PATH):
        return {"error": f"URL model not found: {URL_MODEL_PATH}"}
    bundle = joblib.load(URL_MODEL_PATH)          # NOTE: joblib.load()
    model  = bundle["model"]
    scaler = bundle["scaler"]
    exp    = scaler.n_features_in_
    cur    = X_url.shape[1]
    if cur < exp:
        X_url = np.hstack([X_url, np.zeros((1, exp - cur))])
    elif cur > exp:
        X_url = X_url[:, :exp]
    X_s  = scaler.transform(X_url)
    pred = int(model.predict(X_s)[0])
    prob = float(model.predict_proba(X_s)[0][1]) * 100
    return {
        "prediction":    pred,
        "label":         "PHISHING" if pred == 1 else "LEGITIMATE",
        "phishing_prob": round(prob, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FUSION INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_fusion(X_url: np.ndarray, X_html: np.ndarray, X_visual: np.ndarray,
               bundle: dict) -> dict:
    """
    Apply per-modality scalers + weights then pass through the fusion MLP.

    bundle keys: model, scalers (dict: url/html/visual), weights (dict)
    Returns: {phishing_prob, prediction, label}
    """
    scalers = bundle["scalers"]
    weights = bundle["weights"]
    model   = bundle["model"]

    def _scale_weight(name, X):
        sc = scalers[name]
        exp = sc.n_features_in_
        cur = X.shape[1]
        if cur < exp:
            X = np.hstack([X, np.zeros((1, exp - cur))])
        elif cur > exp:
            X = X[:, :exp]
        return sc.transform(X) * weights[name]

    X_url_sw    = _scale_weight("url",    X_url)
    X_html_sw   = _scale_weight("html",   X_html)
    X_visual_sw = _scale_weight("visual", X_visual)

    X_fused = np.hstack([X_url_sw, X_html_sw, X_visual_sw])

    pred = int(model.predict(X_fused)[0])
    prob = float(model.predict_proba(X_fused)[0][1]) * 100

    return {
        "prediction":    pred,
        "label":         "PHISHING" if pred == 1 else "LEGITIMATE",
        "phishing_prob": round(prob, 2),
        "fused_dim":     X_fused.shape[1],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FEDrA Step 7 — Fusion Model inference for a single URL"
    )
    parser.add_argument("--url",     required=True,        help="URL to classify")
    parser.add_argument("--timeout", type=int, default=20, help="Page load timeout in seconds")
    args   = parser.parse_args()
    url    = args.url

    print(f"\n{_BAR}")
    print("  FEDrA — Fusion Model (Step 7) — Single URL Inference")
    print(_BAR)
    print(f"  URL     : {url}")
    print(f"  Model   : {FUSION_MODEL_PATH}")
    print(_BAR)

    # ── 0. Load fusion model ──────────────────────────────────────────────────
    if not os.path.isfile(FUSION_MODEL_PATH):
        print(f"\n❌  Fusion model not found at {FUSION_MODEL_PATH}")
        print("   Run:  python scripts/train_fusion.py  first.")
        sys.exit(1)

    print("\n  [0/4] Loading fusion model bundle (joblib)...")
    bundle  = joblib.load(FUSION_MODEL_PATH)       # NOTE: always joblib.load()
    weights = bundle["weights"]
    print(f"        Weights → URL={weights['url']}  HTML={weights['html']}  "
          f"Visual={weights['visual']}")
    print(f"        MLP architecture: {bundle['model'].hidden_layer_sizes}")

    # ── 1. DNS check ──────────────────────────────────────────────────────────
    print("\n  [1/4] DNS resolution check...")
    dns_ok   = check_dns(url)
    dns_icon = "✅ resolves" if dns_ok else "❌ UNRESOLVABLE (phishing signal)"
    print(f"        Domain → {dns_icon}")

    # ── 2. Fetch page ─────────────────────────────────────────────────────────
    print(f"\n  [2/4] Fetching page (headless Chrome, timeout={args.timeout}s)...")
    fetch = fetch_page(url, timeout=args.timeout)

    page_available = False
    if fetch["success"]:
        html_kb = len(fetch["html"]) / 1024
        print(f"        ✅ Loaded — HTML: {html_kb:.1f} KB")
        page_available = True
    else:
        etype = fetch["error_type"]
        print(f"        ⚠️  Fetch failed [{etype}]: {fetch['error_msg'][:80]}")

    # ── 3. Feature extraction ─────────────────────────────────────────────────
    # URL features are always extracted (no network needed)
    print("\n  [3/4] Feature extraction...")
    X_url = extract_url_features(url, fetch["error_type"])
    print(f"        URL features : {X_url.shape[1]} dims  [always available]")

    # ── 4. Decide: Fusion or URL-only fallback ────────────────────────────────
    if not page_available:
        # ── FALLBACK PATH: page unavailable → URL-only model ──────────────────
        print("\n  [4/4] Page unavailable → switching to URL-only detection")
        print(f"        Loading URL baseline model: {URL_MODEL_PATH}")
        result    = run_url_only(X_url)
        mode_used = "URL-only (fallback)"

        if "error" in result:
            print(f"        ❌ URL model error: {result['error']}")
            sys.exit(1)

        prob  = result["phishing_prob"]
        pred  = result["prediction"]
        label = result["label"]
        icon  = "🚨" if pred == 1 else "✅"
        print(f"        URL-only phishing probability: {prob:.2f}%")

        # Confidence band
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
        print(f"  {'URL':<20} {X_url.shape[1]:>6}  live features")
        print(f"  {'HTML':<20} {'—':>6}  skipped (page unavailable)")
        print(f"  {'Visual':<20} {'—':>6}  skipped (page unavailable)")
        print(f"  {'-'*45}")
        print(f"\n  URL-only phishing probability : {prob:.2f}%")
        print(f"  Prediction                    : {icon}  {label}")
        print(f"  Confidence band               : {conf}")
        print(f"\n  DNS resolved : {'YES' if dns_ok else 'NO'}")
        print(f"  Page loaded  : NO  →  fusion model NOT used (no zero vectors)")
        print(_BAR)

    else:
        # ── FUSION PATH: page loaded → extract all 3 modalities → run fusion ──
        print(f"\n        HTML features    : ", end="")
        X_html = extract_html_features(fetch["html"], url)
        print(f"{X_html.shape[1]} dims  [live HTML]")

        print(f"        Visual embedding : ", end="")
        X_visual = extract_visual_embedding(fetch["screenshot_path"])
        print(f"{X_visual.shape[1]} dims  [live screenshot]")

        print("\n  [4/4] Running Fusion MLP (URL=0.4, HTML=0.3, Visual=0.3)...")
        result    = run_fusion(X_url, X_html, X_visual, bundle)
        mode_used = "Fusion MLP"
        prob  = result["phishing_prob"]
        pred  = result["prediction"]
        label = result["label"]
        icon  = "🚨" if pred == 1 else "✅"
        print(f"        Fused vector dim : {result['fused_dim']}")
        print(f"        Fusion phishing probability: {prob:.2f}%")

        # Confidence band
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
        print(f"  {'URL':<20} {weights['url']:>8}  {X_url.shape[1]:>6}  live features")
        print(f"  {'HTML':<20} {weights['html']:>8}  {X_html.shape[1]:>6}  live HTML")
        print(f"  {'Visual':<20} {weights['visual']:>8}  {X_visual.shape[1]:>6}  live screenshot")
        print(f"  {'-'*55}")
        print(f"\n  Fusion phishing probability : {prob:.2f}%")
        print(f"  Prediction                  : {icon}  {label}")
        print(f"  Confidence band             : {conf}")
        print(f"\n  DNS resolved : YES")
        print(f"  Page loaded  : YES  →  full fusion model used")
        print(_BAR)

    # Cleanup temp screenshot
    if fetch.get("screenshot_path"):
        try:
            os.unlink(fetch["screenshot_path"])
        except Exception:
            pass


if __name__ == "__main__":
    main()
