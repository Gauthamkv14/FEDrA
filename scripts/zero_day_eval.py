"""
scripts/zero_day_eval.py
========================
FEDrA Step 8 — Zero-Day Evaluation

Fetches 20-30 live phishing URLs from OpenPhish public feed,
runs the full fusion inference pipeline on each, and reports
detection accuracy, latency, and false negatives.

Pipeline per URL (mirrors test_fusion.py):
  1. DNS check
  2. Fetch page (headless Chrome, 10s timeout)
  3a. Page available  → Fusion MLP  (URL + HTML + Visual, weighted concat)
  3b. Page unavailable → URL-only fallback (url_baseline.pkl)
  4. Record result

Output:
  - notebooks/zero_day_results.csv
  - Console report (detection rate, avg latency, false negatives)

NOTE: All models loaded via joblib.load() as per project convention.

Usage:
    /opt/anaconda3/envs/fedra/bin/python scripts/zero_day_eval.py
    /opt/anaconda3/envs/fedra/bin/python scripts/zero_day_eval.py --n 25 --timeout 10
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

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR        = os.path.join(BASE_DIR, "models")
NOTEBOOKS_DIR     = os.path.join(BASE_DIR, "notebooks")
FUSION_MODEL_PATH = os.path.join(MODELS_DIR, "fusion_model.pkl")
URL_MODEL_PATH    = os.path.join(MODELS_DIR, "url_baseline.pkl")
OUTPUT_CSV        = os.path.join(NOTEBOOKS_DIR, "zero_day_results.csv")

OPENPHISH_FEED    = "https://openphish.com/feed.txt"

os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

# Phishing heuristic lists (same as training pipeline)
_SUSPICIOUS_TLDS = {
    "xyz","tk","ml","ga","cf","gq","top","click","work","loan","men",
    "date","racing","party","trade","kim","country","stream","download",
    "gdn","bid","accountant","faith","review","science","win",
}
_FREE_HOSTING = {
    "000webhostapp.com","weebly.com","wixsite.com","wordpress.com",
    "blogspot.com","netlify.app","github.io","glitch.me",
    "firebaseapp.com","web.app","surge.sh","pages.dev",
}
_URL_SHORTENERS = {
    "bit.ly","tinyurl.com","goo.gl","t.co","ow.ly",
    "buff.ly","is.gd","short.io","rebrand.ly",
}
_BRAND_KEYWORDS = [
    "paypal","amazon","apple","google","microsoft","facebook",
    "instagram","netflix","dropbox","linkedin","twitter","ebay",
    "wellsfargo","chase","citibank","bankofamerica","irs",
    "dhl","fedex","usps","whatsapp","telegram",
]
SUSPICIOUS_QUERY_WORDS = {"login","redirect","verify","secure"}

_ERR_UNRESOLVABLE = ("ERR_NAME_NOT_RESOLVED","ERR_NAME_CHANGED")
_ERR_SSL          = ("ERR_SSL_PROTOCOL_ERROR","ERR_CERT_",
                     "ERR_SSL_VERSION_OR_CIPHER_MISMATCH","SSL_ERROR")
_ERR_REFUSED      = ("ERR_CONNECTION_REFUSED","ERR_EMPTY_RESPONSE",
                     "ERR_TUNNEL_CONNECTION_FAILED","ERR_SOCKET_NOT_CONNECTED")
_ERR_TIMEOUT      = ("ERR_TIMED_OUT","ERR_CONNECTION_TIMED_OUT","Timeout")


# ══════════════════════════════════════════════════════════════════════════════
#  FEED FETCHER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_openphish_feed(n: int) -> list:
    """Fetch n URLs from the OpenPhish public feed (no API key needed)."""
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
        print(f"  ⚠️  Could not fetch feed: {e}")
        print("  → Using a small hardcoded set of known phishing patterns for demo.")
        # Fallback demo URLs (known phishing patterns — safe to test against)
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
        parsed   = urllib.parse.urlparse(url if "://" in url else "http://" + url)
        hostname = parsed.hostname or ""
        if not hostname: return False
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
    """Fetch page with headless Chrome. Never raises."""
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
        time.sleep(1)
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
#  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _shannon_entropy(s: str) -> float:
    if not s: return 0.0
    probs = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def _tld_extract(url: str) -> tuple:
    parsed   = urllib.parse.urlparse(url if "://" in url else "http://" + url)
    hostname = parsed.hostname or ""
    parts    = hostname.split(".")
    multi = {"co.uk","co.in","co.jp","co.nz","co.za","com.au","com.br",
             "com.cn","com.mx","net.au","org.uk","gov.uk"}
    if len(parts) >= 3 and ".".join(parts[-2:]) in multi:
        suffix, domain = ".".join(parts[-2:]), parts[-3] if len(parts) >= 3 else ""
        subdomain = ".".join(parts[:-3]) if len(parts) > 3 else ""
    elif len(parts) >= 2:
        suffix, domain, subdomain = parts[-1], parts[-2], ".".join(parts[:-2])
    else:
        suffix, domain, subdomain = "", hostname, ""
    return subdomain, domain, suffix


def extract_url_features(url: str, fetch_error_type: str = "none") -> np.ndarray:
    """26-dim URL feature vector — same schema as training."""
    parsed_url = url if "://" in url else "http://" + url
    parsed     = urllib.parse.urlparse(parsed_url)
    subdomain, domain, suffix = _tld_extract(parsed_url)
    hostname   = parsed.hostname or ""

    url_len          = len(url)
    is_ip            = 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) else 0
    num_subdomains   = len([s for s in subdomain.split(".") if s]) if subdomain else 0
    is_https         = 1 if url.lower().startswith("https") else 0
    num_at           = url.count("@")
    num_dash         = url.count("-")
    num_double_slash = max(0, url.count("//") - 1) if "://" in url else url.count("//")
    domain_entropy   = _shannon_entropy(f"{domain}.{suffix}")
    query_params     = urllib.parse.parse_qsl(parsed.query)
    num_params       = len(query_params)
    has_susp_params  = int(any(
        sw in k.lower() for k, _ in query_params for sw in SUSPICIOUS_QUERY_WORDS
    ))
    core = np.array([url_len, num_subdomains, is_ip, is_https, num_at, num_dash,
                     num_double_slash, domain_entropy, num_params, has_susp_params], dtype=float)

    brand_in_domain    = int(any(b in domain.lower() and domain.lower() != b for b in _BRAND_KEYWORDS))
    brand_in_subdomain = int(any(b in subdomain.lower() for b in _BRAND_KEYWORDS))
    tp = [("0","o"),("1","l"),("3","e"),("4","a"),("5","s")]
    has_typosquat     = int(any(any(b.replace(o,r)==domain.lower() for o,r in tp) for b in _BRAND_KEYWORDS))
    full_host         = hostname.lower()
    extended = np.array([
        brand_in_domain, brand_in_subdomain, has_typosquat,
        int("xn--" in hostname.lower()), int(num_subdomains > 3),
        int(suffix.lower() in _SUSPICIOUS_TLDS),
        int(any(fh in full_host for fh in _FREE_HOSTING)),
        int(any(s in full_host for s in _URL_SHORTENERS)),
        int(domain.count("-") >= 3),
        int(parsed.port is not None and parsed.port not in (80, 443, 8080)),
        int(domain_entropy > 3.8), int(url_len > 100),
    ], dtype=float)

    fail = np.array([
        float(fetch_error_type == "unresolvable"),
        float(fetch_error_type == "ssl"),
        float(fetch_error_type == "refused"),
        float(fetch_error_type not in ("none",)),
    ], dtype=float)

    return np.concatenate([core, extended, fail]).reshape(1, -1)


def _get_domain(url: str) -> str:
    if not url or not isinstance(url, str): return ""
    if not url.startswith("http") and not url.startswith("//"): return ""
    _, domain, suffix = _tld_extract(url)
    return f"{domain}.{suffix}" if domain else ""


def extract_html_features(html_content: str, page_url: str) -> np.ndarray:
    """12-dim HTML feature vector — same schema as training."""
    page_domain = _get_domain(page_url)
    soup        = BeautifulSoup(html_content, "html.parser")
    forms  = soup.find_all("form")
    inputs = soup.find_all("input")
    iframes = soup.find_all("iframe")
    ext_domains, num_ext_links, num_ext_scripts = set(), 0, 0
    for a in soup.find_all("a", href=True):
        ld = _get_domain(a["href"])
        if ld and ld != page_domain: num_ext_links += 1; ext_domains.add(ld)
    for sc in soup.find_all("script", src=True):
        sd = _get_domain(sc["src"])
        if sd and sd != page_domain: num_ext_scripts += 1; ext_domains.add(sd)
    has_pass   = 1 if soup.find("input", type=lambda t: t and t.lower() == "password") else 0
    has_meta_r = 1 if any(m.get("http-equiv","").lower()=="refresh" for m in soup.find_all("meta")) else 0
    script_text = "".join(s.get_text() for s in soup.find_all("script") if s.string)
    script_ratio = len(script_text) / max(1, len(html_content))
    fav_mismatch = 0
    for fav in soup.find_all("link", rel=lambda r: r and "icon" in r.lower()):
        if _get_domain(fav.get("href","")) not in ("", page_domain):
            fav_mismatch = 1; break
    has_auto   = int(bool(forms) and "submit()" in script_text.lower())
    submits    = len(soup.find_all(["input","button"], type=lambda t: t and t.lower()=="submit"))
    submits   += len(soup.find_all("input", type=lambda t: t and t.lower()=="image"))
    in_ratio   = len(inputs) / max(1, submits) if inputs else 0.0
    return np.array([len(forms), len(inputs), len(iframes), num_ext_links, num_ext_scripts,
                     has_pass, has_meta_r, script_ratio, fav_mismatch, has_auto, in_ratio,
                     len(ext_domains)], dtype=float).reshape(1, -1)


def extract_visual_embedding(screenshot_path: str, net, preprocess, device) -> np.ndarray:
    """1280-dim MobileNetV2 embedding — model passed in to avoid reloading per URL."""
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
    """URL-only prediction using url_baseline.pkl bundle."""
    model, scaler = url_bundle["model"], url_bundle["scaler"]
    exp = scaler.n_features_in_
    cur = X_url.shape[1]
    if cur < exp:   X_url = np.hstack([X_url, np.zeros((1, exp - cur))])
    elif cur > exp: X_url = X_url[:, :exp]
    X_s  = scaler.transform(X_url)
    pred = int(model.predict(X_s)[0])
    prob = float(model.predict_proba(X_s)[0][1]) * 100
    return {"prediction": pred, "label": "PHISHING" if pred == 1 else "LEGITIMATE",
            "phishing_prob": round(prob, 2), "mode": "URL-only (fallback)"}


def run_fusion(X_url, X_html, X_visual, fusion_bundle: dict) -> dict:
    """Full fusion MLP prediction."""
    scalers = fusion_bundle["scalers"]
    weights = fusion_bundle["weights"]
    model   = fusion_bundle["model"]

    def sw(name, X):
        sc = scalers[name]
        exp, cur = sc.n_features_in_, X.shape[1]
        if cur < exp:   X = np.hstack([X, np.zeros((1, exp - cur))])
        elif cur > exp: X = X[:, :exp]
        return sc.transform(X) * weights[name]

    X_fused = np.hstack([sw("url", X_url), sw("html", X_html), sw("visual", X_visual)])
    pred = int(model.predict(X_fused)[0])
    prob = float(model.predict_proba(X_fused)[0][1]) * 100
    return {"prediction": pred, "label": "PHISHING" if pred == 1 else "LEGITIMATE",
            "phishing_prob": round(prob, 2), "mode": "Fusion MLP"}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FEDrA Step 8 — Zero-Day Evaluation on live phishing URLs"
    )
    parser.add_argument("--n",       type=int, default=25,  help="Number of URLs to evaluate")
    parser.add_argument("--timeout", type=int, default=10,  help="Per-URL Chrome timeout (s)")
    args = parser.parse_args()

    BAR = "=" * 70
    print(f"\n{BAR}")
    print("  FEDrA Step 8 — Zero-Day Evaluation")
    print(BAR)

    # ── Load models once (shared across all URL evaluations) ─────────────────
    print("\n[Init] Loading models...")

    if not os.path.isfile(FUSION_MODEL_PATH):
        print(f"  ❌ Fusion model not found: {FUSION_MODEL_PATH}")
        print("     Run scripts/train_fusion.py first.")
        sys.exit(1)
    if not os.path.isfile(URL_MODEL_PATH):
        print(f"  ❌ URL baseline not found: {URL_MODEL_PATH}")
        sys.exit(1)

    fusion_bundle = joblib.load(FUSION_MODEL_PATH)    # NOTE: always joblib.load()
    url_bundle    = joblib.load(URL_MODEL_PATH)        # NOTE: always joblib.load()
    print(f"  [✓] Fusion model loaded  (MLP {fusion_bundle['model'].hidden_layer_sizes})")
    print(f"  [✓] URL baseline loaded  ({type(url_bundle['model']).__name__})")

    # Load MobileNetV2 once — reused for every visual embedding
    print("  [✓] Loading MobileNetV2 for visual embeddings...")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mv2_wts   = models.MobileNet_V2_Weights.IMAGENET1K_V1
    mobilenet = models.mobilenet_v2(weights=mv2_wts).features
    mobilenet.eval(); mobilenet.to(device)
    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"  [✓] MobileNetV2 ready on {device}")

    # ── Fetch phishing URLs ───────────────────────────────────────────────────
    print(f"\n[Step 1] Fetching phishing URLs (n={args.n})...")
    phish_urls = fetch_openphish_feed(args.n)
    print(f"  URLs to evaluate: {len(phish_urls)}")

    # ── Evaluate each URL ─────────────────────────────────────────────────────
    print(f"\n[Step 2] Running inference on each URL (timeout={args.timeout}s/URL)...")
    print(f"  {'#':<4} {'URL (truncated)':<55} {'Mode':<18} {'Prob%':>6}  {'Pred'}")
    print("  " + "-" * 95)

    records = []
    for i, url in enumerate(phish_urls, 1):
        url_display = (url[:52] + "...") if len(url) > 55 else url
        t0 = time.time()

        # DNS check
        dns_ok = check_dns(url)

        # Page fetch
        fetch  = fetch_page(url, timeout=args.timeout)
        page_ok = fetch["success"]

        # Feature extraction + inference
        X_url = extract_url_features(url, fetch["error_type"])
        screenshot_path = fetch.get("screenshot_path")

        if page_ok:
            X_html   = extract_html_features(fetch["html"], url)
            X_visual = extract_visual_embedding(screenshot_path, mobilenet, preprocess, device)
            result   = run_fusion(X_url, X_html, X_visual, fusion_bundle)
        else:
            result = run_url_only(X_url, url_bundle)

        elapsed = round(time.time() - t0, 2)

        icon = "🚨" if result["prediction"] == 1 else "✅"
        mode_short = "Fusion" if result["mode"] == "Fusion MLP" else "URL-only"
        print(f"  {i:<4} {url_display:<55} {mode_short:<18} {result['phishing_prob']:>6.1f}%  {icon} {result['label']}")

        records.append({
            "url":              url,
            "true_label":       "PHISHING",          # all OpenPhish URLs are phishing
            "predicted_label":  result["label"],
            "phishing_prob_%":  result["phishing_prob"],
            "prediction":       result["prediction"],
            "correct":          int(result["prediction"] == 1),
            "mode":             result["mode"],
            "dns_resolved":     dns_ok,
            "page_loaded":      page_ok,
            "latency_s":        elapsed,
        })

        # Cleanup screenshot
        if screenshot_path:
            try: os.unlink(screenshot_path)
            except Exception: pass

    # ── Compute summary stats ─────────────────────────────────────────────────
    df = pd.DataFrame(records)

    total          = len(df)
    correct        = int(df["correct"].sum())
    false_neg      = int((df["prediction"] == 0).sum())   # predicted legitimate on phishing
    detection_rate = round(correct / total * 100, 1) if total > 0 else 0.0
    avg_latency    = round(df["latency_s"].mean(), 2)
    fusion_rows    = df[df["mode"] == "Fusion MLP"]
    url_only_rows  = df[df["mode"] == "URL-only (fallback)"]

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Results saved → {OUTPUT_CSV}")

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n{BAR}")
    print("  ZERO-DAY EVALUATION REPORT")
    print(BAR)
    print(f"  URLs evaluated         : {total}")
    print(f"  Correctly detected     : {correct} / {total}  ({detection_rate}%)")
    print(f"  False negatives        : {false_neg}  (phishing → predicted LEGITIMATE)")
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
            print(f"    ⚠️  {row['url'][:70]}  ({row['phishing_prob_%']:.1f}%)")
    else:
        print(f"\n  ✅ No false negatives — all {total} phishing URLs correctly flagged!")

    print(f"\n  Results CSV  → {OUTPUT_CSV}")
    print(BAR + "\n")


if __name__ == "__main__":
    main()
