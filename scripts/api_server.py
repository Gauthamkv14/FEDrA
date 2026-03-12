"""
scripts/api_server.py
=====================
FEDrA Flask API — phishing detection backend.

POST /analyze  { "url": "https://..." }
→ { "prediction", "phishing_probability", "risk_level", "reasons", "mode", "latency_s" }

Models are loaded ONCE at startup (MobileNetV2 + fusion + url-only).
Run:
    /opt/anaconda3/envs/fedra/bin/python scripts/api_server.py

NOTE: All models loaded via joblib.load() — never pickle.
"""

import os
import re
import math
import time
import socket
import tempfile
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

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR        = os.path.join(BASE_DIR, "models")
FUSION_MODEL_PATH = os.path.join(MODELS_DIR, "fusion_model.pkl")
URL_MODEL_PATH    = os.path.join(MODELS_DIR, "url_baseline.pkl")

# Phishing heuristic lists
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
_ERR_SSL   = ("ERR_SSL_PROTOCOL_ERROR","ERR_CERT_","ERR_SSL_VERSION_OR_CIPHER_MISMATCH","SSL_ERROR")
_ERR_REFUSED = ("ERR_CONNECTION_REFUSED","ERR_EMPTY_RESPONSE","ERR_TUNNEL_CONNECTION_FAILED")
_ERR_TIMEOUT = ("ERR_TIMED_OUT","ERR_CONNECTION_TIMED_OUT","Timeout")


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP — load all models once
# ══════════════════════════════════════════════════════════════════════════════

print("[FEDrA] Loading models at startup...")
_fusion_bundle = joblib.load(FUSION_MODEL_PATH)   # NOTE: joblib.load()
_url_bundle    = joblib.load(URL_MODEL_PATH)       # NOTE: joblib.load()

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_mv2_net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
_mv2_net.eval(); _mv2_net.to(_device)
_preprocess = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print(f"[FEDrA] Models ready on {_device}. Server starting...")


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE HELPERS  (mirrors zero_day_eval.py exactly)
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
        suffix = ".".join(parts[-2:])
        domain = parts[-3] if len(parts) >= 3 else ""
        subdomain = ".".join(parts[:-3]) if len(parts) > 3 else ""
    elif len(parts) >= 2:
        suffix, domain, subdomain = parts[-1], parts[-2], ".".join(parts[:-2])
    else:
        suffix, domain, subdomain = "", hostname, ""
    return subdomain, domain, suffix


def _get_domain(url: str) -> str:
    if not url or not isinstance(url, str): return ""
    if not url.startswith("http") and not url.startswith("//"): return ""
    _, domain, suffix = _tld_extract(url)
    return f"{domain}.{suffix}" if domain else ""


def _classify_error(msg: str) -> str:
    m = msg.upper()
    if any(e.upper() in m for e in _ERR_UNRESOLVABLE): return "unresolvable"
    if any(e.upper() in m for e in _ERR_SSL):          return "ssl"
    if any(e.upper() in m for e in _ERR_REFUSED):      return "refused"
    if any(e.upper() in m for e in _ERR_TIMEOUT):      return "timeout"
    return "other"


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


def fetch_page(url: str, timeout: int = 15) -> dict:
    out = {"success": False, "html": None, "screenshot_path": None,
           "error_type": "none", "error_msg": ""}
    opts = Options()
    for arg in ["--headless","--no-sandbox","--disable-dev-shm-usage",
                "--disable-gpu","--window-size=1280,800","--log-level=3","--silent"]:
        opts.add_argument(arg)
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


def extract_url_features(url: str, fetch_error_type: str = "none") -> tuple:
    """Returns (feature_vector, metadata_dict) — metadata used for reasons."""
    parsed_url = url if "://" in url else "http://" + url
    parsed     = urllib.parse.urlparse(parsed_url)
    subdomain, domain, suffix = _tld_extract(parsed_url)
    hostname   = parsed.hostname or ""
    full_host  = hostname.lower()

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

    brand_in_domain    = int(any(b in domain.lower() and domain.lower() != b for b in _BRAND_KEYWORDS))
    brand_in_subdomain = int(any(b in subdomain.lower() for b in _BRAND_KEYWORDS))
    tp = [("0","o"),("1","l"),("3","e"),("4","a"),("5","s")]
    has_typosquat     = int(any(
        any(b.replace(o, r) == domain.lower() for o, r in tp) for b in _BRAND_KEYWORDS
    ))
    has_punycode         = int("xn--" in hostname.lower())
    excessive_subdomains = int(num_subdomains > 3)
    suspicious_tld       = int(suffix.lower() in _SUSPICIOUS_TLDS)
    free_hosting         = int(any(fh in full_host for fh in _FREE_HOSTING))
    is_shortener         = int(any(s in full_host for s in _URL_SHORTENERS))
    excessive_hyphens    = int(domain.count("-") >= 3)
    port                 = parsed.port
    has_nonstandard_port = int(port is not None and port not in (80, 443, 8080))
    high_entropy         = int(domain_entropy > 3.8)
    long_url             = int(url_len > 100)

    feat = np.array([
        url_len, num_subdomains, is_ip, is_https,
        num_at, num_dash, num_double_slash, domain_entropy,
        num_params, has_susp_params,
        brand_in_domain, brand_in_subdomain, has_typosquat,
        has_punycode, excessive_subdomains, suspicious_tld,
        free_hosting, is_shortener, excessive_hyphens,
        has_nonstandard_port, high_entropy, long_url,
        float(fetch_error_type == "unresolvable"),
        float(fetch_error_type == "ssl"),
        float(fetch_error_type == "refused"),
        float(fetch_error_type not in ("none",)),
    ], dtype=float).reshape(1, -1)

    meta = {
        "is_ip": is_ip, "num_subdomains": num_subdomains, "is_https": is_https,
        "num_at": num_at, "brand_in_domain": brand_in_domain,
        "brand_in_subdomain": brand_in_subdomain, "has_typosquat": has_typosquat,
        "has_punycode": has_punycode, "excessive_subdomains": excessive_subdomains,
        "suspicious_tld": suspicious_tld, "free_hosting": free_hosting,
        "is_shortener": is_shortener, "excessive_hyphens": excessive_hyphens,
        "has_nonstandard_port": has_nonstandard_port, "high_entropy": high_entropy,
        "long_url": long_url, "has_susp_params": has_susp_params,
        "fetch_error_type": fetch_error_type, "suffix": suffix,
        "domain": domain, "subdomain": subdomain,
    }
    return feat, meta


def extract_html_features(html_content: str, page_url: str) -> tuple:
    """Returns (feature_vector, metadata_dict)."""
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
    has_meta_r = 1 if any(
        m.get("http-equiv","").lower()=="refresh" for m in soup.find_all("meta")
    ) else 0
    script_text = "".join(s.get_text() for s in soup.find_all("script") if s.string)
    script_ratio = len(script_text) / max(1, len(html_content))
    fav_mismatch = 0
    for fav in soup.find_all("link", rel=lambda r: r and "icon" in r.lower()):
        if _get_domain(fav.get("href","")) not in ("", page_domain):
            fav_mismatch = 1; break
    has_auto = int(bool(forms) and "submit()" in script_text.lower())
    submits  = len(soup.find_all(["input","button"], type=lambda t: t and t.lower()=="submit"))
    submits += len(soup.find_all("input", type=lambda t: t and t.lower()=="image"))
    in_ratio = len(inputs) / max(1, submits) if inputs else 0.0

    feat = np.array([
        len(forms), len(inputs), len(iframes), num_ext_links, num_ext_scripts,
        has_pass, has_meta_r, script_ratio, fav_mismatch, has_auto,
        in_ratio, len(ext_domains),
    ], dtype=float).reshape(1, -1)

    meta = {
        "num_forms": len(forms), "num_inputs": len(inputs), "num_iframes": len(iframes),
        "num_ext_links": num_ext_links, "num_ext_scripts": num_ext_scripts,
        "has_password_field": has_pass, "has_meta_redirect": has_meta_r,
        "favicon_mismatch": fav_mismatch, "has_auto_submit": has_auto,
        "num_unique_ext_domains": len(ext_domains),
    }
    return feat, meta


def extract_visual_embedding(screenshot_path: str) -> np.ndarray:
    try:
        with Image.open(screenshot_path) as img:
            tensor = _preprocess(img.convert("RGB")).unsqueeze(0).to(_device)
        with torch.no_grad():
            emb = _mv2_net(tensor).mean([2, 3]).squeeze(0)
        return emb.cpu().numpy().reshape(1, -1)
    except Exception:
        return np.zeros((1, 1280), dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def _align(X: np.ndarray, exp: int) -> np.ndarray:
    cur = X.shape[1]
    if cur < exp:   return np.hstack([X, np.zeros((1, exp - cur))])
    if cur > exp:   return X[:, :exp]
    return X


def run_url_only(X_url: np.ndarray) -> dict:
    model, scaler = _url_bundle["model"], _url_bundle["scaler"]
    X_s  = scaler.transform(_align(X_url, scaler.n_features_in_))
    pred = int(model.predict(X_s)[0])
    prob = float(model.predict_proba(X_s)[0][1]) * 100
    return {"prediction": pred, "phishing_prob": round(prob, 2),
            "mode": "URL-only (fallback)"}


def run_fusion(X_url, X_html, X_visual) -> dict:
    scalers = _fusion_bundle["scalers"]
    weights = _fusion_bundle["weights"]
    model   = _fusion_bundle["model"]
    def sw(name, X):
        sc = scalers[name]
        return sc.transform(_align(X, sc.n_features_in_)) * weights[name]
    X_fused = np.hstack([sw("url", X_url), sw("html", X_html), sw("visual", X_visual)])
    pred = int(model.predict(X_fused)[0])
    prob = float(model.predict_proba(X_fused)[0][1]) * 100
    return {"prediction": pred, "phishing_prob": round(prob, 2),
            "mode": "Fusion MLP"}


# ══════════════════════════════════════════════════════════════════════════════
#  REASONS  — human-readable explanation of the decision
# ══════════════════════════════════════════════════════════════════════════════

def build_reasons(url_meta: dict, html_meta: dict | None, prob: float,
                  dns_ok: bool, page_ok: bool) -> list:
    reasons = []

    # DNS / reachability
    if not dns_ok:
        reasons.append("🔴 Domain does not resolve — likely dead or newly registered phishing domain")
    if url_meta["fetch_error_type"] == "ssl":
        reasons.append("🔴 Invalid SSL certificate — HTTPS spoofing or self-signed cert")
    if url_meta["fetch_error_type"] == "refused":
        reasons.append("⚠️ Connection refused — server may be shutting down after phishing campaign")

    # URL structure signals
    if url_meta["is_ip"]:
        reasons.append("🔴 URL uses raw IP address instead of a domain name")
    if url_meta["brand_in_domain"]:
        reasons.append("🔴 Brand name impersonated in the domain (e.g. paypal-secure.xyz)")
    if url_meta["brand_in_subdomain"]:
        reasons.append("🔴 Brand name used in subdomain to appear legitimate (e.g. paypal.evil.com)")
    if url_meta["has_typosquat"]:
        reasons.append("🔴 Typosquatting detected — domain mimics a known brand with character substitution")
    if url_meta["has_punycode"]:
        reasons.append("🔴 Punycode/homograph attack — uses Unicode characters to impersonate a real domain")
    if url_meta["suspicious_tld"]:
        reasons.append(f"🔴 Suspicious TLD: '.{url_meta['suffix']}' is commonly abused by phishers")
    if url_meta["free_hosting"]:
        reasons.append("⚠️ Free hosting / subdomain abuse platform detected (Netlify, Wix, etc.)")
    if url_meta["is_shortener"]:
        reasons.append("⚠️ URL shortener detected — hides the true destination")
    if url_meta["excessive_subdomains"]:
        reasons.append(f"⚠️ Excessive subdomains ({url_meta['num_subdomains']}) — common in subdomain-abuse phishing")
    if url_meta["excessive_hyphens"]:
        reasons.append("⚠️ Domain contains 3+ hyphens — pattern common in phishing URLs")
    if url_meta["high_entropy"]:
        reasons.append("⚠️ High character entropy in domain — looks like a randomly generated string")
    if url_meta["long_url"]:
        reasons.append("⚠️ Unusually long URL (>100 chars) — used to hide the real domain")
    if url_meta["num_at"] > 0:
        reasons.append("🔴 '@' symbol in URL — causes browsers to ignore text before it")
    if url_meta["has_susp_params"]:
        reasons.append("⚠️ Suspicious query parameters (login/verify/redirect/secure)")
    if not url_meta["is_https"] and dns_ok:
        reasons.append("⚠️ Page served over HTTP (not HTTPS)")
    if url_meta["has_nonstandard_port"]:
        reasons.append("⚠️ Non-standard port detected — legitimate sites rarely use custom ports")

    # HTML signals (only when page loaded)
    if html_meta:
        if html_meta["has_password_field"]:
            reasons.append("🔴 Password input field detected on page")
        if html_meta["has_meta_redirect"]:
            reasons.append("🔴 Meta-refresh redirect found — page redirects automatically")
        if html_meta["favicon_mismatch"]:
            reasons.append("⚠️ Favicon loaded from a different domain — impersonation indicator")
        if html_meta["has_auto_submit"]:
            reasons.append("🔴 Auto-submitting form detected — page submits data without user action")
        if html_meta["num_ext_scripts"] > 5:
            reasons.append(f"⚠️ High number of external scripts ({html_meta['num_ext_scripts']}) from outside domains")
        if html_meta["num_forms"] > 2:
            reasons.append(f"⚠️ Multiple forms ({html_meta['num_forms']}) on page — unusual for legitimate sites")
        if html_meta["num_iframes"] > 0:
            reasons.append(f"⚠️ {html_meta['num_iframes']} hidden iframe(s) — used for clickjacking or content injection")

    # If model is uncertain but flagging
    if not reasons and prob > 50:
        reasons.append("⚠️ Visual and structural features show patterns consistent with phishing")
    if not reasons:
        reasons.append("✅ No strong phishing indicators detected in URL, HTML, or visual features")

    return reasons


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": True})


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    url  = (data or {}).get("url", "").strip()

    if not url:
        return jsonify({"error": "Missing 'url' field"}), 400

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    t0 = time.time()

    # 1. DNS check
    dns_ok = check_dns(url)

    # 2. Fetch page
    fetch   = fetch_page(url, timeout=15)
    page_ok = fetch["success"]

    # 3. URL features (always)
    X_url, url_meta = extract_url_features(url, fetch["error_type"])

    # 4. Choose inference path
    html_meta = None
    if page_ok:
        X_html, html_meta = extract_html_features(fetch["html"], url)
        X_visual = extract_visual_embedding(fetch["screenshot_path"])
        result   = run_fusion(X_url, X_html, X_visual)
    else:
        result = run_url_only(X_url)

    # 5. Cleanup screenshot
    if fetch.get("screenshot_path"):
        try: os.unlink(fetch["screenshot_path"])
        except Exception: pass

    prob      = result["phishing_prob"]
    pred_int  = result["prediction"]
    label     = "PHISHING" if pred_int == 1 else "LEGITIMATE"
    latency   = round(time.time() - t0, 2)

    # Risk level
    if prob >= 80:   risk = "HIGH"
    elif prob >= 55: risk = "MEDIUM"
    elif prob >= 35: risk = "LOW"
    else:            risk = "SAFE"

    reasons = build_reasons(url_meta, html_meta, prob, dns_ok, page_ok)

    return jsonify({
        "prediction":          label,
        "phishing_probability": prob,
        "risk_level":          risk,
        "reasons":             reasons,
        "mode":                result["mode"],
        "dns_resolved":        dns_ok,
        "page_loaded":         page_ok,
        "latency_s":           latency,
    })


if __name__ == "__main__":
    print("[FEDrA] API server running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
