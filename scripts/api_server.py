"""
scripts/api_server.py
=====================
FEDrA Flask API — Multimodal Phishing Detection Backend.

Supports dual acquisition and feature-extraction modes:
1. Browser-Native Execution (Default for Chrome Extension):
   - Consumes browser-extracted 22 URL features and 12 HTML features directly from content script.
   - Decodes browser-captured visible screenshot in-memory (1280-dim MobileNetV2 embedding).
   - Zero Selenium processes spawned.
   - Zero server DNS lookups.
   - Sub-100ms detection latency.
2. Server-Side Fallback (Legacy CLI / Standalone testing):
   - Re-extracts features on server and uses Selenium when client vectors are not supplied.

Feature & Model Dimensionality Invariants:
- URL feature vector: 22 dimensions
- HTML feature vector: 12 dimensions
- Visual embedding vector: 1280 dimensions
- Concatenated fusion input: 1314 dimensions

Run:
    python scripts/api_server.py
"""

import os
import sys
import time
import socket
import base64
import io
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

_ERR_UNRESOLVABLE = ("ERR_NAME_NOT_RESOLVED", "ERR_NAME_CHANGED")
_ERR_SSL = ("ERR_SSL_PROTOCOL_ERROR", "ERR_CERT_", "ERR_SSL_VERSION_OR_CIPHER_MISMATCH", "SSL_ERROR")
_ERR_REFUSED = ("ERR_CONNECTION_REFUSED", "ERR_EMPTY_RESPONSE", "ERR_TUNNEL_CONNECTION_FAILED")
_ERR_TIMEOUT = ("ERR_TIMED_OUT", "ERR_CONNECTION_TIMED_OUT", "Timeout")


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP — Load all models once into memory
# ══════════════════════════════════════════════════════════════════════════════

print("[FEDrA] Loading models at startup (Canonical Schema v1)...")
_fusion_bundle = joblib.load(FUSION_MODEL_PATH)
_url_bundle    = joblib.load(URL_MODEL_PATH)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_mv2_net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
_mv2_net.eval()
_mv2_net.to(_device)
_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print(f"[FEDrA] Models ready on {_device}. URL feature dim: {URL_FEATURE_DIM}, Fused dim: 1314. Server starting...")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS & ACQUISITION
# ══════════════════════════════════════════════════════════════════════════════

def _get_domain(url: str) -> str:
    if not url or not isinstance(url, str): return ""
    if not url.startswith("http") and not url.startswith("//"): return ""
    parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
    return parsed.hostname or ""


def _classify_error(msg: str) -> str:
    m = msg.upper()
    if any(e.upper() in m for e in _ERR_UNRESOLVABLE): return "unresolvable"
    if any(e.upper() in m for e in _ERR_SSL):          return "ssl"
    if any(e.upper() in m for e in _ERR_REFUSED):      return "refused"
    if any(e.upper() in m for e in _ERR_TIMEOUT):      return "timeout"
    return "other"


def check_dns(url: str) -> bool:
    """Diagnostic DNS check for legacy fallback mode."""
    try:
        parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
        hostname = parsed.hostname or ""
        if not hostname: return False
        import re
        if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname): return True
        socket.setdefaulttimeout(3)
        socket.getaddrinfo(hostname, None)
        return True
    except Exception:
        return False


def fetch_page_selenium(url: str, timeout: int = 15) -> dict:
    """Legacy server-side Selenium fallback when client DOM is not supplied."""
    out = {"success": False, "html": None, "screenshot_path": None,
           "error_type": "none", "error_msg": ""}
    opts = Options()
    for arg in ["--headless", "--no-sandbox", "--disable-dev-shm-usage",
                "--disable-gpu", "--window-size=1280,800", "--log-level=3", "--silent"]:
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


def extract_html_features(html_content: str, page_url: str) -> tuple:
    """Extract canonical 12 DOM features from raw HTML string."""
    if not html_content or not isinstance(html_content, str):
        feat = np.zeros((1, 12), dtype=float)
        meta = {
            "num_forms": 0, "num_inputs": 0, "num_iframes": 0,
            "num_ext_links": 0, "num_ext_scripts": 0,
            "has_password_field": 0, "has_meta_redirect": 0,
            "script_content_ratio": 0.0, "favicon_mismatch": 0,
            "has_auto_submit": 0, "input_submit_ratio": 0.0,
            "num_unique_ext_domains": 0,
        }
        return feat, meta

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
    has_meta_r = 1 if any(
        m.get("http-equiv", "").lower() == "refresh" for m in soup.find_all("meta")
    ) else 0
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


def extract_visual_embedding(screenshot_source) -> np.ndarray:
    """
    Extract 1280-dim MobileNetV2 embedding from:
    - Base64 data URL (e.g. 'data:image/png;base64,...')
    - Raw image bytes
    - Filepath string
    - None / empty -> returns zero vector (1, 1280)
    """
    if not screenshot_source:
        return np.zeros((1, 1280), dtype=np.float32)

    try:
        img = None
        if isinstance(screenshot_source, str):
            if screenshot_source.startswith("data:image"):
                b64_data = screenshot_source.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_data)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            elif os.path.isfile(screenshot_source):
                with Image.open(screenshot_source) as f:
                    img = f.convert("RGB")
        elif isinstance(screenshot_source, bytes):
            img = Image.open(io.BytesIO(screenshot_source)).convert("RGB")

        if img is None:
            return np.zeros((1, 1280), dtype=np.float32)

        tensor = _preprocess(img).unsqueeze(0).to(_device)
        with torch.no_grad():
            emb = _mv2_net(tensor).mean([2, 3]).squeeze(0)
        return emb.cpu().numpy().reshape(1, -1)
    except Exception as e:
        print(f"[FEDrA] Visual extraction error: {e}")
        return np.zeros((1, 1280), dtype=np.float32)


def validate_client_visual_embedding(emb) -> bool:
    """
    Strictly validates client-supplied visual embedding:
    - Must be a list of exactly 1280 numeric elements.
    - No booleans, strings, None, NaN, or Infinity.
    - Must convert cleanly to finite float32 array.
    """
    if not isinstance(emb, list) or len(emb) != 1280:
        return False
    for x in emb:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return False
        if not np.isfinite(x):
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE ROUTINES
# ══════════════════════════════════════════════════════════════════════════════

def run_url_only(X_url: np.ndarray) -> dict:
    model, scaler = _url_bundle["model"], _url_bundle["scaler"]
    assert X_url.shape[1] == scaler.n_features_in_, (
        f"URL dim mismatch: got {X_url.shape[1]}, expected {scaler.n_features_in_}"
    )
    X_s = scaler.transform(X_url)
    pred = int(model.predict(X_s)[0])
    prob = float(model.predict_proba(X_s)[0][1]) * 100
    return {
        "prediction": pred,
        "phishing_prob": round(prob, 2),
        "mode": "URL-only (fallback)",
    }


def run_fusion(X_url: np.ndarray, X_html: np.ndarray, X_visual: np.ndarray) -> dict:
    scalers = _fusion_bundle["scalers"]
    weights = _fusion_bundle["weights"]
    model   = _fusion_bundle["model"]

    assert X_url.shape[1] == 22, f"Expected 22 URL features, got {X_url.shape[1]}"
    assert X_html.shape[1] == 12, f"Expected 12 HTML features, got {X_html.shape[1]}"
    assert X_visual.shape[1] == 1280, f"Expected 1280 visual features, got {X_visual.shape[1]}"

    X_url_sw = scalers["url"].transform(X_url) * weights["url"]
    X_html_sw = scalers["html"].transform(X_html) * weights["html"]
    X_visual_sw = scalers["visual"].transform(X_visual) * weights["visual"]

    X_fused = np.hstack([X_url_sw, X_html_sw, X_visual_sw])
    assert X_fused.shape[1] == 1314, f"Expected 1314 fused dims, got {X_fused.shape[1]}"

    pred = int(model.predict(X_fused)[0])
    prob = float(model.predict_proba(X_fused)[0][1]) * 100
    return {
        "prediction": pred,
        "phishing_prob": round(prob, 2),
        "mode": "Fusion MLP",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  REASONS GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def build_reasons(url_meta: dict, html_meta: dict | None, prob: float,
                  dns_ok: bool, page_ok: bool, fetch_error_type: str = "none") -> list:
    reasons = []

    # DNS / reachability
    if not dns_ok:
        reasons.append("Domain does not resolve — likely dead or newly registered phishing domain")
    if fetch_error_type == "ssl":
        reasons.append("Invalid SSL certificate — HTTPS spoofing or self-signed cert")
    if fetch_error_type == "refused":
        reasons.append("Connection refused — server may be shutting down after phishing campaign")

    # URL structure signals
    if url_meta.get("has_ip"):
        reasons.append("URL uses raw IP address instead of a domain name")
    if url_meta.get("brand_in_domain"):
        reasons.append("Brand name impersonated in domain")
    if url_meta.get("brand_in_subdomain"):
        reasons.append("Brand name used in subdomain to appear legitimate")
    if url_meta.get("has_typosquat"):
        reasons.append("Typosquatting detected — domain mimics a known brand with character substitution")
    if url_meta.get("has_punycode"):
        reasons.append("Punycode/homograph attack — uses Unicode characters to impersonate a real domain")
    if url_meta.get("is_suspicious_tld"):
        reasons.append(f"Suspicious TLD (.{url_meta.get('_suffix', '')}) commonly abused by phishers")
    if url_meta.get("free_hosting"):
        reasons.append("Free hosting / subdomain abuse platform detected")
    if url_meta.get("is_shortener"):
        reasons.append("URL shortener detected — hides the true destination")
    if url_meta.get("excessive_subdomains"):
        reasons.append(f"Excessive subdomains ({url_meta.get('num_subdomains', 0)}) — common in phishing")
    if url_meta.get("excessive_hyphens"):
        reasons.append("Domain contains 3+ hyphens — pattern common in phishing URLs")
    if url_meta.get("high_entropy"):
        reasons.append("High character entropy in domain — looks like a randomly generated string")
    if url_meta.get("long_url"):
        reasons.append("Unusually long URL (>100 chars) — used to hide real domain")
    if url_meta.get("count_at", 0) > 0:
        reasons.append("'@' symbol in URL — causes browsers to ignore text before it")
    if url_meta.get("has_suspicious_params"):
        reasons.append("Suspicious query parameters (login/verify/redirect/secure)")
    if not url_meta.get("is_https") and dns_ok:
        reasons.append("Page served over unencrypted HTTP (not HTTPS)")
    if url_meta.get("has_nonstandard_port"):
        reasons.append("Non-standard port detected")

    # HTML signals
    if html_meta:
        if html_meta["has_password_field"]:
            reasons.append("Password input field detected on page")
        if html_meta["has_meta_redirect"]:
            reasons.append("Meta-refresh redirect found — page redirects automatically")
        if html_meta["favicon_mismatch"]:
            reasons.append("Favicon loaded from a different domain — impersonation indicator")
        if html_meta["has_auto_submit"]:
            reasons.append("Auto-submitting form detected")
        if html_meta["num_ext_scripts"] > 5:
            reasons.append(f"High number of external scripts ({html_meta['num_ext_scripts']})")
        if html_meta["num_forms"] > 2:
            reasons.append(f"Multiple forms ({html_meta['num_forms']}) on page")
        if html_meta["num_iframes"] > 0:
            reasons.append(f"{html_meta['num_iframes']} iframe(s) detected")

    if not reasons and prob > 50:
        reasons.append("Visual and structural features show patterns consistent with phishing")
    if not reasons:
        reasons.append("No strong phishing indicators detected in URL, HTML, or visual features")

    return reasons


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTING
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": True,
        "schema_version": URL_SCHEMA_VERSION,
        "url_feature_dim": URL_FEATURE_DIM,
        "html_feature_dim": 12,
        "visual_feature_dim": 1280,
        "fused_dim": 1314,
        "device": str(_device),
        "supported_acquisition_modes": ["browser_native", "server_selenium_fallback"]
    })


@app.route("/test/legit_sample", methods=["GET"])
def test_legit_sample():
    return """<!DOCTYPE html>
<html>
<head><title>Legitimate Corporate Portal</title></head>
<body style="font-family: Arial, sans-serif; padding: 40px; background: #f8f9fa;">
    <h1>Welcome to Corporate Portal</h1>
    <p>This is a standard informational page without credentials inputs.</p>
    <a href="https://example.com/about">About Us</a> | <a href="https://example.com/contact">Contact</a>
</body>
</html>"""


@app.route("/test/phish_sample", methods=["GET"])
def test_phish_sample():
    return """<!DOCTYPE html>
<html>
<head><title>Security Update — Account Verification Required</title></head>
<body style="font-family: Arial, sans-serif; padding: 40px; background: #fff;">
    <div style="max-width: 400px; margin: auto; border: 1px solid #ccc; padding: 20px;">
        <h2>Verify Your PayPal Account</h2>
        <form action="/login" method="POST">
            <input type="text" name="email" placeholder="Email Address" style="width: 100%; margin-bottom: 10px;" />
            <input type="password" name="password" placeholder="Password" style="width: 100%; margin-bottom: 10px;" />
            <input type="submit" value="Log In & Verify" style="width: 100%; background: #0070ba; color: #fff; padding: 8px;" />
        </form>
    </div>
</body>
</html>"""


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True) or {}
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "Missing 'url' field"}), 400

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    t0 = time.time()

    client_html = data.get("html")
    client_screenshot = data.get("screenshot")
    client_url_vec = data.get("url_features_vector")
    client_html_vec = data.get("html_features_vector")
    client_url_dict = data.get("url_features_dict")
    client_html_dict = data.get("html_features_dict")
    is_dead = bool(data.get("dead", False))
    client_timings = data.get("timings", {})

    temp_screenshot_path = None
    html_meta = None
    dns_ok = True
    page_ok = False
    fetch_error = "none"

    # 1. URL Feature Vector (Use client-extracted if valid 22 dims, else extract)
    t_feat_start = time.time()
    if isinstance(client_url_vec, list) and len(client_url_vec) == 22:
        X_url = np.array(client_url_vec, dtype=np.float64).reshape(1, -1)
        url_meta = client_url_dict if isinstance(client_url_dict, dict) else extract_canonical_url_features_dict(url)
    else:
        X_url = extract_canonical_url_features_vector(url)
        url_meta = extract_canonical_url_features_dict(url)

    # 2. Determine Acquisition Mode & Extract Modality Vectors
    if is_dead:
        # Browser reported page is dead / unreachable
        acquisition_source = "client_dead_site"
        dns_ok = False
        page_ok = False
        t_feat_end = time.time()

        t_inf_start = time.time()
        result = run_url_only(X_url)
        t_inf_end = time.time()

    elif (client_html_vec is not None and len(client_html_vec) == 12) or (client_html is not None and len(client_html) > 0):
        # ── Primary Path: Browser-Native Acquisition & Features ────────────
        acquisition_source = "browser_native"
        dns_ok = True
        page_ok = True

        if isinstance(client_html_vec, list) and len(client_html_vec) == 12:
            X_html = np.array(client_html_vec, dtype=np.float64).reshape(1, -1)
            html_meta = client_html_dict if isinstance(client_html_dict, dict) else (
                extract_html_features(client_html, url)[1] if client_html else None
            )
        else:
            X_html, html_meta = extract_html_features(client_html, url)

        client_visual_emb = data.get("visual_embedding")
        visual_source = "server_pytorch"
        if validate_client_visual_embedding(client_visual_emb):
            X_visual = np.array(client_visual_emb, dtype=np.float32).reshape(1, -1)
            visual_source = "browser_onnx"
        else:
            X_visual = extract_visual_embedding(client_screenshot)
            visual_source = "server_pytorch"

        t_feat_end = time.time()

        t_inf_start = time.time()
        result = run_fusion(X_url, X_html, X_visual)
        result["visual_embedding_source"] = visual_source
        t_inf_end = time.time()

    else:
        # ── Fallback Path: Server-Side Selenium Re-fetch ───────────────────
        acquisition_source = "server_selenium_fallback"
        dns_ok = check_dns(url)

        fetch = fetch_page_selenium(url, timeout=15)
        page_ok = fetch["success"]
        fetch_error = fetch.get("error_type", "none")
        temp_screenshot_path = fetch.get("screenshot_path")

        if page_ok and fetch["html"]:
            X_html, html_meta = extract_html_features(fetch["html"], url)
            X_visual = extract_visual_embedding(temp_screenshot_path)
            t_feat_end = time.time()

            t_inf_start = time.time()
            result = run_fusion(X_url, X_html, X_visual)
            t_inf_end = time.time()
        else:
            t_feat_end = time.time()
            t_inf_start = time.time()
            result = run_url_only(X_url)
            t_inf_end = time.time()

    # Clean up temporary screenshot if created during Selenium fallback
    if temp_screenshot_path:
        try: os.unlink(temp_screenshot_path)
        except Exception: pass

    prob = result["phishing_prob"]
    pred_int = result["prediction"]
    label = "PHISHING" if pred_int == 1 else "LEGITIMATE"
    t_total_backend_s = round(time.time() - t0, 3)

    if prob >= 80:   risk = "HIGH"
    elif prob >= 55: risk = "MEDIUM"
    elif prob >= 35: risk = "LOW"
    else:            risk = "SAFE"

    reasons = build_reasons(url_meta, html_meta, prob, dns_ok, page_ok, fetch_error)

    t_feat_ms = round((t_feat_end - t_feat_start) * 1000, 2)
    t_inf_ms = round((t_inf_end - t_inf_start) * 1000, 2)
    t_total_ms = round(t_total_backend_s * 1000, 2)

    return jsonify({
        "prediction": label,
        "phishing_probability": prob,
        "risk_level": risk,
        "reasons": reasons,
        "mode": result["mode"],
        "acquisition_source": acquisition_source,
        "visual_embedding_source": result.get("visual_embedding_source", "server_pytorch"),
        "dns_resolved": dns_ok,
        "page_loaded": page_ok,
        "latency_s": t_total_backend_s,
        "timings": {
            "t_backend_feat_extract_ms": t_feat_ms,
            "t_backend_inference_ms": t_inf_ms,
            "t_backend_total_ms": t_total_ms,
            "t_client_readiness_ms": client_timings.get("t_readiness_ms", 0),
            "t_client_dom_ms": client_timings.get("t_dom_ms", 0),
            "t_client_url_feat_ms": client_timings.get("t_url_feat_ms", 0),
            "t_client_html_feat_ms": client_timings.get("t_html_feat_ms", 0),
            "t_client_screenshot_ms": client_timings.get("t_screenshot_ms", 0),
        },
        "schema_version": URL_SCHEMA_VERSION,
        "url_feature_dim": URL_FEATURE_DIM,
        "html_feature_dim": 12,
        "visual_feature_dim": 1280,
        "fused_dim": 1314,
    })


if __name__ == "__main__":
    print("[FEDrA] API server running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
