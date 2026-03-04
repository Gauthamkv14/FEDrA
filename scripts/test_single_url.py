"""
scripts/test_single_url.py
==========================
FEDrA — Phishing detector for a single URL.

Works even when the page is unreachable:
  • Always extracts URL features (zero network needed)
  • Uses fetch failure itself as a phishing signal
  • Falls back to URL-only verdict when HTML/Visual are unavailable

Local only — no external APIs.

NOTE: Models saved with joblib.dump() — always load with joblib.load()

Usage:
    python scripts/test_single_url.py --url "https://example.com"
    python scripts/test_single_url.py --url "https://example.com" --timeout 15
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
import joblib                           # NOTE: always joblib.load() for .pkl files
from bs4 import BeautifulSoup
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILES = {
    "url":   os.path.join(MODELS_DIR, "url_baseline.pkl"),
    "html":  os.path.join(MODELS_DIR, "html_baseline.pkl"),
    "image": os.path.join(MODELS_DIR, "image_baseline.pkl"),
}

# Chrome errors that map to fetch-failure phishing signals
_ERR_UNRESOLVABLE  = ("ERR_NAME_NOT_RESOLVED", "ERR_NAME_CHANGED")
_ERR_SSL           = ("ERR_SSL_PROTOCOL_ERROR", "ERR_CERT_", "ERR_BAD_SSL_CLIENT_AUTH_CERT",
                      "ERR_SSL_VERSION_OR_CIPHER_MISMATCH", "SSL_ERROR")
_ERR_REFUSED       = ("ERR_CONNECTION_REFUSED", "ERR_EMPTY_RESPONSE",
                      "ERR_TUNNEL_CONNECTION_FAILED", "ERR_SOCKET_NOT_CONNECTED")
_ERR_TIMEOUT       = ("ERR_TIMED_OUT", "ERR_CONNECTION_TIMED_OUT", "Timeout")

# Suspicious TLDs commonly abused by phishers
_SUSPICIOUS_TLDS = {
    "xyz", "tk", "ml", "ga", "cf", "gq", "top", "click",
    "work", "loan", "men", "date", "racing", "party", "trade",
    "kim", "country", "stream", "download", "gdn", "bid",
    "accountant", "faith", "review", "science", "win",
}

# Free-hosting / subdomain-abuse domains
_FREE_HOSTING = {
    "000webhostapp.com", "weebly.com", "wixsite.com", "wordpress.com",
    "blogspot.com", "netlify.app", "github.io", "glitch.me",
    "firebaseapp.com", "web.app", "surge.sh", "pages.dev",
}

# Known URL-shortener domains
_URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "buff.ly", "is.gd", "short.io", "rebrand.ly",
}

# Brands commonly impersonated
_BRAND_KEYWORDS = [
    "paypal", "amazon", "apple", "google", "microsoft", "facebook",
    "instagram", "netflix", "dropbox", "linkedin", "twitter", "ebay",
    "wellsfargo", "chase", "citibank", "bankofamerica", "irs",
    "dhl", "fedex", "usps", "whatsapp", "telegram",
]

SUSPICIOUS_QUERY_WORDS = {"login", "redirect", "verify", "secure"}


# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN RESOLUTION CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_domain_resolves(url: str) -> bool:
    """Return True if the hostname resolves via DNS, False otherwise."""
    try:
        parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
        hostname = parsed.hostname or ""
        if not hostname:
            return False
        # Quick IP check — IPs always "resolve"
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
    """Holds outcome of a page-fetch attempt."""
    def __init__(self):
        self.html: str | None         = None
        self.screenshot: str | None   = None
        self.success: bool            = False
        self.error_type: str          = "none"   # none | unresolvable | ssl | refused | timeout | other
        self.error_msg: str           = ""

    @property
    def available(self) -> bool:
        return self.success and self.html is not None


def _classify_error(msg: str) -> str:
    m = msg.upper()
    if any(e.upper() in m for e in _ERR_UNRESOLVABLE):
        return "unresolvable"
    if any(e.upper() in m for e in _ERR_SSL):
        return "ssl"
    if any(e.upper() in m for e in _ERR_REFUSED):
        return "refused"
    if any(e.upper() in m for e in _ERR_TIMEOUT):
        return "timeout"
    return "other"


def fetch_page(url: str, timeout: int = 20) -> FetchResult:
    """
    Attempt to load the page with headless Chrome.
    Returns a FetchResult — never raises; always continues.
    """
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
        result.error_msg  = str(e)
        result.error_type = _classify_error(str(e))

    except Exception as e:
        result.error_msg  = str(e)
        result.error_type = "other"

    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  URL FEATURE EXTRACTION  (extended — same core as extract_url_features.py
#  plus 12 additional phishing-pattern detections)
# ══════════════════════════════════════════════════════════════════════════════

def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def _tld_extract_simple(url: str) -> tuple[str, str, str]:
    """Lightweight tldextract substitute (no external lib needed)."""
    parsed   = urllib.parse.urlparse(url if "://" in url else "http://" + url)
    hostname = parsed.hostname or ""
    parts    = hostname.split(".")

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


def extract_url_features(url: str, fetch_error_type: str = "none") -> np.ndarray:
    """
    Returns a feature vector covering:
      - Original 10 features (matching training schema)
      - 12 extended phishing-pattern features
      - 4 fetch-failure signal features
    Total: 26 features before padding to scaler's expected width.
    """
    parsed_url = url if "://" in url else "http://" + url
    parsed     = urllib.parse.urlparse(parsed_url)
    subdomain, domain, suffix = _tld_extract_simple(parsed_url)
    hostname   = parsed.hostname or ""

    # ── Original 10 features (must match training order) ──────────────────
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

    # ── Extended phishing-pattern features ────────────────────────────────

    # 1. Brand impersonation — brand keyword in domain but domain ≠ brand
    brand_in_domain = int(any(
        b in domain.lower() and domain.lower() != b
        for b in _BRAND_KEYWORDS
    ))

    # 2. Brand in subdomain (misleading path: paypal.com.evil.net)
    brand_in_subdomain = int(any(b in subdomain.lower() for b in _BRAND_KEYWORDS))

    # 3. Typosquatting — digit substitutions (0→o, 1→l, 3→e, etc.)
    typo_patterns  = [("0", "o"), ("1", "l"), ("3", "e"), ("4", "a"), ("5", "s")]
    has_typosquat  = int(any(
        any(b.replace(orig, rep) == domain.lower() for orig, rep in typo_patterns)
        for b in _BRAND_KEYWORDS
    ))

    # 4. Homograph / punycode attack
    has_punycode = int("xn--" in hostname.lower())

    # 5. Excessive subdomains (>3 levels is suspicious)
    excessive_subdomains = int(num_subdomains > 3)

    # 6. Suspicious TLD
    suspicious_tld = int(suffix.lower() in _SUSPICIOUS_TLDS)

    # 7. Free-hosting / subdomain abuse
    full_host      = hostname.lower()
    free_hosting   = int(any(fh in full_host for fh in _FREE_HOSTING))

    # 8. URL shortener
    is_shortener   = int(any(s in full_host for s in _URL_SHORTENERS))

    # 9. Excessive hyphens in domain (≥3)
    excessive_hyphens = int(domain.count("-") >= 3)

    # 10. Non-standard port
    port = parsed.port
    has_nonstandard_port = int(port is not None and port not in (80, 443, 8080))

    # 11. High character entropy in domain token (random strings)
    #     Legitimate domains tend to be low entropy; gibberish = high
    high_entropy = int(domain_entropy > 3.8)

    # 12. Excessive URL length (>100 chars is suspicious)
    long_url = int(url_len > 100)

    extended = np.array([
        brand_in_domain, brand_in_subdomain, has_typosquat,
        has_punycode, excessive_subdomains, suspicious_tld,
        free_hosting, is_shortener, excessive_hyphens,
        has_nonstandard_port, high_entropy, long_url,
    ], dtype=float)

    # ── Fetch-failure signal features ─────────────────────────────────────
    # These elevate phishing probability when the page is unreachable.
    is_unresolvable   = float(fetch_error_type == "unresolvable")
    is_ssl_error      = float(fetch_error_type == "ssl")
    is_refused        = float(fetch_error_type == "refused")
    is_fetch_failed   = float(fetch_error_type not in ("none",))

    fail_feats = np.array([
        is_unresolvable, is_ssl_error, is_refused, is_fetch_failed,
    ], dtype=float)

    return np.concatenate([core, extended, fail_feats]).reshape(1, -1)


# ══════════════════════════════════════════════════════════════════════════════
#  HTML FEATURE EXTRACTION  (unchanged — mirrors extract_html_features.py)
# ══════════════════════════════════════════════════════════════════════════════

def _get_domain(url: str) -> str:
    if not url or not isinstance(url, str):
        return ""
    if not url.startswith("http") and not url.startswith("//"):
        return ""
    _, domain, suffix = _tld_extract_simple(url)
    return f"{domain}.{suffix}" if domain else ""


def extract_html_features(html_content: str, page_url: str) -> np.ndarray:
    """12-feature vector — same schema as html_features.csv."""
    page_domain = _get_domain(page_url)
    soup        = BeautifulSoup(html_content, "html.parser")
    content     = html_content

    forms, inputs, iframes = soup.find_all("form"), soup.find_all("input"), soup.find_all("iframe")
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

    favicon_mismatch = 0
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


# ══════════════════════════════════════════════════════════════════════════════
#  VISUAL EMBEDDING  (unchanged — mirrors extract_visual_embeddings.py)
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


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _align(X: np.ndarray, expected: int) -> np.ndarray:
    cur = X.shape[1]
    if cur < expected:
        return np.hstack([X, np.zeros((1, expected - cur))])
    return X[:, :expected]


def _predict_model(modality: str, X: np.ndarray) -> dict:
    """
    Load a single model via joblib.load() and return prediction dict.
    NOTE: Models saved with joblib.dump() — always load with joblib.load()
    """
    path = MODEL_FILES[modality]
    if not os.path.isfile(path):
        return {"error": f"Model not found: {path}"}
    try:
        bundle  = joblib.load(path)          # NOTE: joblib.load()
        model   = bundle["model"]
        scaler  = bundle["scaler"]
        X_s     = scaler.transform(_align(X, scaler.n_features_in_))
        pred    = int(model.predict(X_s)[0])
        prob    = float(model.predict_proba(X_s)[0][1])
        return {
            "prediction":   pred,
            "label":        "PHISHING" if pred == 1 else "LEGITIMATE",
            "phishing_prob": round(prob * 100, 2),
        }
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  VERDICT LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def determine_verdict(
    url_prob: float,
    domain_resolved: bool,
    html_result: dict | None,
    vis_result: dict | None,
    fetch_result: FetchResult,
) -> tuple[str, str]:
    """
    Returns (verdict_label, confidence_note).
    """
    u = url_prob / 100.0  # normalise to [0,1]

    if not domain_resolved:
        if u > 0.5:
            return "🚨 PHISHING", "HIGH CONFIDENCE — URL risk + domain unresolvable"
        else:
            return "⚠️  SUSPICIOUS", "dead/parked domain, low URL risk — flag for review"

    # Domain resolved but page fetch failed for other reasons
    if not fetch_result.success:
        if u > 0.5:
            return "🚨 PHISHING", "MEDIUM CONFIDENCE — URL risk (HTML/Visual unavailable)"
        else:
            return "⚠️  SUSPICIOUS", "page unreachable, URL appears low risk — needs manual check"

    # All three models available — standard majority vote
    votes = []
    for r in (html_result, vis_result):
        if r and "prediction" in r:
            votes.append(r["prediction"])
    if u > 0.5:
        votes.append(1)
    else:
        votes.append(0)

    majority = int(sum(votes) > len(votes) / 2)
    if majority == 1:
        return "🚨 PHISHING", "majority vote (URL + HTML + Visual)"
    else:
        return "✅ LEGITIMATE", "majority vote (URL + HTML + Visual)"


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

_BAR = "=" * 65

def _fmt_row(name: str, result: dict | None, unavailable_reason: str = "UNAVAILABLE") -> str:
    if result is None:
        return f"  {name:<12}  {'N/A':>17}   ⚠️  {unavailable_reason}"
    if "error" in result:
        return f"  {name:<12}  {'ERROR':>17}   ❌ {result['error'][:35]}"
    icon  = "🚨" if result["prediction"] == 1 else "✅"
    prob  = result["phishing_prob"]
    label = result["label"]
    return f"  {name:<12}  {prob:>15.2f}%   {icon}  {label}"


def _fmt_domain_row(resolved: bool, error_type: str) -> str:
    if resolved:
        return f"  {'Domain resolved?':<12}  {'YES':>17}   ✅ OK"
    labels = {
        "unresolvable": "🚨 PHISHING SIGNAL — DNS not found",
        "ssl":          "🚨 PHISHING SIGNAL — SSL cert invalid",
        "refused":      "⚠️  mild signal — connection refused",
        "timeout":      "⚠️  mild signal — timed out",
        "other":        "⚠️  page unreachable",
    }
    note = labels.get(error_type, "⚠️  unknown error")
    return f"  {'Domain resolved?':<12}  {'NO':>17}   {note}"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FEDrA — Phishing detector for a single URL (graceful fallback mode)"
    )
    parser.add_argument("--url",     required=True, help="URL to classify")
    parser.add_argument("--timeout", type=int, default=20, help="Page load timeout (s)")
    args = parser.parse_args()
    url  = args.url

    print(f"\n{_BAR}")
    print(f"  FEDrA — Single URL Phishing Detector")
    print(_BAR)
    print(f"  URL : {url}\n")

    # ── 1. DNS check (fast, no browser needed) ─────────────────────────────
    print("  [1/4] Checking DNS resolution...")
    domain_resolved = check_domain_resolves(url)
    dns_icon = "✅ resolves" if domain_resolved else "❌ UNRESOLVABLE"
    print(f"        Domain → {dns_icon}")

    # ── 2. URL model (always runs — zero network needed) ──────────────────
    print("\n  [2/4] Extracting URL features & running URL model...")
    # We'll re-run after fetch to inject error type; for now use "none"
    X_url_pre = extract_url_features(url, "none")
    url_result_pre = _predict_model("url", X_url_pre)
    print(f"        Raw URL features: {X_url_pre.shape[1]} dims → "
          f"phishing prob = {url_result_pre.get('phishing_prob', '?')}%")

    # ── 3. Fetch page (graceful — never aborts) ────────────────────────────
    print(f"\n  [3/4] Fetching page (headless Chrome, timeout={args.timeout}s)...")
    fetch = fetch_page(url, timeout=args.timeout)

    if fetch.success:
        html_kb = len(fetch.html) / 1024
        print(f"        ✅ Loaded — HTML: {html_kb:.1f} KB | screenshot: {fetch.screenshot}")
    else:
        etype = fetch.error_type
        print(f"        ⚠️  Fetch failed [{etype}]: {fetch.error_msg[:80]}")
        print(f"        → Continuing in URL-only mode")

    # Re-run URL extraction with error type now known
    X_url      = extract_url_features(url, fetch.error_type)
    url_result = _predict_model("url", X_url)

    # ── 4. HTML + Visual (only when page loaded) ───────────────────────────
    html_result = None
    vis_result  = None

    if fetch.available:
        print("\n  [4/4] Extracting HTML and Visual features...")
        X_html      = extract_html_features(fetch.html, url)
        html_result = _predict_model("html", X_html)
        print(f"        HTML features: {X_html.shape[1]} dims")

        X_vis      = extract_visual_embedding(fetch.screenshot)
        vis_result = _predict_model("image", X_vis)
        print(f"        Visual embedding: {X_vis.shape[1]} dims")
    else:
        print("\n  [4/4] Skipping HTML + Visual (page unavailable)")

    # ── 5. Verdict ─────────────────────────────────────────────────────────
    url_prob = url_result.get("phishing_prob", 0.0)
    verdict, confidence = determine_verdict(
        url_prob, domain_resolved, html_result, vis_result, fetch
    )

    # ── 6. Print report ────────────────────────────────────────────────────
    print(f"\n{_BAR}")
    print(f"  {'MODEL':<12}  {'PROB (phishing)':>17}   LABEL")
    print(f"  {'-' * 61}")

    print(_fmt_row("URL",    url_result))

    if fetch.available:
        print(_fmt_row("HTML",   html_result))
        print(_fmt_row("Visual", vis_result))
    else:
        unavail = {
            "unresolvable": "domain not found",
            "ssl":          "SSL error",
            "refused":      "connection refused",
            "timeout":      "timed out",
            "other":        "fetch failed",
        }.get(fetch.error_type, "UNAVAILABLE")
        print(_fmt_row("HTML",   None, unavail))
        print(_fmt_row("Visual", None, unavail))

    print(_fmt_domain_row(domain_resolved, fetch.error_type))

    print(f"\n  {'-' * 61}")
    print(f"  Final verdict → {verdict}")
    print(f"  Confidence    → {confidence}")
    print(f"{_BAR}\n")

    # Cleanup temp screenshot
    if fetch.screenshot:
        try:
            os.unlink(fetch.screenshot)
        except Exception:
            pass


if __name__ == "__main__":
    main()
