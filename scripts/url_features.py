"""
scripts/url_features.py
=======================
Canonical URL feature extraction module for the FEDrA pipeline.

Single source of truth for:
  - Canonical 22 URL feature names and order
  - Deterministic lexical URL feature extraction
  - Security keyword vocabularies (brands, suspicious TLDs, free hosting, shorteners)

Zero runtime network/DNS/Selenium dependencies.
"""

import math
import re
import urllib.parse
from typing import Dict, List, Tuple, Any
import numpy as np

try:
    import tldextract
    _HAS_TLDEXTRACT = True
except ImportError:
    _HAS_TLDEXTRACT = False

# ── Schema Metadata ───────────────────────────────────────────────────────────
URL_SCHEMA_VERSION = "v1"
URL_FEATURE_DIM = 22

CANONICAL_URL_FEATURE_NAMES: List[str] = [
    "url_len",
    "num_subdomains",
    "has_ip",
    "is_https",
    "count_at",
    "count_dash",
    "count_double_slash",
    "domain_entropy",
    "num_params",
    "has_suspicious_params",
    "brand_in_domain",
    "brand_in_subdomain",
    "has_typosquat",
    "has_punycode",
    "excessive_subdomains",
    "is_suspicious_tld",
    "free_hosting",
    "is_shortener",
    "excessive_hyphens",
    "has_nonstandard_port",
    "high_entropy",
    "long_url",
]

# ── Security Vocabularies & Heuristic Sets ────────────────────────────────────
SUSPICIOUS_QUERY_WORDS = {"login", "redirect", "verify", "secure"}

SUSPICIOUS_TLDS = {
    "xyz", "tk", "ml", "ga", "cf", "gq", "top", "click",
    "work", "loan", "men", "date", "racing", "party", "trade",
    "kim", "country", "stream", "download", "gdn", "bid",
    "accountant", "faith", "review", "science", "win",
}

FREE_HOSTING_DOMAINS = {
    "000webhostapp.com", "weebly.com", "wixsite.com", "wordpress.com",
    "blogspot.com", "netlify.app", "github.io", "glitch.me",
    "firebaseapp.com", "web.app", "surge.sh", "pages.dev",
}

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "buff.ly", "is.gd", "short.io", "rebrand.ly",
}

BRAND_KEYWORDS = [
    "paypal", "amazon", "apple", "google", "microsoft", "facebook",
    "instagram", "netflix", "dropbox", "linkedin", "twitter", "ebay",
    "wellsfargo", "chase", "citibank", "bankofamerica", "irs",
    "dhl", "fedex", "usps", "whatsapp", "telegram",
]

TYPO_PATTERNS: List[Tuple[str, str]] = [
    ("0", "o"), ("1", "l"), ("3", "e"), ("4", "a"), ("5", "s")
]

MULTI_PART_TLDS = {
    "co.uk", "co.in", "co.jp", "co.nz", "co.za", "com.au", "com.br",
    "com.cn", "com.mx", "net.au", "org.uk", "gov.uk", "co.id", "co.kr",
    "co.th", "com.ar", "com.co", "com.do", "com.eg", "com.hk", "com.my",
    "com.ng", "com.ph", "com.pk", "com.sa", "com.sg", "com.tr", "com.tw",
    "com.ua", "com.uy", "com.vn", "net.br", "net.in", "net.tr",
}


# ── Parsing Helpers ───────────────────────────────────────────────────────────
def shannon_entropy(s: str) -> float:
    """Calculates Shannon entropy of a string."""
    if not s:
        return 0.0
    probabilities = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probabilities)


def extract_tld_parts(url_or_domain: str) -> Tuple[str, str, str]:
    """
    Extract (subdomain, domain, suffix).
    Uses tldextract if available; falls back to manual parsing.
    """
    parsed_url = url_or_domain if "://" in url_or_domain else "http://" + url_or_domain
    
    if _HAS_TLDEXTRACT:
        extracted = tldextract.extract(parsed_url)
        return extracted.subdomain or "", extracted.domain or "", extracted.suffix or ""

    # Fallback parser
    parsed = urllib.parse.urlparse(parsed_url)
    hostname = parsed.hostname or ""
    parts = hostname.split(".")
    
    if len(parts) >= 3 and ".".join(parts[-2:]) in MULTI_PART_TLDS:
        suffix = ".".join(parts[-2:])
        domain = parts[-3] if len(parts) >= 3 else ""
        subdomain = ".".join(parts[:-3]) if len(parts) > 3 else ""
    elif len(parts) >= 2:
        suffix = parts[-1]
        domain = parts[-2]
        subdomain = ".".join(parts[:-2])
    else:
        suffix, domain, subdomain = "", hostname, ""
        
    return subdomain, domain, suffix


# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_canonical_url_features_dict(url: str) -> Dict[str, Any]:
    """
    Extracts dictionary of canonical 22 URL features plus metadata.
    Pure string parsing with zero network dependencies.
    """
    if not isinstance(url, str) or not url:
        return {name: 0.0 for name in CANONICAL_URL_FEATURE_NAMES}

    parsed_url = url if "://" in url else "http://" + url
    parsed = urllib.parse.urlparse(parsed_url)
    subdomain, domain, suffix = extract_tld_parts(parsed_url)
    hostname = parsed.hostname or ""
    full_host = hostname.lower()
    domain_lower = domain.lower()
    subdomain_lower = subdomain.lower()

    # 0. URL length
    url_len = len(url)

    # 1. Number of subdomains
    num_subdomains = len([s for s in subdomain.split(".") if s]) if subdomain else 0

    # 2. IP address in domain
    has_ip = 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) else 0

    # 3. HTTPS scheme
    is_https = 1 if url.lower().startswith("https") else 0

    # 4. Count of '@'
    count_at = url.count("@")

    # 5. Count of '-'
    count_dash = url.count("-")

    # 6. Count of '//' (excluding protocol)
    count_double_slash = max(0, url.count("//") - 1) if "://" in url else url.count("//")

    # 7. Domain entropy (domain + suffix token)
    domain_token = f"{domain}.{suffix}" if suffix else domain
    domain_entropy = shannon_entropy(domain_token)

    # 8. Query parameter count
    query_params = urllib.parse.parse_qsl(parsed.query)
    num_params = len(query_params)

    # 9. Suspicious query parameter names
    has_suspicious_params = int(any(
        any(susp in k.lower() for susp in SUSPICIOUS_QUERY_WORDS)
        for k, _ in query_params
    ))

    # 10. Brand keyword in domain (impersonation check)
    brand_in_domain = int(any(
        b in domain_lower and domain_lower != b
        for b in BRAND_KEYWORDS
    ))

    # 11. Brand keyword in subdomain
    brand_in_subdomain = int(any(b in subdomain_lower for b in BRAND_KEYWORDS))

    # 12. Typosquatting
    has_typosquat = int(any(
        any(b.replace(orig, rep) == domain_lower for orig, rep in TYPO_PATTERNS)
        for b in BRAND_KEYWORDS
    ))

    # 13. Punycode / IDN homograph
    has_punycode = int("xn--" in hostname.lower())

    # 14. Excessive subdomains (> 3 levels)
    excessive_subdomains = int(num_subdomains > 3)

    # 15. Suspicious TLD
    is_suspicious_tld = int(suffix.lower() in SUSPICIOUS_TLDS)

    # 16. Free hosting / subdomain abuse platform
    free_hosting = int(any(fh in full_host for fh in FREE_HOSTING_DOMAINS))

    # 17. URL shortener service
    is_shortener = int(any(s in full_host for s in URL_SHORTENERS))

    # 18. Excessive hyphens (>= 3 in domain)
    excessive_hyphens = int(domain.count("-") >= 3)

    # 19. Non-standard port
    port = parsed.port
    has_nonstandard_port = int(port is not None and port not in (80, 443, 8080))

    # 20. High entropy threshold (> 3.8)
    high_entropy = int(domain_entropy > 3.8)

    # 21. Long URL threshold (> 100 characters)
    long_url = int(url_len > 100)

    return {
        "url_len": url_len,
        "num_subdomains": num_subdomains,
        "has_ip": has_ip,
        "is_https": is_https,
        "count_at": count_at,
        "count_dash": count_dash,
        "count_double_slash": count_double_slash,
        "domain_entropy": domain_entropy,
        "num_params": num_params,
        "has_suspicious_params": has_suspicious_params,
        "brand_in_domain": brand_in_domain,
        "brand_in_subdomain": brand_in_subdomain,
        "has_typosquat": has_typosquat,
        "has_punycode": has_punycode,
        "excessive_subdomains": excessive_subdomains,
        "is_suspicious_tld": is_suspicious_tld,
        "free_hosting": free_hosting,
        "is_shortener": is_shortener,
        "excessive_hyphens": excessive_hyphens,
        "has_nonstandard_port": has_nonstandard_port,
        "high_entropy": high_entropy,
        "long_url": long_url,
        # Metadata fields for reasons / inspection
        "_domain": domain,
        "_subdomain": subdomain,
        "_suffix": suffix,
        "_hostname": hostname,
    }


def extract_canonical_url_features_vector(url: str) -> np.ndarray:
    """
    Extracts exactly 22 canonical URL features as a 2D numpy array shape (1, 22).
    Guarantees exact canonical feature order.
    """
    feat_dict = extract_canonical_url_features_dict(url)
    vector = [float(feat_dict[name]) for name in CANONICAL_URL_FEATURE_NAMES]
    return np.array(vector, dtype=np.float64).reshape(1, -1)
