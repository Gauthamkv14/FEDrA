"""
scripts/export_attribution_params.py
====================================
Exports exact model coefficients, intercepts, scalers, feature names,
and human-readable description templates for URL and HTML baseline models
to models/onnx/attribution_parameters.json and extension/models/attribution_parameters.json.
"""

import os
import sys
import json
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from url_features import CANONICAL_URL_FEATURE_NAMES

URL_DESCRIPTIONS = {
    "url_len": {
        "phishing": "Elevated URL character length relative to baseline.",
        "legitimate": "Standard URL character length relative to baseline."
    },
    "num_subdomains": {
        "phishing": "Subdomain depth relative to baseline.",
        "legitimate": "Low subdomain depth consistent with standard domain structure."
    },
    "has_ip": {
        "phishing": "Hostname uses raw numeric IP address format instead of a domain name.",
        "legitimate": "Hostname uses standard domain name formatting."
    },
    "is_https": {
        "phishing": "Unencrypted HTTP protocol in use.",
        "legitimate": "HTTPS encryption protocol in use."
    },
    "count_at": {
        "phishing": "Presence of '@' symbol in URL string.",
        "legitimate": "No '@' symbol in URL string."
    },
    "count_dash": {
        "phishing": "Elevated hyphen count in URL string.",
        "legitimate": "Low hyphen count in URL string."
    },
    "count_double_slash": {
        "phishing": "Presence of additional '//' sequence in URL path.",
        "legitimate": "No additional '//' sequence in URL path."
    },
    "domain_entropy": {
        "phishing": "Elevated domain character entropy relative to baseline.",
        "legitimate": "Standard domain character entropy relative to baseline."
    },
    "num_params": {
        "phishing": "Elevated number of query parameters.",
        "legitimate": "Low query parameter count."
    },
    "has_suspicious_params": {
        "phishing": "Query string contains targeted authentication-related parameter keywords.",
        "legitimate": "No targeted authentication keywords in query parameters."
    },
    "brand_in_domain": {
        "phishing": "Recognized brand keyword present in domain name.",
        "legitimate": "No recognized brand keyword match in domain name."
    },
    "brand_in_subdomain": {
        "phishing": "Recognized brand keyword present in subdomain prefix.",
        "legitimate": "No recognized brand keyword match in subdomain prefix."
    },
    "has_typosquat": {
        "phishing": "Domain contains edit-distance match to a recognized brand name.",
        "legitimate": "Domain spelling does not match known brand edit-distance patterns."
    },
    "has_punycode": {
        "phishing": "Internationalized domain name encoding ('xn--' Punycode) in use.",
        "legitimate": "Standard ASCII character encoding in domain name."
    },
    "excessive_subdomains": {
        "phishing": "Subdomain depth exceeds 3 levels.",
        "legitimate": "Subdomain depth is 3 levels or fewer."
    },
    "is_suspicious_tld": {
        "phishing": "Top-Level Domain (TLD) matches list of higher-risk extensions.",
        "legitimate": "Top-Level Domain (TLD) is not in the higher-risk list."
    },
    "free_hosting": {
        "phishing": "Domain hosted on a recognized free or shared web hosting domain.",
        "legitimate": "Domain not hosted on known free or shared web hosting domains."
    },
    "is_shortener": {
        "phishing": "URL uses a recognized link shortening service.",
        "legitimate": "URL does not use a known link shortening service."
    },
    "excessive_hyphens": {
        "phishing": "Domain contains more than 3 hyphens.",
        "legitimate": "Domain contains 3 or fewer hyphens."
    },
    "has_nonstandard_port": {
        "phishing": "Non-standard network port specified in URL.",
        "legitimate": "Standard web protocol port used."
    },
    "high_entropy": {
        "phishing": "Domain character entropy exceeds the 3.5 threshold.",
        "legitimate": "Domain character entropy does not exceed the 3.5 threshold."
    },
    "long_url": {
        "phishing": "URL length exceeds 75 characters.",
        "legitimate": "URL length is 75 characters or fewer."
    }
}

HTML_FEATURE_NAMES = [
    "num_forms", "num_inputs", "num_iframes", "num_ext_links", "num_ext_scripts",
    "has_password_field", "has_meta_redirect", "script_content_ratio",
    "favicon_mismatch", "has_auto_submit", "input_submit_ratio", "num_unique_ext_domains"
]

HTML_DESCRIPTIONS = {
    "num_forms": {
        "phishing": "Form element count relative to baseline.",
        "legitimate": "Form element count consistent with baseline."
    },
    "num_inputs": {
        "phishing": "Input field count relative to baseline.",
        "legitimate": "Input field count consistent with baseline."
    },
    "num_iframes": {
        "phishing": "Presence of embedded iframe elements in page markup.",
        "legitimate": "Zero or expected iframe element count in page markup."
    },
    "num_ext_links": {
        "phishing": "External domain hyperlink count relative to baseline.",
        "legitimate": "External domain hyperlink count consistent with baseline."
    },
    "num_ext_scripts": {
        "phishing": "External script source count relative to baseline.",
        "legitimate": "External script source count consistent with baseline."
    },
    "has_password_field": {
        "phishing": "Password input field present in DOM.",
        "legitimate": "No password input field present in DOM."
    },
    "has_meta_redirect": {
        "phishing": "Meta-refresh tag present in DOM header.",
        "legitimate": "No meta-refresh tag present in DOM header."
    },
    "script_content_ratio": {
        "phishing": "Elevated inline script-to-markup ratio.",
        "legitimate": "Standard inline script-to-markup ratio."
    },
    "favicon_mismatch": {
        "phishing": "Favicon loaded from external domain differing from page host.",
        "legitimate": "Favicon loaded from same origin as page host."
    },
    "has_auto_submit": {
        "phishing": "Automated form submit script pattern detected in DOM.",
        "legitimate": "No automated form submit script pattern detected in DOM."
    },
    "input_submit_ratio": {
        "phishing": "Input-to-submit button ratio relative to baseline.",
        "legitimate": "Input-to-submit button ratio consistent with baseline."
    },
    "num_unique_ext_domains": {
        "phishing": "Unique external domain reference count relative to baseline.",
        "legitimate": "Unique external domain reference count consistent with baseline."
    }
}

def export_params():
    url_bundle = joblib.load(os.path.join(BASE_DIR, "models", "url_baseline.pkl"))
    html_bundle = joblib.load(os.path.join(BASE_DIR, "models", "html_baseline.pkl"))

    url_scaler = url_bundle["scaler"]
    url_model = url_bundle["model"]

    html_scaler = html_bundle["scaler"]
    html_model = html_bundle["model"]

    params = {
        "schema_version": "v1",
        "url": {
            "feature_names": CANONICAL_URL_FEATURE_NAMES,
            "feature_dim": 22,
            "intercept": float(url_model.intercept_[0]),
            "coefficients": url_model.coef_[0].tolist(),
            "scaler": {
                "mean": url_scaler.mean_.tolist(),
                "scale": url_scaler.scale_.tolist()
            },
            "descriptions": URL_DESCRIPTIONS
        },
        "html": {
            "feature_names": HTML_FEATURE_NAMES,
            "feature_dim": 12,
            "intercept": float(html_model.intercept_[0]),
            "coefficients": html_model.coef_[0].tolist(),
            "scaler": {
                "mean": html_scaler.mean_.tolist(),
                "scale": html_scaler.scale_.tolist()
            },
            "descriptions": HTML_DESCRIPTIONS
        }
    }

    # Save to models/onnx/attribution_parameters.json
    out_dir1 = os.path.join(BASE_DIR, "models", "onnx")
    os.makedirs(out_dir1, exist_ok=True)
    out_path1 = os.path.join(out_dir1, "attribution_parameters.json")
    with open(out_path1, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"[OK] Exported {out_path1}")

    # Save to extension/models/attribution_parameters.json
    out_dir2 = os.path.join(BASE_DIR, "extension", "models")
    os.makedirs(out_dir2, exist_ok=True)
    out_path2 = os.path.join(out_dir2, "attribution_parameters.json")
    with open(out_path2, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"[OK] Exported {out_path2}")

if __name__ == "__main__":
    export_params()
