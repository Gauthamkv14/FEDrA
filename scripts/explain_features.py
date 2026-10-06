"""
scripts/explain_features.py
===========================
Python Reference Implementation of Model-Faithful Feature Attribution (Step 7.1).
Computes exact linear logit decomposition for Logistic Regression baseline models.
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

def explain_linear_model(raw_vector: np.ndarray, bundle: dict, feature_names: list, descriptions: dict, modality_name: str) -> dict:
    scaler = bundle["scaler"]
    model = bundle["model"]
    
    assert raw_vector.shape[1] == scaler.n_features_in_, (
        f"Expected {scaler.n_features_in_} features, got {raw_vector.shape[1]}"
    )
    
    # 1. Scale feature vector exactly as model does
    scaled_vector = scaler.transform(raw_vector) # (1, D)
    
    # 2. Extract coefficients and intercept
    coefs = model.coef_[0] # (D,)
    intercept = float(model.intercept_[0])
    
    # 3. Compute exact contributions: c_i = w_i * z_i
    contributions = coefs * scaled_vector[0] # (D,)
    
    # 4. Decision score and reconstruction
    decision_score = float(model.decision_function(scaled_vector)[0])
    reconstructed_score = intercept + float(np.sum(contributions))
    reconstruction_error = abs(decision_score - reconstructed_score)
    
    prob_phishing = float(model.predict_proba(scaled_vector)[0][1])
    predicted_label = int(model.predict(scaled_vector)[0])
    prediction = "PHISHING" if predicted_label == 1 else "LEGITIMATE"
    
    feature_list = []
    for i, name in enumerate(feature_names):
        val = float(raw_vector[0, i])
        scaled_val = float(scaled_vector[0, i])
        c = float(contributions[i])
        direction = "phishing" if c > 0 else "legitimate"
        desc_obj = descriptions.get(name, {})
        desc = desc_obj.get(direction, f"{direction.capitalize()} indicator: {name}")
        
        feature_list.append({
            "name": name,
            "raw_value": val,
            "scaled_value": scaled_val,
            "coefficient": float(coefs[i]),
            "contribution": c,
            "abs_contribution": abs(c),
            "direction": direction,
            "description": desc
        })
        
    ranked_by_impact = sorted(feature_list, key=lambda x: x["abs_contribution"], reverse=True)
    top_phishing = sorted([f for f in feature_list if f["contribution"] > 0], key=lambda x: x["contribution"], reverse=True)
    top_legitimate = sorted([f for f in feature_list if f["contribution"] < 0], key=lambda x: x["contribution"])
    
    ranked_reasons = [{
        "feature": f["name"],
        "direction": f["direction"],
        "contribution": f["contribution"],
        "abs_contribution": f["abs_contribution"],
        "reason": f["description"]
    } for f in ranked_by_impact]
    
    return {
        "modality": modality_name,
        "prediction": prediction,
        "predicted_label": predicted_label,
        "phishing_probability": prob_phishing,
        "phishing_probability_pct": round(prob_phishing * 100, 2),
        "decision_score": decision_score,
        "intercept": intercept,
        "reconstructed_score": reconstructed_score,
        "reconstruction_error": reconstruction_error,
        "features": feature_list,
        "top_phishing_contributors": top_phishing,
        "top_legitimate_contributors": top_legitimate,
        "ranked_reasons": ranked_reasons
    }

def explain_url_features(raw_url_vector: np.ndarray) -> dict:
    url_bundle = joblib.load(os.path.join(BASE_DIR, "models", "url_baseline.pkl"))
    with open(os.path.join(BASE_DIR, "models", "onnx", "attribution_parameters.json"), "r", encoding="utf-8") as f:
        params = json.load(f)
    return explain_linear_model(
        raw_url_vector, url_bundle,
        params["url"]["feature_names"], params["url"]["descriptions"], "url"
    )

def explain_html_features(raw_html_vector: np.ndarray) -> dict:
    html_bundle = joblib.load(os.path.join(BASE_DIR, "models", "html_baseline.pkl"))
    with open(os.path.join(BASE_DIR, "models", "onnx", "attribution_parameters.json"), "r", encoding="utf-8") as f:
        params = json.load(f)
    return explain_linear_model(
        raw_html_vector, html_bundle,
        params["html"]["feature_names"], params["html"]["descriptions"], "html"
    )
