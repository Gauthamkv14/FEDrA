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
    
    raw_vector = np.asarray(raw_vector, dtype=np.float32)
    if raw_vector.ndim == 1:
        raw_vector = raw_vector[None, :]
    
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

def bilinear_interpolate_7x7_to_224x224(grid_7x7: np.ndarray) -> np.ndarray:
    """
    Exact bilinear interpolation from 7x7 grid to 224x224 heatmap (align_corners=False).
    Matches PyTorch F.interpolate(..., mode='bilinear', align_corners=False) to < 1e-6.
    Vectorized NumPy implementation.
    """
    ys = (np.arange(224, dtype=np.float32) + 0.5) / 32.0 - 0.5
    ys = np.clip(ys, 0.0, 6.0)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.minimum(y0 + 1, 6)
    wy = (ys - y0)[:, None]  # (224, 1)

    xs = (np.arange(224, dtype=np.float32) + 0.5) / 32.0 - 0.5
    xs = np.clip(xs, 0.0, 6.0)
    x0 = np.floor(xs).astype(np.int32)
    x1 = np.minimum(x0 + 1, 6)
    wx = (xs - x0)[None, :]  # (1, 224)

    # Gather 4 corners: (224, 224)
    v00 = grid_7x7[y0[:, None], x0[None, :]]
    v01 = grid_7x7[y0[:, None], x1[None, :]]
    v10 = grid_7x7[y1[:, None], x0[None, :]]
    v11 = grid_7x7[y1[:, None], x1[None, :]]

    out = (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v01
        + (1.0 - wx) * wy * v10
        + wx * wy * v11
    )
    return out.astype(np.float32)

def compute_visual_gradcam(spatial_features: np.ndarray, channel_weights: np.ndarray = None) -> dict:
    """
    Computes visual Grad-CAM heatmap from MobileNetV2 Layer 18 spatial activations (1280, 7, 7)
    and Image Baseline exact Grad-CAM channel weights: alpha_k = w_k / (49 * sigma_k).
    """
    if channel_weights is None:
        with open(os.path.join(BASE_DIR, "models", "onnx", "attribution_parameters.json"), "r", encoding="utf-8") as f:
            params = json.load(f)
        channel_weights = np.array(params["visual"]["gradcam_channel_weights"], dtype=np.float32)
    else:
        channel_weights = np.asarray(channel_weights, dtype=np.float32)

    if spatial_features.ndim == 4:
        spatial = spatial_features[0] # (1280, 7, 7)
    else:
        spatial = spatial_features

    assert spatial.shape == (1280, 7, 7), f"Expected (1280, 7, 7) spatial features, got {spatial.shape}"
    assert channel_weights.shape == (1280,), f"Expected (1280,) channel weights, got {channel_weights.shape}"

    # 1. Compute weighted combination over 1280 channels: L(i, j) = ReLU(sum_k alpha_k * A_{k, i, j})
    # spatial: (1280, 7, 7), channel_weights: (1280,)
    raw_linear = np.tensordot(channel_weights, spatial, axes=(0, 0)) # (7, 7)
    relu_grid = np.maximum(0.0, raw_linear) # (7, 7)

    # 2. Bilinear upsampling to (224, 224)
    upsampled_heatmap = bilinear_interpolate_7x7_to_224x224(relu_grid) # (224, 224)

    # 3. Min-Max Normalization to [0.0, 1.0]
    min_val = float(np.min(upsampled_heatmap))
    max_val = float(np.max(upsampled_heatmap))
    denom = max_val - min_val

    if denom > 1e-8:
        normalized_heatmap = (upsampled_heatmap - min_val) / denom
    else:
        normalized_heatmap = np.zeros_like(upsampled_heatmap)

    # 4. Extract peak regions from 7x7 grid
    peak_cells = []
    max_grid_val = float(np.max(relu_grid))
    if max_grid_val > 1e-8:
        norm_grid = (relu_grid - np.min(relu_grid)) / (max_grid_val - np.min(relu_grid))
        for gy in range(7):
            for gx in range(7):
                score = float(norm_grid[gy, gx])
                if score >= 0.5:
                    peak_cells.append({
                        "grid_y": gy,
                        "grid_x": gx,
                        "intensity": score,
                        "bbox_224": [gx * 32, gy * 32, (gx + 1) * 32, (gy + 1) * 32]
                    })
        peak_cells.sort(key=lambda x: x["intensity"], reverse=True)

    return {
        "modality": "visual",
        "target_layer": "mobilenet_v2.features.18",
        "spatial_grid_7x7": relu_grid.tolist(),
        "normalized_heatmap_224x224": normalized_heatmap.tolist(),
        "heatmap_min": min_val,
        "heatmap_max": max_val,
        "peak_regions": peak_cells[:5],
        "description": "The heatmap identifies spatial regions receiving high visual importance for the phishing-class prediction."
    }

def summarize_linear_modality(modality_expl: dict, top_n: int = 3) -> dict:
    if not modality_expl or not modality_expl.get("features"):
        return {
            "available": False,
            "modality": modality_expl.get("modality", "unknown") if modality_expl else "unknown",
            "error_code": "MODALITY_EXPLANATION_UNAVAILABLE",
            "description": "Explanation data for this modality is unavailable."
        }

    modality_name = modality_expl.get("modality", "unknown")
    features = modality_expl.get("features", [])

    # Deterministic top-N features by absolute attribution magnitude
    sorted_features = sorted(
        features,
        key=lambda f: f.get("abs_contribution", abs(f.get("contribution", 0.0))),
        reverse=True
    )

    top_features = []
    for f in sorted_features[:top_n]:
        c = float(f.get("contribution", 0.0))
        direction = "phishing" if c > 0 else ("legitimate" if c < 0 else "neutral")
        top_features.append({
            "feature": f.get("name", ""),
            "raw_value": f.get("raw_value", 0.0),
            "scaled_value": f.get("scaled_value", 0.0),
            "attribution": c,
            "abs_attribution": abs(c),
            "direction": direction,
            "description": f.get("description", "")
        })

    pos_attr = float(sum(f.get("contribution", 0.0) for f in features if f.get("contribution", 0.0) > 0))
    neg_attr = float(sum(f.get("contribution", 0.0) for f in features if f.get("contribution", 0.0) < 0))

    return {
        "available": True,
        "modality": modality_name,
        "prediction": modality_expl.get("prediction", "UNKNOWN"),
        "predicted_label": modality_expl.get("predicted_label", 0),
        "phishing_probability": modality_expl.get("phishing_probability", 0.0),
        "phishing_probability_pct": modality_expl.get("phishing_probability_pct", 0.0),
        "decision_score": modality_expl.get("decision_score", 0.0),
        "top_contributing_features": top_features,
        "top_features": top_features,
        "total_positive_attribution": pos_attr,
        "total_negative_attribution": neg_attr,
        "attribution_magnitude": abs(pos_attr) + abs(neg_attr)
    }

def summarize_visual_modality(visual_expl: dict, image_prediction: dict = None) -> dict:
    if not visual_expl:
        return {
            "available": False,
            "modality": "visual",
            "error_code": "VISUAL_EXPLANATION_UNAVAILABLE",
            "description": "Visual explanation data is unavailable."
        }

    grid = visual_expl.get("spatial_grid_7x7", [])
    active_count = 0
    if grid:
        for row in grid:
            for v in row:
                if v > 0:
                    active_count += 1

    heat_min = float(visual_expl.get("heatmap_min", 0.0))
    heat_max = float(visual_expl.get("heatmap_max", 0.0))
    heat_mean = float((heat_min + heat_max) / 2.0)

    pred_label = image_prediction.get("predicted_label", 1) if image_prediction else (1 if active_count > 0 else 0)
    pred_str = image_prediction.get("prediction", "PHISHING" if pred_label == 1 else "LEGITIMATE") if image_prediction else ("PHISHING" if pred_label == 1 else "LEGITIMATE")
    prob = image_prediction.get("phishing_probability", 0.5) if image_prediction else 0.5
    prob_pct = image_prediction.get("phishing_probability_pct", 50.0) if image_prediction else 50.0

    return {
        "available": True,
        "modality": "visual",
        "method": "Grad-CAM",
        "prediction": pred_str,
        "predicted_label": pred_label,
        "phishing_probability": prob,
        "phishing_probability_pct": prob_pct,
        "target_class": 1,
        "target_label": "phishing",
        "heatmap_width": 224,
        "heatmap_height": 224,
        "heatmap_min": heat_min,
        "heatmap_max": heat_max,
        "heatmap_mean": heat_mean,
        "peak_regions": visual_expl.get("peak_regions", []),
        "active_cells_count": active_count,
        "description": "The heatmap identifies spatial regions receiving high visual importance for the phishing-class prediction."
    }

def evaluate_cross_modal_agreement(url_summary: dict, html_summary: dict, visual_summary: dict) -> dict:
    evaluated = []
    phish_count = 0
    legit_count = 0

    if url_summary and url_summary.get("available"):
        evaluated.append("url")
        if url_summary.get("predicted_label") == 1 or url_summary.get("prediction") == "PHISHING":
            phish_count += 1
        else:
            legit_count += 1

    if html_summary and html_summary.get("available"):
        evaluated.append("html")
        if html_summary.get("predicted_label") == 1 or html_summary.get("prediction") == "PHISHING":
            phish_count += 1
        else:
            legit_count += 1

    if visual_summary and visual_summary.get("available"):
        evaluated.append("visual")
        if visual_summary.get("predicted_label") == 1 or visual_summary.get("prediction") == "PHISHING":
            phish_count += 1
        else:
            legit_count += 1

    total = len(evaluated)
    if total == 0:
        status = "UNAVAILABLE"
        summary_text = "No modality predictions available to assess cross-modal agreement."
    elif total == 3:
        if phish_count == 3:
            status = "ALL_PHISHING"
            summary_text = "Strong multimodal agreement: All 3 modalities indicate phishing risk."
        elif legit_count == 3:
            status = "ALL_LEGITIMATE"
            summary_text = "Strong multimodal agreement: All 3 modalities indicate legitimate page patterns."
        else:
            status = "MIXED"
            summary_text = f"Mixed multimodal evidence: {phish_count} modality/modalities indicate phishing and {legit_count} indicate legitimate."
    else:  # partial evaluation (1 or 2 available)
        if phish_count == total:
            status = "PARTIAL_PHISHING"
            summary_text = f"Partial multimodal evidence: All {total} available modality/modalities indicate phishing risk ({3 - total} unavailable)."
        elif legit_count == total:
            status = "PARTIAL_LEGITIMATE"
            summary_text = f"Partial multimodal evidence: All {total} available modality/modalities indicate legitimate page patterns ({3 - total} unavailable)."
        else:
            status = "PARTIAL_MIXED"
            summary_text = f"Partial mixed multimodal evidence: {phish_count} indicate phishing, {legit_count} indicate legitimate ({3 - total} unavailable)."

    return {
        "status": status,
        "phishing_modalities_count": phish_count,
        "legitimate_modalities_count": legit_count,
        "total_available_modalities": total,
        "modalities_evaluated": evaluated,
        "summary_text": summary_text
    }

def fuse_multimodal_explanations(
    url_explanation: dict = None,
    html_explanation: dict = None,
    visual_explanation: dict = None,
    image_prediction: dict = None,
    final_prediction: dict = None,
    top_n: int = 3
) -> dict:
    url_sum = summarize_linear_modality(url_explanation, top_n=top_n) if url_explanation else {
        "available": False, "modality": "url", "error_code": "URL_EXPLANATION_UNAVAILABLE", "description": "URL explanation unavailable."
    }
    html_sum = summarize_linear_modality(html_explanation, top_n=top_n) if html_explanation else {
        "available": False, "modality": "html", "error_code": "HTML_EXPLANATION_UNAVAILABLE", "description": "HTML explanation unavailable."
    }
    vis_sum = summarize_visual_modality(visual_explanation, image_prediction) if visual_explanation else {
        "available": False, "modality": "visual", "error_code": "VISUAL_EXPLANATION_UNAVAILABLE", "description": "Visual explanation unavailable."
    }

    agreement = evaluate_cross_modal_agreement(url_sum, html_sum, vis_sum)

    # Authoritative final verdict from fusion model
    final_pred_obj = None
    if final_prediction:
        final_cls = 1 if (
            final_prediction.get("class") == 1 or
            final_prediction.get("predicted_label") == 1 or
            final_prediction.get("label") == "PHISHING" or
            final_prediction.get("prediction") == "PHISHING"
        ) else 0
        final_lbl = final_prediction.get("label") or final_prediction.get("prediction") or ("PHISHING" if final_cls == 1 else "LEGITIMATE")
        
        final_pred_obj = {
            "class": final_cls,
            "label": final_lbl,
            "phishing_probability": float(final_prediction.get("phishing_probability", 0.0)),
            "phishing_probability_pct": float(final_prediction.get("phishing_probability_pct", 0.0)),
            "source": final_prediction.get("source", "fusion_model")
        }

    return {
        "schema_version": "1.0",
        "explanation_type": "multimodal_evidence_summary",
        "final_prediction": final_pred_obj,
        "cross_modal_agreement": agreement,
        "modalities": {
            "url": url_sum,
            "html": html_sum,
            "visual": vis_sum
        },
        "metadata": {
            "top_n_features": top_n,
            "fusion_weights": {
                "url": 0.4,
                "html": 0.3,
                "visual": 0.3
            },
            "disclaimer": "Multimodal explanation provides a structured summary of modality-specific evidence. Final decision is authoritative from the fusion classifier."
        }
    }


