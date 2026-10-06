/**
 * background.js — FEDrA Chrome Extension (Manifest V3 Service Worker)
 * ===================================================================
 * Browser-Native Acquisition & Hybrid Inference Dispatcher (Step 6.2):
 * 1. Receives URL, Full DOM, and Browser-Extracted Feature Vectors from content.js.
 * 2. Captures browser-native screenshot via chrome.tabs.captureVisibleTab().
 * 3. Runs local browser-side ONNX inference on:
 *    - URL (22) via url_baseline.onnx
 *    - HTML (12) via html_baseline.onnx
 *    - Screenshot via mobilenet_v2_visual.onnx -> 1280 visual embedding.
 * 4. Forwards browser-acquired payload, feature vectors, and 1280 visual embedding to Flask backend.
 * 5. Flask backend evaluates image baseline and multimodal fusion MLP (1314 dims).
 * 6. Merges complete timing instrumentation (T1..T7 + local ONNX timings).
 * 7. Manages local storage (last_result) and notification alerts.
 */

// Import ONNX Runtime Web, In-Browser Inference Engine, and Feature Attribution Engine
try {
    importScripts("libs/onnxruntime-web/ort.min.js", "inference.js", "attribution.js");
    console.log("[FEDrA Background] Successfully imported onnxruntime-web, inference.js, and attribution.js");
} catch (e) {
    console.warn("[FEDrA Background] importScripts failed or not in worker scope:", e);
}

var FLASK_API_URL = "http://localhost:5000/analyze";

/**
 * Safely capture the visible tab as a PNG data URL.
 */
function captureVisibleTabScreenshot(windowId, callback) {
    var t_screen_start = performance.now();
    try {
        var winTarget = typeof windowId === "number" ? windowId : null;
        chrome.tabs.captureVisibleTab(winTarget, { format: "png" }, function (dataUrl) {
            var t_screen_ms = Math.round(performance.now() - t_screen_start);
            if (chrome.runtime.lastError || !dataUrl) {
                console.log(
                    "[FEDrA Background] captureVisibleTab unavailable:",
                    chrome.runtime.lastError ? chrome.runtime.lastError.message : "No data URL returned"
                );
                callback(null, t_screen_ms);
            } else {
                console.log("[FEDrA Background] captureVisibleTab succeeded in " + t_screen_ms + "ms");
                callback(dataUrl, t_screen_ms);
            }
        });
    } catch (err) {
        var t_screen_err_ms = Math.round(performance.now() - t_screen_start);
        console.warn("[FEDrA Background] captureVisibleTab exception:", err);
        callback(null, t_screen_err_ms);
    }
}

/**
 * Execute local in-browser ONNX inference across all modalities:
 * - URL (22) via url_baseline.onnx
 * - HTML (12) via html_baseline.onnx
 * - Screenshot via mobilenet_v2_visual.onnx -> 1280 embedding
 * - Image Baseline (1280) via image_baseline.onnx
 * - Multimodal Fusion (1314) via fusion_model.onnx -> Browser-Native Final Verdict
 * Fails safely; returns structured results without throwing.
 */
async function executeLocalInference(requestData) {
    var localUrlResult = null;
    var localHtmlResult = null;
    var localVisualResult = null;
    var localImageResult = null;
    var localFusionResult = null;
    var browserVerdict = null;
    var browserProbPct = null;
    var t_local_start = performance.now();

    if (typeof FedraInference !== "undefined") {
        // 1. Local URL Inference (22 features)
        if (requestData.url_features_vector && requestData.url_features_vector.length === 22) {
            try {
                localUrlResult = await FedraInference.runUrlInference(requestData.url_features_vector);
                console.log(
                    "[FEDrA Background] Local URL ONNX prediction:",
                    localUrlResult.prediction,
                    "(" + localUrlResult.phishing_probability_pct + "%)",
                    "in " + localUrlResult.inference_time_ms + "ms"
                );
            } catch (err) {
                console.warn("[FEDrA Background] Local URL ONNX inference failed:", err);
            }
        }

        // 2. Local HTML Inference (12 features)
        if (requestData.html_features_vector && requestData.html_features_vector.length === 12) {
            try {
                localHtmlResult = await FedraInference.runHtmlInference(requestData.html_features_vector);
                console.log(
                    "[FEDrA Background] Local HTML ONNX prediction:",
                    localHtmlResult.prediction,
                    "(" + localHtmlResult.phishing_probability_pct + "%)",
                    "in " + localHtmlResult.inference_time_ms + "ms"
                );
            } catch (err) {
                console.warn("[FEDrA Background] Local HTML ONNX inference failed:", err);
            }
        }

        // 3. Local MobileNetV2 Visual Feature Extraction (1280 dims + 7x7 spatial maps)
        if (requestData.screenshot && !requestData.dead) {
            try {
                localVisualResult = await FedraInference.runVisualInference(requestData.screenshot);
                console.log(
                    "[FEDrA Background] Local MobileNetV2 ONNX visual extraction: 1280 dims in " +
                    localVisualResult.inference_time_ms + "ms (prep: " +
                    localVisualResult.preprocess_time_ms + "ms)"
                );
                requestData.visual_embedding = localVisualResult.visual_embedding;
                requestData.visual_embedding_source = "browser_onnx";
            } catch (err) {
                console.warn("[FEDrA Background] Local MobileNetV2 visual extraction failed, falling back to server:", err);
                requestData.visual_embedding = null;
                requestData.visual_embedding_source = "server_pytorch";
            }
        }

        // 4. Local Image Baseline Inference (1280 dims)
        var visualEmbedding = requestData.visual_embedding || (localVisualResult ? localVisualResult.visual_embedding : null);
        if (visualEmbedding && visualEmbedding.length === 1280) {
            try {
                localImageResult = await FedraInference.runImageBaselineInference(visualEmbedding);
                console.log(
                    "[FEDrA Background] Local Image Baseline ONNX prediction:",
                    localImageResult.prediction,
                    "(" + localImageResult.phishing_probability_pct + "%)",
                    "in " + localImageResult.inference_time_ms + "ms"
                );
            } catch (err) {
                console.warn("[FEDrA Background] Local Image Baseline ONNX inference failed:", err);
            }
        }

        // 5. Local Fusion MLP Inference (22 URL + 12 HTML + 1280 Visual -> 1314 dims)
        if (
            requestData.url_features_vector && requestData.url_features_vector.length === 22 &&
            requestData.html_features_vector && requestData.html_features_vector.length === 12 &&
            visualEmbedding && visualEmbedding.length === 1280
        ) {
            try {
                localFusionResult = await FedraInference.runFusionInference(
                    requestData.url_features_vector,
                    requestData.html_features_vector,
                    visualEmbedding
                );
                browserVerdict = localFusionResult.prediction;
                browserProbPct = localFusionResult.phishing_probability_pct;
                console.log(
                    "[FEDrA Background] Local Fusion MLP ONNX prediction:",
                    localFusionResult.prediction,
                    "(" + localFusionResult.phishing_probability_pct + "%)",
                    "in " + localFusionResult.inference_time_ms + "ms (prep: " +
                    localFusionResult.preprocess_time_ms + "ms)"
                );
            } catch (err) {
                console.warn("[FEDrA Background] Local Fusion MLP ONNX inference failed:", err);
            }
        } else if (localUrlResult) {
            // URL-only fallback if visual or HTML is unavailable
            browserVerdict = localUrlResult.prediction;
            browserProbPct = localUrlResult.phishing_probability_pct;
        }
    } else {
        console.warn("[FEDrA Background] FedraInference module unavailable for local inference.");
    }

    // 6. Formulate Authoritative Prediction Object
    var authoritativePrediction = null;
    if (localFusionResult) {
        authoritativePrediction = {
            class: localFusionResult.predicted_label,
            label: localFusionResult.prediction,
            phishing_probability: localFusionResult.phishing_probability,
            phishing_probability_pct: localFusionResult.phishing_probability_pct,
            source: "browser_local_fusion"
        };
    } else if (localUrlResult) {
        authoritativePrediction = {
            class: localUrlResult.predicted_label,
            label: localUrlResult.prediction,
            phishing_probability: localUrlResult.phishing_probability,
            phishing_probability_pct: localUrlResult.phishing_probability_pct,
            source: "browser_local_url_fallback"
        };
    }

    // 7. Local Multimodal Feature Attribution & Explainability (Downstream of Prediction)
    var localExplanation = null;
    var t_exp_start = performance.now();
    if (typeof FedraAttribution !== "undefined") {
        try {
            localExplanation = await FedraAttribution.explainMultimodalPipeline({
                urlFeatures: requestData.url_features_vector,
                htmlFeatures: requestData.html_features_vector,
                spatialFeatures: localVisualResult ? localVisualResult.spatial_features : null,
                imagePrediction: localImageResult,
                finalPrediction: authoritativePrediction,
                topN: 3
            });
            console.log(
                "[FEDrA Background] Multimodal explanation completed in " +
                (localExplanation && localExplanation.metadata ? localExplanation.metadata.explanation_latency_ms : 0) + "ms"
            );
        } catch (err) {
            console.warn("[FEDrA Background] Local multimodal explanation failed:", err);
        }
    }
    var t_exp_ms = Math.round((performance.now() - t_exp_start) * 100) / 100;

    var t_local_total_ms = Math.round((performance.now() - t_local_start) * 100) / 100;

    return {
        prediction: authoritativePrediction,
        modalities: {
            url: localUrlResult,
            html: localHtmlResult,
            visual: localVisualResult ? {
                embedding_dim: localVisualResult.embedding_dim,
                preprocess_time_ms: localVisualResult.preprocess_time_ms,
                inference_time_ms: localVisualResult.inference_time_ms,
                source: localVisualResult.source
            } : null,
            image: localImageResult,
            fusion: localFusionResult
        },
        explanation: localExplanation,
        timings: {
            t_url_inference_ms: localUrlResult ? localUrlResult.inference_time_ms : 0,
            t_html_inference_ms: localHtmlResult ? localHtmlResult.inference_time_ms : 0,
            t_visual_inference_ms: localVisualResult ? localVisualResult.inference_time_ms : 0,
            t_visual_prep_ms: localVisualResult ? localVisualResult.preprocess_time_ms : 0,
            t_image_inference_ms: localImageResult ? localImageResult.inference_time_ms : 0,
            t_fusion_prep_ms: localFusionResult ? localFusionResult.preprocess_time_ms : 0,
            t_fusion_inference_ms: localFusionResult ? localFusionResult.inference_time_ms : 0,
            t_explanation_ms: t_exp_ms,
            t_combined_inference_ms: t_local_total_ms
        },
        // Backwards compatibility aliases
        url: localUrlResult,
        html: localHtmlResult,
        visual: localVisualResult,
        image: localImageResult,
        fusion: localFusionResult,
        browser_verdict: browserVerdict,
        browser_prob_pct: browserProbPct,
        explanations: localExplanation,
        t_local_total_ms: t_local_total_ms
    };
}

/**
 * Extract human-readable reasons from the multimodal explanation object.
 */
function extractReasonsFromExplanation(explanation) {
    var reasons = [];
    if (!explanation || !explanation.modalities) return reasons;

    if (explanation.modalities.url && explanation.modalities.url.top_contributing_features) {
        explanation.modalities.url.top_contributing_features.forEach(function (f) {
            if (f.direction === "phishing" && f.description) {
                reasons.push(f.description);
            }
        });
    }

    if (explanation.modalities.html && explanation.modalities.html.top_contributing_features) {
        explanation.modalities.html.top_contributing_features.forEach(function (f) {
            if (f.direction === "phishing" && f.description) {
                reasons.push(f.description);
            }
        });
    }

    if (explanation.modalities.visual && explanation.modalities.visual.available &&
        (explanation.modalities.visual.prediction === "PHISHING" || explanation.modalities.visual.predicted_label === 1)) {
        if (explanation.modalities.visual.peak_regions && explanation.modalities.visual.peak_regions.length > 0) {
            var peak = explanation.modalities.visual.peak_regions[0];
            var gy = peak.grid_y !== undefined ? peak.grid_y : peak.grid_row;
            var gx = peak.grid_x !== undefined ? peak.grid_x : peak.grid_col;
            reasons.push("High visual activation localized in layout grid region (row " + gy + ", col " + gx + ").");
        }
    }

    return reasons;
}

chrome.runtime.onMessage.addListener(function (msg, sender, sendResponse) {
    console.log("[FEDrA Background] Received acquisition message:", msg.url, "(dead: " + msg.dead + ")");
    var url = msg.url || "";

    // Skip non-HTTP internal schemes
    if (
        url.startsWith("chrome://") ||
        url.startsWith("chrome-extension://") ||
        url.startsWith("about:") ||
        url.startsWith("devtools://") ||
        url.startsWith("file://")
    ) {
        return;
    }

    sendResponse({ status: "received" });

    var t_bg_received_ts = Date.now();
    var windowId = sender && sender.tab ? sender.tab.windowId : null;

    if (msg.dead) {
        // Site failed in-browser navigation
        dispatchToBackend({
            url: url,
            html: "",
            screenshot: null,
            dead: true,
            url_features_vector: msg.url_features_vector || null,
            html_features_vector: null,
            url_features_dict: msg.url_features_dict || null,
            html_features_dict: null,
            visual_embedding: null,
            visual_embedding_source: "none",
            timings: {
                t_readiness_ms: msg.t_readiness_ms || 0,
                t_dom_ms: msg.t_dom_ms || 0,
                t_url_feat_ms: msg.t_url_feat_ms || 0,
                t_html_feat_ms: msg.t_html_feat_ms || 0,
                t_screenshot_ms: 0,
                t_content_sent_ts: msg.t_content_sent_ts || t_bg_received_ts,
                t_bg_received_ts: t_bg_received_ts
            }
        });
    } else {
        // Live page: capture browser-native screenshot then dispatch
        captureVisibleTabScreenshot(windowId, function (screenshotDataUrl, t_screenshot_ms) {
            dispatchToBackend({
                url: url,
                html: msg.html || "",
                screenshot: screenshotDataUrl || null,
                dead: false,
                url_features_vector: msg.url_features_vector || null,
                html_features_vector: msg.html_features_vector || null,
                url_features_dict: msg.url_features_dict || null,
                html_features_dict: msg.html_features_dict || null,
                visual_embedding: null,
                visual_embedding_source: "none",
                timings: {
                    t_readiness_ms: msg.t_readiness_ms || 0,
                    t_dom_ms: msg.t_dom_ms || 0,
                    t_url_feat_ms: msg.t_url_feat_ms || 0,
                    t_html_feat_ms: msg.t_html_feat_ms || 0,
                    t_screenshot_ms: t_screenshot_ms || 0,
                    t_content_sent_ts: msg.t_content_sent_ts || t_bg_received_ts,
                    t_bg_received_ts: t_bg_received_ts
                }
            });
        });
    }

    return true;
});

async function dispatchToBackend(requestData) {
    var t_fetch_start = performance.now();
    var url = requestData.url;

    // 1. Run authoritative local in-browser ONNX inference and explainability
    var localInference = await executeLocalInference(requestData);

    console.log(
        "[FEDrA Background] Dispatching to Flask API (URL: " + url +
        ", Client URL Vector: " + (requestData.url_features_vector ? requestData.url_features_vector.length : 0) +
        ", Client HTML Vector: " + (requestData.html_features_vector ? requestData.html_features_vector.length : 0) +
        ", Visual Embedding: " + (requestData.visual_embedding ? "1280 dims (" + requestData.visual_embedding_source + ")" : "None") +
        ", screenshot: " + (requestData.screenshot ? "Present" : "None") + ")"
    );

    var reasons = extractReasonsFromExplanation(localInference.explanation);

    // 2. Dispatch to Flask API for verification/telemetry (Non-blocking / parallel)
    fetch(FLASK_API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestData)
    })
    .then(function (res) {
        if (!res.ok) {
            throw new Error("HTTP error " + res.status);
        }
        return res.json();
    })
    .then(function (flaskResult) {
        var t_fetch_ms = Math.round(performance.now() - t_fetch_start);
        var t_total_e2e_ms = Date.now() - (requestData.timings.t_content_sent_ts || Date.now());

        var finalVerdict = localInference.browser_verdict || flaskResult.prediction || "UNKNOWN";
        var finalProbPct = localInference.browser_prob_pct !== null ? localInference.browser_prob_pct : (flaskResult.phishing_probability || 0.0);
        var riskLevel = finalProbPct >= 70 ? "HIGH" : (finalProbPct >= 40 ? "MEDIUM" : "LOW");

        var authoritativePred = localInference.prediction;
        var acquisitionSource = "browser_local_onnx_verified";
        if (!authoritativePred && flaskResult.prediction) {
            authoritativePred = {
                class: flaskResult.prediction === "PHISHING" ? 1 : 0,
                label: flaskResult.prediction,
                phishing_probability: typeof flaskResult.phishing_probability === "number" ? flaskResult.phishing_probability / 100.0 : 0.0,
                phishing_probability_pct: flaskResult.phishing_probability || 0.0,
                source: "flask_server_fallback"
            };
            acquisitionSource = "flask_server_fallback";
        }

        var combinedResult = {
            error: false,
            scanned_url: url,
            dead_site: requestData.dead || false,
            prediction: finalVerdict,
            phishing_probability: finalProbPct,
            risk_level: riskLevel,
            authoritative_prediction: authoritativePred,
            modalities: localInference.modalities,
            explanation: localInference.explanation,
            reasons: reasons.length > 0 ? reasons : (flaskResult.reasons || []),
            acquisition_source: acquisitionSource,
            flask_validation: {
                prediction: flaskResult.prediction,
                phishing_probability: flaskResult.phishing_probability,
                status: "validated"
            },
            local_inference: localInference,
            timings: {
                t_client_readiness_ms: requestData.timings.t_readiness_ms || 0,
                t_client_dom_ms: requestData.timings.t_dom_ms || 0,
                t_client_url_feat_ms: requestData.timings.t_url_feat_ms || 0,
                t_client_html_feat_ms: requestData.timings.t_html_feat_ms || 0,
                t_client_screenshot_ms: requestData.timings.t_screenshot_ms || 0,
                t_client_transfer_ms: t_fetch_ms,
                t_url_inference_ms: localInference.timings.t_url_inference_ms,
                t_html_inference_ms: localInference.timings.t_html_inference_ms,
                t_visual_prep_ms: localInference.timings.t_visual_prep_ms,
                t_visual_inference_ms: localInference.timings.t_visual_inference_ms,
                t_image_inference_ms: localInference.timings.t_image_inference_ms,
                t_fusion_prep_ms: localInference.timings.t_fusion_prep_ms,
                t_fusion_inference_ms: localInference.timings.t_fusion_inference_ms,
                t_explanation_ms: localInference.timings.t_explanation_ms,
                t_combined_local_ms: localInference.t_local_total_ms,
                t_total_e2e_ms: t_total_e2e_ms
            }
        };

        console.log(
            "[FEDrA Background] Detection completed in " + t_total_e2e_ms + "ms (" +
            "Local Verdict: " + combinedResult.prediction + " " + combinedResult.phishing_probability + "%, " +
            "Flask Validation: " + flaskResult.prediction + " " + flaskResult.phishing_probability + "%)"
        );

        // Override text for dead/unreachable sites
        if (requestData.dead) {
            combinedResult.prediction = "PHISHING";
            combinedResult.risk_level = "HIGH";
            var deadReason = "Target web host is unreachable or failed network connection.";
            if (!combinedResult.reasons || combinedResult.reasons.length === 0) {
                combinedResult.reasons = [
                    deadReason,
                    "Web server failed connection attempt during page acquisition.",
                    "URL lexical features evaluated via fallback baseline model."
                ];
            } else if (!combinedResult.reasons.includes(deadReason)) {
                combinedResult.reasons.unshift(deadReason);
            }
        }

        chrome.storage.local.set({ last_result: combinedResult });

        // Notification on Phishing (driven by authoritative browser verdict)
        if (combinedResult.prediction === "PHISHING") {
            var topReasons = (combinedResult.reasons || []).slice(0, 2).join(". ");
            chrome.notifications.create("fedra_alert", {
                type: "basic",
                iconUrl: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                title: "\u26A0\uFE0F Phishing Detected — " + Math.round(combinedResult.phishing_probability || 95) + "%",
                message: topReasons || "This site exhibits high-confidence phishing indicators.",
                priority: 2
            });
        }
    })
    .catch(function (err) {
        console.warn("[FEDrA Background] Flask backend offline or error, operating standalone:", err);
        var finalVerdict = localInference.browser_verdict || "UNKNOWN";
        var finalProbPct = localInference.browser_prob_pct !== null ? localInference.browser_prob_pct : 0.0;
        var riskLevel = finalProbPct >= 70 ? "HIGH" : (finalProbPct >= 40 ? "MEDIUM" : "LOW");

        var fallbackResult = {
            error: false,
            scanned_url: url,
            dead_site: requestData.dead || false,
            prediction: finalVerdict,
            phishing_probability: finalProbPct,
            risk_level: riskLevel,
            authoritative_prediction: localInference.prediction,
            modalities: localInference.modalities,
            explanation: localInference.explanation,
            reasons: reasons,
            acquisition_source: localInference.browser_verdict ? "browser_local_onnx_standalone" : "none",
            message: localInference.browser_verdict ? "Operating in browser-local standalone mode (Flask backend offline)." : "Flask server not running and local inference incomplete. Start api_server.py.",
            local_inference: localInference,
            timings: {
                t_client_readiness_ms: requestData.timings.t_readiness_ms || 0,
                t_client_dom_ms: requestData.timings.t_dom_ms || 0,
                t_client_url_feat_ms: requestData.timings.t_url_feat_ms || 0,
                t_client_html_feat_ms: requestData.timings.t_html_feat_ms || 0,
                t_client_screenshot_ms: requestData.timings.t_screenshot_ms || 0,
                t_url_inference_ms: localInference.timings.t_url_inference_ms,
                t_html_inference_ms: localInference.timings.t_html_inference_ms,
                t_visual_prep_ms: localInference.timings.t_visual_prep_ms,
                t_visual_inference_ms: localInference.timings.t_visual_inference_ms,
                t_image_inference_ms: localInference.timings.t_image_inference_ms,
                t_fusion_prep_ms: localInference.timings.t_fusion_prep_ms,
                t_fusion_inference_ms: localInference.timings.t_fusion_inference_ms,
                t_explanation_ms: localInference.timings.t_explanation_ms,
                t_combined_local_ms: localInference.t_local_total_ms
            }
        };

        if (!localInference.browser_verdict) {
            fallbackResult.error = true;
            fallbackResult.authoritative_prediction = null;
            fallbackResult.prediction = "UNKNOWN";
        }

        chrome.storage.local.set({ last_result: fallbackResult });
    });
}

// ── Fallback for navigation failure events (dead / unreachable site) ────────
chrome.webNavigation.onErrorOccurred.addListener(function (details) {
    if (details.frameId !== 0) return; // Main frame only

    var url = details.url || "";
    if (
        url.startsWith("chrome://") ||
        url.startsWith("chrome-extension://") ||
        url.startsWith("about:") ||
        url.startsWith("devtools://") ||
        url.startsWith("file://")
    ) {
        return;
    }

    console.log("[FEDrA Background] webNavigation onErrorOccurred:", url, details.error);

    var t_start = Date.now();
    dispatchToBackend({
        url: url,
        html: "",
        screenshot: null,
        dead: true,
        url_features_vector: null,
        html_features_vector: null,
        url_features_dict: null,
        html_features_dict: null,
        visual_embedding: null,
        visual_embedding_source: "none",
        timings: {
            t_readiness_ms: 0,
            t_dom_ms: 0,
            t_url_feat_ms: 0,
            t_html_feat_ms: 0,
            t_screenshot_ms: 0,
            t_content_sent_ts: t_start,
            t_bg_received_ts: t_start
        }
    });
});
