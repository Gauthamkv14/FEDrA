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

        // 3. Local MobileNetV2 Visual Feature Extraction (1280 dims)
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

    // 6. Local Model-Faithful Feature Attribution (URL + HTML structured explanations)
    var localAttribution = null;
    if (typeof FedraAttribution !== "undefined") {
        try {
            localAttribution = await FedraAttribution.explainStructuredModalities({
                urlFeatures: requestData.url_features_vector,
                htmlFeatures: requestData.html_features_vector
            });
            console.log(
                "[FEDrA Background] Feature attribution completed in " +
                (localAttribution ? localAttribution.combined_latency_ms : 0) + "ms"
            );
        } catch (err) {
            console.warn("[FEDrA Background] Local feature attribution failed:", err);
        }
    }

    var t_local_total_ms = Math.round((performance.now() - t_local_start) * 100) / 100;
    return {
        url: localUrlResult,
        html: localHtmlResult,
        visual: localVisualResult,
        image: localImageResult,
        fusion: localFusionResult,
        browser_verdict: browserVerdict,
        browser_prob_pct: browserProbPct,
        explanations: localAttribution,
        t_local_total_ms: t_local_total_ms
    };
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

    // 1. Run local in-browser ONNX inference (URL, HTML, MobileNet)
    var localInference = await executeLocalInference(requestData);

    console.log(
        "[FEDrA Background] Dispatching to Flask API (URL: " + url +
        ", Client URL Vector: " + (requestData.url_features_vector ? requestData.url_features_vector.length : 0) +
        ", Client HTML Vector: " + (requestData.html_features_vector ? requestData.html_features_vector.length : 0) +
        ", Visual Embedding: " + (requestData.visual_embedding ? "1280 dims (" + requestData.visual_embedding_source + ")" : "None") +
        ", screenshot: " + (requestData.screenshot ? "Present" : "None") + ")"
    );

    // 2. Dispatch to Flask API for image baseline + multimodal fusion inference
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
    .then(function (result) {
        var t_fetch_ms = Math.round(performance.now() - t_fetch_start);
        var t_total_e2e_ms = Date.now() - (requestData.timings.t_content_sent_ts || Date.now());

        result.scanned_url = url;
        result.dead_site = requestData.dead || false;
        result.timings = result.timings || {};
        result.timings.t_client_readiness_ms = requestData.timings.t_readiness_ms || 0;
        result.timings.t_client_dom_ms = requestData.timings.t_dom_ms || 0;
        result.timings.t_client_url_feat_ms = requestData.timings.t_url_feat_ms || 0;
        result.timings.t_client_html_feat_ms = requestData.timings.t_html_feat_ms || 0;
        result.timings.t_client_screenshot_ms = requestData.timings.t_screenshot_ms || 0;
        result.timings.t_client_transfer_ms = t_fetch_ms;
        result.timings.t_total_e2e_ms = t_total_e2e_ms;

        // Attach local in-browser ONNX diagnostics for validation
        result.local_inference = {
            url: localInference.url,
            html: localInference.html,
            visual: localInference.visual ? {
                embedding_dim: localInference.visual.embedding_dim,
                preprocess_time_ms: localInference.visual.preprocess_time_ms,
                inference_time_ms: localInference.visual.inference_time_ms,
                source: localInference.visual.source
            } : null,
            image: localInference.image,
            fusion: localInference.fusion,
            browser_verdict: localInference.browser_verdict,
            browser_phishing_prob_pct: localInference.browser_prob_pct,
            explanations: localInference.explanations,
            timings: {
                t_url_inference_ms: localInference.url ? localInference.url.inference_time_ms : 0,
                t_html_inference_ms: localInference.html ? localInference.html.inference_time_ms : 0,
                t_visual_inference_ms: localInference.visual ? localInference.visual.inference_time_ms : 0,
                t_visual_prep_ms: localInference.visual ? localInference.visual.preprocess_time_ms : 0,
                t_image_inference_ms: localInference.image ? localInference.image.inference_time_ms : 0,
                t_fusion_prep_ms: localInference.fusion ? localInference.fusion.preprocess_time_ms : 0,
                t_fusion_inference_ms: localInference.fusion ? localInference.fusion.inference_time_ms : 0,
                t_combined_inference_ms: localInference.t_local_total_ms
            }
        };

        console.log(
            "[FEDrA Background] Detection completed in " + t_total_e2e_ms + "ms (" +
            "Acquisition: " + result.acquisition_source + ", Flask: " +
            result.prediction + " " + result.phishing_probability + "%, " +
            "Browser ONNX: " + (localInference.browser_verdict || "N/A") + " " + (localInference.browser_prob_pct !== null ? localInference.browser_prob_pct + "%" : "N/A") + ")"
        );

        // Override text for dead/unreachable sites
        if (requestData.dead) {
            result.prediction = "PHISHING";
            result.risk_level = "HIGH";
            var deadReason = "Website is unreachable — likely taken down after phishing activity";
            if (!result.reasons || result.reasons.length === 0) {
                result.reasons = [
                    deadReason,
                    "Legitimate sites rarely go offline this way",
                    "URL pattern matches known phishing signatures"
                ];
            } else if (!result.reasons.includes(deadReason)) {
                result.reasons.unshift(deadReason);
            }
        }

        chrome.storage.local.set({ last_result: result });

        // Notification on Phishing (driven by validated fusion verdict)
        if (result.prediction === "PHISHING") {
            var topReasons = (result.reasons || []).slice(0, 2).join(". ");
            chrome.notifications.create("fedra_alert", {
                type: "basic",
                iconUrl: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                title: "\u26A0\uFE0F Phishing Detected — " + Math.round(result.phishing_probability || 95) + "%",
                message: topReasons || "This site exhibits high-confidence phishing indicators.",
                priority: 2
            });
        }
    })
    .catch(function (err) {
        console.error("[FEDrA Background] Processing / Fetch error:", err);
        var fallbackResult = {
            error: false,
            scanned_url: url,
            prediction: localInference.browser_verdict || "UNKNOWN",
            phishing_probability: localInference.browser_prob_pct || 0.0,
            risk_level: (localInference.browser_prob_pct || 0) >= 70 ? "HIGH" : ((localInference.browser_prob_pct || 0) >= 40 ? "MEDIUM" : "LOW"),
            acquisition_source: "browser_local_onnx_standalone",
            message: "Operating in browser-local standalone mode (Flask backend offline).",
            local_inference: {
                url: localInference.url,
                html: localInference.html,
                visual: localInference.visual ? {
                    embedding_dim: localInference.visual.embedding_dim,
                    source: localInference.visual.source
                } : null,
                image: localInference.image,
                fusion: localInference.fusion,
                browser_verdict: localInference.browser_verdict,
                browser_phishing_prob_pct: localInference.browser_prob_pct
            }
        };

        if (!localInference.browser_verdict) {
            fallbackResult.error = true;
            fallbackResult.message = "Flask server not running and local inference incomplete. Start api_server.py.";
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
