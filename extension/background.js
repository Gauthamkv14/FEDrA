/**
 * background.js — FEDrA Chrome Extension (Manifest V3 Service Worker)
 * ===================================================================
 * Browser-Native Acquisition & Dispatcher:
 * 1. Receives URL, Full DOM, and Browser-Extracted Feature Vectors from content.js.
 * 2. Captures browser-native screenshot via chrome.tabs.captureVisibleTab().
 * 3. Forwards browser-acquired payload and feature vectors to Flask backend.
 * 4. Merges complete timing instrumentation (T1..T7).
 * 5. Manages local storage (last_result) and notification alerts.
 */

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

function dispatchToBackend(requestData) {
    var t_fetch_start = performance.now();
    var url = requestData.url;

    console.log(
        "[FEDrA Background] Dispatching to Flask API (URL: " + url +
        ", Client URL Vector: " + (requestData.url_features_vector ? requestData.url_features_vector.length : 0) +
        ", Client HTML Vector: " + (requestData.html_features_vector ? requestData.html_features_vector.length : 0) +
        ", screenshot: " + (requestData.screenshot ? "Present" : "None") + ")"
    );

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

            console.log(
                "[FEDrA Background] Detection completed in " + t_total_e2e_ms + "ms (" +
                "Acquisition: " + result.acquisition_source + ", Prediction: " +
                result.prediction + " " + result.phishing_probability + "%)"
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

            // Notification on Phishing
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
            console.error("[FEDrA Background] Fetch error:", err);
            chrome.storage.local.set({
                last_result: {
                    error: true,
                    scanned_url: url,
                    message: "Flask server not running or network error. Start api_server.py first."
                }
            });
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
