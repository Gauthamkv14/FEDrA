/**
 * content.js — FEDrA Chrome Extension (Manifest V3 Content Script)
 * ================================================================
 * Browser-Native Page Acquisition & Feature Extraction:
 * 1. Executes at "document_idle".
 * 2. Implements dynamic DOM readiness (~250ms stabilization delay).
 * 3. Extracts canonical 22 URL features directly in-browser (url_features.js).
 * 4. Extracts canonical 12 HTML features directly from live document DOM (html_features.js).
 * 5. Captures full document outerHTML up to 2MB safety threshold.
 * 6. Measures granular diagnostic timings (T1 readiness, T2 DOM capture, T_url_feat, T_html_feat).
 * 7. Dispatches structured payload to background service worker.
 */

(function () {
    var startTime = performance.now();
    var url = window.location.href;

    // Skip non-HTTP(S) schemes (chrome://, chrome-extension://, about:, devtools://, file://)
    if (!url.startsWith("http://") && !url.startsWith("https://")) {
        return;
    }

    console.log("[FEDrA Content] Initialized on:", url);

    function acquireAndExtract() {
        var readinessTime = performance.now();
        var t_readiness_ms = Math.round(readinessTime - startTime);

        var bodyText = document.body ? document.body.innerText : "";
        var title = document.title || "";

        // Check for browser network / navigation error signatures
        var dead =
            title.includes("ERR_") ||
            title.includes("can't be reached") ||
            title.includes("not available") ||
            title.includes("Security error") ||
            title.includes("Dangerous") ||
            bodyText.includes("DNS_PROBE") ||
            bodyText.includes("ERR_NAME_NOT_RESOLVED") ||
            bodyText.includes("This site can't be reached") ||
            bodyText.includes("Attackers on the site");

        var domStart = performance.now();
        var rawHtml = "";
        var htmlTruncated = false;

        if (!dead && document.documentElement) {
            rawHtml = document.documentElement.outerHTML || "";
            var MAX_DOM_LENGTH = 2000000;
            if (rawHtml.length > MAX_DOM_LENGTH) {
                console.warn("[FEDrA Content] HTML exceeds 2MB limit (" + rawHtml.length + " chars), truncating safely.");
                rawHtml = rawHtml.substring(0, MAX_DOM_LENGTH);
                htmlTruncated = true;
            }
        }
        var domEnd = performance.now();
        var t_dom_ms = Math.round(domEnd - domStart);

        // ── 1. Browser-Native URL Feature Extraction ──────────────────────
        var t_url_start = performance.now();
        var url_vector = null;
        var url_dict = null;
        if (typeof FEDrA_URL !== "undefined") {
            url_vector = FEDrA_URL.extractCanonicalUrlFeaturesVector(url);
            url_dict = FEDrA_URL.extractCanonicalUrlFeaturesDict(url);
        }
        var t_url_feat_ms = Math.round((performance.now() - t_url_start) * 100) / 100;

        // ── 2. Browser-Native HTML Feature Extraction from Live DOM ───────
        var t_html_start = performance.now();
        var html_vector = null;
        var html_dict = null;
        if (!dead && typeof FEDrA_HTML !== "undefined") {
            html_vector = FEDrA_HTML.extractHtmlFeaturesVector(document, url);
            html_dict = FEDrA_HTML.extractHtmlFeaturesFromDoc(document, url, rawHtml.length);
        }
        var t_html_feat_ms = Math.round((performance.now() - t_html_start) * 100) / 100;

        var payload = {
            url: url,
            html: rawHtml,
            html_length: rawHtml.length,
            html_truncated: htmlTruncated,
            dead: dead,
            url_features_vector: url_vector,
            url_features_dict: url_dict,
            html_features_vector: html_vector,
            html_features_dict: html_dict,
            t_readiness_ms: t_readiness_ms,
            t_dom_ms: t_dom_ms,
            t_url_feat_ms: t_url_feat_ms,
            t_html_feat_ms: t_html_feat_ms,
            t_content_sent_ts: Date.now()
        };

        console.log("[FEDrA Content] In-browser extraction complete: URL feat: " +
                    t_url_feat_ms + "ms (dim: " + (url_vector ? url_vector.length : 0) +
                    "), HTML feat: " + t_html_feat_ms + "ms (dim: " + (html_vector ? html_vector.length : 0) +
                    "), DOM capture: " + t_dom_ms + "ms");

        try {
            chrome.runtime.sendMessage(payload, function (response) {
                if (chrome.runtime.lastError) {
                    console.warn("[FEDrA Content] Message delivery error:", chrome.runtime.lastError.message);
                } else if (response && response.status === "received") {
                    console.log("[FEDrA Content] Background acknowledged payload");
                }
            });
        } catch (err) {
            console.error("[FEDrA Content] Exception sending payload to background:", err);
        }
    }

    // Dynamic Readiness Strategy:
    // If document is complete, wait 250ms stabilization delay.
    // If loading / interactive, wait for window load event (or max 1000ms safety timeout).
    if (document.readyState === "complete") {
        setTimeout(acquireAndExtract, 250);
    } else {
        var executed = false;
        function onReady() {
            if (executed) return;
            executed = true;
            setTimeout(acquireAndExtract, 250);
        }

        window.addEventListener("load", onReady, { once: true });
        setTimeout(onReady, 1000);
    }
})();
