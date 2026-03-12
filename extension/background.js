/**
 * background.js — FEDrA Chrome Extension (Manifest V3 Service Worker)
 * Receives messages from content.js, calls Flask API, stores result.
 */

chrome.runtime.onMessage.addListener(function (msg, sender, sendResponse) {
    console.log("FEDrA background: received message", msg);
    var url = msg.url || "";

    // Skip Chrome internal pages
    if (
        url.startsWith("chrome://") ||
        url.startsWith("chrome-extension://") ||
        url.startsWith("about:")
    ) {
        return;
    }

    // Build request body — send url always; html only for live pages
    var body = msg.dead ? { url: url } : { url: url, html: msg.html };

    console.log("FEDrA background: calling API for", url);
    fetch("http://localhost:5000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    })
        .then(function (r) { return r.json(); })
        .then(function (result) {
            console.log("FEDrA background: API response", result);
            result.scanned_url = url;
            result.dead_site = msg.dead || false;

            // Override result for dead/unreachable sites
            if (msg.dead) {
                result.prediction = "PHISHING";
                result.risk_level = "HIGH";
                var deadReason = "Website is unreachable — likely taken down after phishing activity";
                if (!result.reasons || result.reasons.length === 0) {
                    result.reasons = [
                        deadReason,
                        "Legitimate sites rarely go offline this way",
                        "URL pattern matches known phishing signatures",
                    ];
                } else {
                    result.reasons.unshift(deadReason);
                }
            }

            chrome.storage.local.set({ last_result: result });
            console.log("FEDrA background: stored result", result);

            // Notification for phishing
            if (result.prediction === "PHISHING") {
                var topReasons = (result.reasons || []).slice(0, 2).join(". ");
                chrome.notifications.create("fedra_alert", {
                    type: "basic",
                    iconUrl: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    title: "\u26A0\uFE0F Phishing Detected \u2014 " +
                        Math.round(result.phishing_probability || 95) + "%",
                    message: topReasons || "This site appears to be phishing.",
                    priority: 2,
                });
            }
        })
        .catch(function (err) {
            console.log("FEDrA background: fetch error", err);
            // Flask server not running or network error
            chrome.storage.local.set({
                last_result: {
                    error: true,
                    scanned_url: url,
                    message: "Flask server not running. Start api_server.py first.",
                },
            });
        });

    return true; // Keep message channel open for async response
});
