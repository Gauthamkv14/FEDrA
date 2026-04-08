/**
 * content.js — FEDrA Chrome Extension
 * Waits 2 s, collects URL + HTML, detects error pages, sends to background.
 */

setTimeout(function () {
    var url = window.location.href;
    console.log("FEDrA content: page loaded", url);

    // Skip non-HTTP pages (chrome://, about:, etc.)
    if (!url.startsWith("http://") && !url.startsWith("https://")) return;

    var bodyText = document.body ? document.body.innerText : "";
    var title = document.title || "";

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

    var data = {
        url: url,
        html: dead ? "" : document.documentElement.outerHTML.substring(0, 30000),
        dead: dead,
    };
    console.log("FEDrA content: dead page?", data.dead);
    console.log("FEDrA content: sending message", data.url);

    try {
        chrome.runtime.sendMessage(data, function (response) {
            if (chrome.runtime.lastError) {
                console.log("FEDrA: message error", chrome.runtime.lastError.message);
            }
        });
    } catch (e) {
        console.log("FEDrA content error:", e);
    }
}, 2000);
