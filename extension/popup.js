/**
 * popup.js — FEDrA Chrome Extension
 * Reads last_result from storage, matches against active tab, renders state.
 * No inline scripts. No onclick= attributes. Fully CSP-compliant.
 */

document.addEventListener("DOMContentLoaded", function () {

    function renderScanning() {
        return '<div class="state scanning">' +
            '<div class="scan-icon">&#128269;</div>' +
            '<div class="scan-text">Scanning page...</div>' +
            '<div class="scan-sub">Results appear in 5&#8211;15 seconds</div>' +
            '</div>';
    }

    function renderOffline(msg) {
        return '<div class="state offline">' +
            '<div class="icon">&#128274;</div>' +
            '<div class="title">Server Offline</div>' +
            '<div class="sub">' + (msg || "Start api_server.py first.") + '</div>' +
            '</div>';
    }

    function renderDead(r) {
        var reasons = (r.reasons || [])
            .map(function (reason) { return '<li>&#9888;&#65039; ' + reason + '</li>'; })
            .join('');
        return '<div class="state dead">' +
            '<div class="icon">&#128128;</div>' +
            '<div class="title">DANGEROUS SITE</div>' +
            '<div class="subtitle">This site has been taken down</div>' +
            '<div class="url-display">' + (r.scanned_url || r.url || '').substring(0, 55) + '</div>' +
            '<div class="warning-text">' +
            'Phishing sites are frequently taken offline after being reported. ' +
            'This URL matches phishing patterns.' +
            '</div>' +
            (reasons ? '<div class="reasons-title">Why it was flagged:</div><ul class="reasons">' + reasons + '</ul>' : '') +
            '<button id="btn-escape" class="btn-escape">&#8592; Return to Safety</button>' +
            '</div>';
    }

    function renderPhishing(r) {
        var prob = r.phishing_probability || 0;
        var reasons = (r.reasons || [])
            .map(function (reason) { return '<li>&#9888;&#65039; ' + reason + '</li>'; })
            .join('');
        var riskColor = r.risk_level === 'HIGH' ? '#ff3333' :
            r.risk_level === 'MEDIUM' ? '#ff9900' : '#ffcc00';
        return '<div class="state phishing">' +
            '<div class="icon">&#128680;</div>' +
            '<div class="title">PHISHING DETECTED</div>' +
            '<div class="url-display">' + (r.scanned_url || r.url || '').substring(0, 55) + '</div>' +
            '<div class="confidence-label">Confidence: ' + prob.toFixed(1) + '%</div>' +
            '<div class="bar-bg"><div class="bar-fill" style="width:' + Math.min(prob, 100) + '%;background:#ff3333;"></div></div>' +
            '<span class="badge" style="background:' + riskColor + '">' + (r.risk_level || 'HIGH') + ' RISK</span>' +
            (reasons ? '<div class="reasons-title">Why it was flagged:</div><ul class="reasons">' + reasons + '</ul>' : '') +
            '<div class="meta">Engine: ' + (r.mode || 'Fusion MLP') + ' &nbsp;&middot;&nbsp; Latency: ' + (r.latency_s || 0) + 's</div>' +
            '</div>';
    }

    function renderLegit(r) {
        var prob = 100 - (r.phishing_probability || 0);
        return '<div class="state legit">' +
            '<div class="icon">&#9989;</div>' +
            '<div class="title">SAFE</div>' +
            '<div class="url-display">' + (r.scanned_url || r.url || '').substring(0, 55) + '</div>' +
            '<div class="confidence-label">Safety score: ' + prob.toFixed(1) + '%</div>' +
            '<div class="bar-bg"><div class="bar-fill" style="width:' + Math.min(prob, 100) + '%;background:#00cc66;"></div></div>' +
            '<div class="meta">Engine: ' + (r.mode || 'Fusion MLP') + ' &nbsp;&middot;&nbsp; Latency: ' + (r.latency_s || 0) + 's</div>' +
            '</div>';
    }

    function attachEscapeButton() {
        var btn = document.getElementById('btn-escape');
        if (!btn) return;
        btn.addEventListener('click', function () {
            chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
                if (tabs && tabs[0]) {
                    chrome.tabs.update(tabs[0].id, { url: 'https://www.google.com' });
                }
            });
        });
    }

    function safeHost(url) {
        try { return new URL(url).hostname; }
        catch (e) { return ''; }
    }

    // ── Main: get active tab then read storage ────────────────────────────
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        var currentUrl = (tabs && tabs[0]) ? (tabs[0].url || '') : '';
        console.log("FEDrA popup: current tab", currentUrl);

        // If current tab is a chrome error page, show last result anyway
        var isErrorPage = currentUrl.startsWith("chrome-error://") ||
            currentUrl.startsWith("chrome://") ||
            currentUrl === "";

        var container = document.getElementById('container');

        chrome.storage.local.get('last_result', function (data) {
            var result = data.last_result;
            console.log("FEDrA popup: stored result", result);

            // No result yet
            if (!result) {
                container.innerHTML = renderScanning();
                return;
            }

            // Server offline
            if (result.error) {
                container.innerHTML = renderOffline(result.message);
                return;
            }

            // For error pages, just show whatever last result we have
            if (isErrorPage) {
                renderResult(result, container);
                return;
            }

            // Normal pages — match hostname
            var resultHost = safeHost(result.scanned_url || result.url || '');
            var currentHost = safeHost(currentUrl);

            console.log("FEDrA popup: resultHost", resultHost);
            console.log("FEDrA popup: currentHost", currentHost);
            console.log("FEDrA popup: hosts match?", resultHost === currentHost);

            if (resultHost === currentHost) {
                renderResult(result, container);
            } else {
                container.innerHTML = renderScanning();
            }
        });
    });

    // Single render function that picks correct state
    function renderResult(result, container) {
        if (result.dead_site) {
            container.innerHTML = renderDead(result);
            var btn = document.getElementById("btn-escape");
            if (btn) {
                btn.addEventListener("click", function () {
                    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
                        if (tabs && tabs[0]) chrome.tabs.update(tabs[0].id, { url: "https://www.google.com" });
                    });
                });
            }
        } else if (result.prediction === "PHISHING") {
            container.innerHTML = renderPhishing(result);
        } else {
            container.innerHTML = renderLegit(result);
        }
    }

    // Auto-refresh every 3 seconds by reloading the popup document
    setTimeout(function () { location.reload(); }, 3000);
});
