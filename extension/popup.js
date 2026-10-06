/**
 * popup.js — FEDrA Chrome Extension (Explainability UI Renderer)
 * 
 * Strictly a rendering layer:
 * - Zero prediction logic / Zero ONNX inference
 * - Zero feature extraction / Zero Grad-CAM recomputation
 * - Zero external network calls / CDNs / telemetry
 * - Full CSP compliance and safe DOM rendering
 */

(function () {
    var container = null;
    var engineBadge = null;
    var lastRenderedKey = null;

    /**
     * Escape text for safe rendering
     */
    function sanitize(str) {
        if (str === null || str === undefined) return "";
        return String(str);
    }

    /**
     * Extract hostname safely
     */
    function safeHost(url) {
        try {
            return new URL(url).hostname;
        } catch (e) {
            return "";
        }
    }

    /**
     * Ensure DOM element references are initialized
     */
    function initRefs() {
        if (!container) container = document.getElementById("container");
        if (!engineBadge) engineBadge = document.getElementById("engine-badge");
    }

    /**
     * Render the Scanning State
     */
    function renderScanning(url) {
        initRefs();
        if (!container) return;
        container.innerHTML = "";
        var card = document.createElement("div");
        card.className = "state-card";

        var icon = document.createElement("div");
        icon.className = "scan-pulse";
        icon.textContent = "🔍";

        var title = document.createElement("div");
        title.className = "scan-title";
        title.textContent = "Analyzing page...";

        var sub = document.createElement("div");
        sub.className = "scan-sub";
        sub.textContent = url ? "Evaluating " + sanitize(url).substring(0, 45) + "..." : "Multimodal inference in progress";

        card.appendChild(icon);
        card.appendChild(title);
        card.appendChild(sub);
        container.appendChild(card);

        if (engineBadge) {
            engineBadge.textContent = "Scanning...";
            engineBadge.style.color = "var(--info)";
        }
    }

    /**
     * Render the Offline / Server State
     */
    function renderOffline(msg) {
        initRefs();
        if (!container) return;
        container.innerHTML = "";
        var card = document.createElement("div");
        card.className = "state-card";

        var icon = document.createElement("div");
        icon.className = "verdict-icon";
        icon.textContent = "🔒";

        var title = document.createElement("div");
        title.className = "verdict-title inconclusive";
        title.textContent = "Detection Unavailable";

        var sub = document.createElement("div");
        sub.className = "scan-sub";
        sub.textContent = sanitize(msg || "No analysis data recorded yet.");

        card.appendChild(icon);
        card.appendChild(title);
        card.appendChild(sub);
        container.appendChild(card);

        if (engineBadge) {
            engineBadge.textContent = "Standby";
            engineBadge.style.color = "var(--warn)";
        }
    }

    /**
     * Build Verdict Card
     */
    function createVerdictCard(r) {
        var card = document.createElement("div");
        var isPhish = r.prediction === "PHISHING";
        var isLegit = r.prediction === "LEGITIMATE";
        var isDead = !!r.dead_site;
        var prob = typeof r.phishing_probability === "number" ? r.phishing_probability : 0.0;
        var riskLevel = r.risk_level || (prob >= 70 ? "HIGH" : (prob >= 40 ? "MEDIUM" : "LOW"));

        if (isDead) {
            card.className = "verdict-card dead";
        } else if (isPhish) {
            card.className = "verdict-card phishing";
        } else if (isLegit) {
            card.className = "verdict-card legit";
        } else {
            card.className = "verdict-card inconclusive";
        }

        // Icon
        var icon = document.createElement("div");
        icon.className = "verdict-icon";
        icon.textContent = isDead ? "💀" : (isPhish ? "🚨" : (isLegit ? "🛡️" : "⚠️"));

        // Title
        var title = document.createElement("div");
        var titleClass = isDead ? "dead" : (isPhish ? "phishing" : (isLegit ? "legit" : "inconclusive"));
        title.className = "verdict-title " + titleClass;
        title.textContent = isDead ? "DANGEROUS SITE (OFFLINE)" : (isPhish ? "PHISHING DETECTED" : (isLegit ? "AUTHENTIC / LEGITIMATE" : "INCONCLUSIVE / PARTIAL"));

        // Scanned URL Box
        var urlBox = document.createElement("div");
        urlBox.className = "url-box";
        var displayUrl = sanitize(r.scanned_url || r.url || "");
        urlBox.textContent = displayUrl.length > 55 ? displayUrl.substring(0, 52) + "..." : displayUrl;
        urlBox.title = displayUrl;

        // Metric Row (Confidence / Safety Score)
        var metricRow = document.createElement("div");
        metricRow.className = "metric-row";
        var metricLabel = document.createElement("span");
        metricLabel.textContent = isLegit ? "Safety Confidence:" : "Phishing Probability:";
        var metricVal = document.createElement("span");
        metricVal.className = "metric-value";
        var displayPct = isLegit ? (100 - prob).toFixed(1) : prob.toFixed(1);
        metricVal.textContent = displayPct + "%";
        metricRow.appendChild(metricLabel);
        metricRow.appendChild(metricVal);

        // Progress Track & Bar
        var track = document.createElement("div");
        track.className = "progress-track";
        var bar = document.createElement("div");
        var barFillClass = isPhish || isDead ? "phishing" : (isLegit ? "legit" : "warn");
        bar.className = "progress-bar " + barFillClass;
        var fillWidth = isLegit ? Math.min(100, Math.max(0, 100 - prob)) : Math.min(100, Math.max(0, prob));
        bar.style.width = fillWidth + "%";
        track.appendChild(bar);

        // Risk Pill Badge
        var pill = document.createElement("span");
        var pillClass = riskLevel === "HIGH" || isDead ? "high" : (riskLevel === "MEDIUM" ? "medium" : "low");
        pill.className = "pill-badge " + pillClass;
        pill.textContent = (isDead ? "HIGH" : riskLevel) + " RISK";

        card.appendChild(icon);
        card.appendChild(title);
        card.appendChild(urlBox);
        card.appendChild(metricRow);
        card.appendChild(track);
        card.appendChild(pill);

        // Escape Button for dangerous or phishing sites
        if (isPhish || isDead) {
            var escapeBtn = document.createElement("button");
            escapeBtn.className = "btn-escape";
            escapeBtn.id = "btn-escape";
            escapeBtn.textContent = "← Return to Safety";
            escapeBtn.addEventListener("click", function () {
                if (typeof chrome !== "undefined" && chrome.tabs && chrome.tabs.query) {
                    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
                        if (tabs && tabs[0]) {
                            chrome.tabs.update(tabs[0].id, { url: "https://www.google.com" });
                        }
                    });
                } else {
                    window.location.href = "https://www.google.com";
                }
            });
            card.appendChild(escapeBtn);
        }

        return card;
    }

    /**
     * Build Cross-Modal Agreement Banner
     */
    function createAgreementCard(explanation) {
        var card = document.createElement("div");
        card.className = "agreement-card";

        var header = document.createElement("div");
        header.className = "agreement-header";

        var title = document.createElement("span");
        title.className = "agreement-title";
        title.textContent = "Cross-Modal Agreement";

        var chip = document.createElement("span");
        var cma = explanation && explanation.cross_modal_agreement ? explanation.cross_modal_agreement : null;
        var status = cma ? cma.status : "UNAVAILABLE";

        var chipClass = "unavailable";
        if (status === "ALL_PHISHING") chipClass = "consensus-phish";
        else if (status === "ALL_LEGITIMATE") chipClass = "consensus-legit";
        else if (status === "MIXED") chipClass = "mixed";
        else if (status && status.startsWith("PARTIAL")) chipClass = "partial";

        chip.className = "agreement-chip " + chipClass;
        chip.textContent = status.replace(/_/g, " ");

        header.appendChild(title);
        header.appendChild(chip);

        var desc = document.createElement("div");
        desc.className = "agreement-desc";
        desc.textContent = cma && cma.summary_text
            ? cma.summary_text
            : "Multimodal agreement signals evaluated across available modalities.";

        card.appendChild(header);
        card.appendChild(desc);
        return card;
    }

    /**
     * Helper to create a feature list item
     */
    function createFeatureItem(featureName, direction, description) {
        var item = document.createElement("div");
        item.className = "feature-item";

        var meta = document.createElement("div");
        meta.className = "feature-meta";

        var key = document.createElement("span");
        key.className = "feature-key";
        key.textContent = sanitize(featureName);

        var dirTag = document.createElement("span");
        var isPhish = direction === "phishing";
        dirTag.className = "feature-dir-tag " + (isPhish ? "phish" : "legit");
        dirTag.textContent = isPhish ? "+ Risk Indicator" : "- Authentic Signal";

        meta.appendChild(key);
        meta.appendChild(dirTag);

        var desc = document.createElement("div");
        desc.className = "feature-desc";
        desc.textContent = sanitize(description || "Feature contribution evaluated via linear attribution.");

        item.appendChild(meta);
        item.appendChild(desc);
        return item;
    }

    /**
     * Build URL Modality Card
     */
    function createUrlModalityCard(explanation, modalities) {
        var card = document.createElement("div");
        card.className = "modality-card";

        var urlExpl = explanation && explanation.modalities ? explanation.modalities.url : null;
        var urlMod = modalities && modalities.url ? modalities.url : null;
        var isAvail = urlExpl && urlExpl.available;

        var header = document.createElement("div");
        header.className = "modality-card-header";

        var name = document.createElement("div");
        name.className = "modality-name";
        name.innerHTML = "<span>🔗</span><span>URL Lexical Structure</span>";

        var tag = document.createElement("span");
        if (isAvail) {
            var isPhish = urlExpl.prediction === "PHISHING" || urlExpl.predicted_label === 1;
            tag.className = "modality-verdict-tag " + (isPhish ? "phish" : "legit");
            var prob = urlExpl.phishing_probability_pct !== undefined ? urlExpl.phishing_probability_pct : (urlMod ? urlMod.phishing_probability_pct : 0);
            tag.textContent = (isPhish ? "PHISHING " : "LEGIT ") + Number(prob).toFixed(1) + "%";
        } else {
            tag.className = "modality-verdict-tag na";
            tag.textContent = "UNAVAILABLE";
        }

        header.appendChild(name);
        header.appendChild(tag);
        card.appendChild(header);

        if (isAvail) {
            var topFeats = urlExpl.top_features || urlExpl.top_contributing_features || [];
            if (topFeats.length > 0) {
                topFeats.slice(0, 3).forEach(function (f) {
                    var fName = f.feature || f.name || "Feature";
                    var fDir = f.direction || (f.attribution > 0 ? "phishing" : "legitimate");
                    var fDesc = f.description || f.reason || "";
                    card.appendChild(createFeatureItem(fName, fDir, fDesc));
                });
            } else {
                var emptyMsg = document.createElement("div");
                emptyMsg.className = "visual-meta";
                emptyMsg.textContent = "URL baseline features evaluated without strong individual outliers.";
                card.appendChild(emptyMsg);
            }
        } else {
            var naMsg = document.createElement("div");
            naMsg.className = "visual-meta";
            naMsg.textContent = "URL lexical modality was not available for analysis.";
            card.appendChild(naMsg);
        }

        return card;
    }

    /**
     * Build HTML / DOM Modality Card
     */
    function createHtmlModalityCard(explanation, modalities) {
        var card = document.createElement("div");
        card.className = "modality-card";

        var htmlExpl = explanation && explanation.modalities ? explanation.modalities.html : null;
        var htmlMod = modalities && modalities.html ? modalities.html : null;
        var isAvail = htmlExpl && htmlExpl.available;

        var header = document.createElement("div");
        header.className = "modality-card-header";

        var name = document.createElement("div");
        name.className = "modality-name";
        name.innerHTML = "<span>📄</span><span>Page Content (DOM)</span>";

        var tag = document.createElement("span");
        if (isAvail) {
            var isPhish = htmlExpl.prediction === "PHISHING" || htmlExpl.predicted_label === 1;
            tag.className = "modality-verdict-tag " + (isPhish ? "phish" : "legit");
            var prob = htmlExpl.phishing_probability_pct !== undefined ? htmlExpl.phishing_probability_pct : (htmlMod ? htmlMod.phishing_probability_pct : 0);
            tag.textContent = (isPhish ? "PHISHING " : "LEGIT ") + Number(prob).toFixed(1) + "%";
        } else {
            tag.className = "modality-verdict-tag na";
            tag.textContent = "UNAVAILABLE";
        }

        header.appendChild(name);
        header.appendChild(tag);
        card.appendChild(header);

        if (isAvail) {
            var topFeats = htmlExpl.top_features || htmlExpl.top_contributing_features || [];
            if (topFeats.length > 0) {
                topFeats.slice(0, 3).forEach(function (f) {
                    var fName = f.feature || f.name || "Feature";
                    var fDir = f.direction || (f.attribution > 0 ? "phishing" : "legitimate");
                    var fDesc = f.description || f.reason || "";
                    card.appendChild(createFeatureItem(fName, fDir, fDesc));
                });
            } else {
                var emptyMsg = document.createElement("div");
                emptyMsg.className = "visual-meta";
                emptyMsg.textContent = "HTML DOM baseline evaluated with baseline nominal weights.";
                card.appendChild(emptyMsg);
            }
        } else {
            var naMsg = document.createElement("div");
            naMsg.className = "visual-meta";
            naMsg.textContent = "HTML DOM modality was not available for analysis.";
            card.appendChild(naMsg);
        }

        return card;
    }

    /**
     * Build Visual Grad-CAM Modality Card
     */
    function createVisualModalityCard(explanation, modalities) {
        var card = document.createElement("div");
        card.className = "modality-card";

        var visExpl = explanation && explanation.modalities ? explanation.modalities.visual : null;
        var imgMod = modalities && modalities.image ? modalities.image : null;
        var isAvail = visExpl && visExpl.available;

        var header = document.createElement("div");
        header.className = "modality-card-header";

        var name = document.createElement("div");
        name.className = "modality-name";
        name.innerHTML = "<span>👁️</span><span>Visual Layout (Grad-CAM)</span>";

        var tag = document.createElement("span");
        if (isAvail) {
            var isPhish = visExpl.prediction === "PHISHING" || visExpl.predicted_label === 1;
            tag.className = "modality-verdict-tag " + (isPhish ? "phish" : "legit");
            var prob = visExpl.phishing_probability_pct !== undefined ? visExpl.phishing_probability_pct : (imgMod ? imgMod.phishing_probability_pct : 0);
            tag.textContent = (isPhish ? "PHISHING " : "LEGIT ") + Number(prob).toFixed(1) + "%";
        } else {
            tag.className = "modality-verdict-tag na";
            tag.textContent = "UNAVAILABLE";
        }

        header.appendChild(name);
        header.appendChild(tag);
        card.appendChild(header);

        if (isAvail) {
            var visContainer = document.createElement("div");
            visContainer.className = "visual-container";

            // 7x7 Grad-CAM Grid
            var grid = document.createElement("div");
            grid.className = "gradcam-grid";
            grid.title = "MobileNetV2 Layer 18 Spatial Activation Heatmap (7x7)";

            var peakSet = new Set();
            var peakRegions = visExpl.peak_regions || [];
            peakRegions.forEach(function (p) {
                var gy = p.grid_y !== undefined ? p.grid_y : p.grid_row;
                var gx = p.grid_x !== undefined ? p.grid_x : p.grid_col;
                if (gy !== undefined && gx !== undefined) {
                    peakSet.add(gy + "_" + gx);
                }
            });

            var spatial2D = visExpl.spatial_grid_7x7 || [];
            for (var r = 0; r < 7; r++) {
                for (var c = 0; c < 7; c++) {
                    var cell = document.createElement("div");
                    cell.className = "gradcam-cell";
                    var isPeak = peakSet.has(r + "_" + c);
                    if (isPeak) {
                        cell.classList.add("peak");
                        cell.style.background = "#ff334b";
                    } else if (spatial2D[r] && spatial2D[r][c] > 0) {
                        var val = spatial2D[r][c];
                        cell.style.background = "rgba(255, 100, 50, " + Math.min(1.0, 0.2 + val * 0.8) + ")";
                    } else {
                        cell.style.background = "rgba(255, 255, 255, 0.03)";
                    }
                    grid.appendChild(cell);
                }
            }

            var visDetails = document.createElement("div");
            visDetails.className = "visual-details";

            var visDesc = document.createElement("div");
            visDesc.className = "visual-text";
            visDesc.textContent = visExpl.description || "Spatial attention heatmap indicating visual feature activation.";

            var visMeta = document.createElement("div");
            visMeta.className = "visual-meta";
            var peakStr = peakRegions.length > 0
                ? "Peak focus: Row " + (peakRegions[0].grid_y !== undefined ? peakRegions[0].grid_y : peakRegions[0].grid_row) +
                  ", Col " + (peakRegions[0].grid_x !== undefined ? peakRegions[0].grid_x : peakRegions[0].grid_col) +
                  " (intensity: " + (peakRegions[0].intensity || 1.0) + ")"
                : "Active cells: " + (visExpl.active_cells_count || 0) + " / 49";
            visMeta.textContent = peakStr;

            visDetails.appendChild(visDesc);
            visDetails.appendChild(visMeta);

            visContainer.appendChild(grid);
            visContainer.appendChild(visDetails);
            card.appendChild(visContainer);
        } else {
            var naMsg = document.createElement("div");
            naMsg.className = "visual-meta";
            naMsg.textContent = "Visual layout modality was not evaluated for this page.";
            card.appendChild(naMsg);
        }

        return card;
    }

    /**
     * Build Technical Diagnostics Accordion
     */
    function createDiagnosticsAccordion(r) {
        var details = document.createElement("details");
        details.className = "diag-accordion";

        var summary = document.createElement("summary");
        summary.className = "diag-summary";
        summary.textContent = "⚡ Technical & Inference Diagnostics";

        var content = document.createElement("div");
        content.className = "diag-content";

        var timings = r.timings || {};
        var mods = r.modalities || {};
        var expl = r.explanation || {};

        function addRow(label, value) {
            var row = document.createElement("div");
            row.className = "diag-row";
            var l = document.createElement("span");
            l.textContent = label;
            var v = document.createElement("span");
            v.textContent = sanitize(value);
            row.appendChild(l);
            row.appendChild(v);
            content.appendChild(row);
        }

        addRow("Decision Source", r.acquisition_source || (r.authoritative_prediction ? r.authoritative_prediction.source : "local_fusion"));
        addRow("Explanation Schema", expl.schema_version || "1.0");
        addRow("URL Baseline Inference", (timings.t_url_inference_ms || 0) + " ms");
        addRow("HTML Baseline Inference", (timings.t_html_inference_ms || 0) + " ms");
        addRow("MobileNetV2 Inference", (timings.t_visual_inference_ms || 0) + " ms");
        addRow("Fusion MLP Inference", (timings.t_fusion_inference_ms || 0) + " ms");
        addRow("Explanation Synthesis", (timings.t_explanation_ms || 0) + " ms");
        addRow("Total Local Latency", (timings.t_combined_local_ms || 0) + " ms");

        details.appendChild(summary);
        details.appendChild(content);
        return details;
    }

    /**
     * Master Render Function
     */
    function renderResult(result) {
        initRefs();
        if (!container) return;
        container.innerHTML = "";

        // Header engine badge state
        if (engineBadge) {
            var source = result.acquisition_source || "";
            if (source.includes("fallback")) {
                engineBadge.textContent = "Server Fallback";
                engineBadge.style.color = "var(--warn)";
            } else {
                engineBadge.textContent = "ONNX Local";
                engineBadge.style.color = "var(--info)";
            }
        }

        // 1. Verdict Card
        var verdictCard = createVerdictCard(result);
        container.appendChild(verdictCard);

        // 2. Cross-Modal Agreement Banner
        if (result.explanation) {
            var agreementCard = createAgreementCard(result.explanation);
            container.appendChild(agreementCard);
        }

        // 3. Modality Breakdown Heading
        var heading = document.createElement("div");
        heading.className = "section-heading";
        heading.textContent = "Observational Modality Evidence";
        container.appendChild(heading);

        // Modality Cards List
        var list = document.createElement("div");
        list.className = "modality-list";

        var urlCard = createUrlModalityCard(result.explanation, result.modalities);
        var htmlCard = createHtmlModalityCard(result.explanation, result.modalities);
        var visualCard = createVisualModalityCard(result.explanation, result.modalities);

        list.appendChild(urlCard);
        list.appendChild(htmlCard);
        list.appendChild(visualCard);
        container.appendChild(list);

        // 4. Technical Diagnostics Accordion
        var diagAccordion = createDiagnosticsAccordion(result);
        container.appendChild(diagAccordion);

        // 5. Footer
        var footer = document.createElement("div");
        footer.className = "footer";
        footer.textContent = "FEDrA Client-Side Explainable AI Engine";
        container.appendChild(footer);
    }

    /**
     * Poll / Check storage and active tab
     */
    function updatePopup() {
        initRefs();
        if (typeof chrome === "undefined" || !chrome.tabs || !chrome.tabs.query || !chrome.storage || !chrome.storage.local) {
            return;
        }

        chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
            var currentUrl = (tabs && tabs[0]) ? (tabs[0].url || "") : "";
            var isErrorPage = currentUrl.startsWith("chrome-error://") ||
                              currentUrl.startsWith("chrome://") ||
                              currentUrl.startsWith("about:") ||
                              currentUrl === "";

            chrome.storage.local.get("last_result", function (data) {
                var result = data ? data.last_result : null;

                if (!result) {
                    renderScanning(currentUrl);
                    return;
                }

                if (result.error && !result.prediction) {
                    renderOffline(result.message);
                    return;
                }

                if (isErrorPage) {
                    var key = JSON.stringify(result);
                    if (key !== lastRenderedKey) {
                        lastRenderedKey = key;
                        renderResult(result);
                    }
                    return;
                }

                var resultHost = safeHost(result.scanned_url || result.url || "");
                var currentHost = safeHost(currentUrl);

                if (resultHost && currentHost && resultHost === currentHost) {
                    var newKey = (result.scanned_url || "") + "_" + (result.prediction || "") + "_" + (result.phishing_probability || 0);
                    if (newKey !== lastRenderedKey) {
                        lastRenderedKey = newKey;
                        renderResult(result);
                    }
                } else if (!resultHost || !currentHost) {
                    var keyFallback = JSON.stringify(result);
                    if (keyFallback !== lastRenderedKey) {
                        lastRenderedKey = keyFallback;
                        renderResult(result);
                    }
                } else {
                    renderScanning(currentUrl);
                }
            });
        });
    }

    // Expose for automated validation harnesses
    window.FedraPopup = {
        renderResult: renderResult,
        renderScanning: renderScanning,
        renderOffline: renderOffline,
        updatePopup: updatePopup
    };

    // Auto-initialize when DOM is ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", function () {
            initRefs();
            updatePopup();
        });
    } else {
        initRefs();
        updatePopup();
    }

    // Listen to storage changes reactively
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.onChanged) {
        chrome.storage.onChanged.addListener(function (changes, areaName) {
            if (areaName === "local" && changes.last_result) {
                updatePopup();
            }
        });
    }

    // Polling interval (1.5 seconds)
    if (typeof chrome !== "undefined" && chrome.tabs) {
        setInterval(updatePopup, 1500);
    }
})();
