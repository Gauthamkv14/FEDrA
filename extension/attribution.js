/**
 * attribution.js — FEDrA Model-Faithful Feature Attribution Module (Step 7.1)
 * ===========================================================================
 * Exact Linear Logit Decomposition for Logistic Regression Models:
 * 
 * For a binary logistic model with StandardScaler:
 *   Decision Score (Logit): z = b_0 + sum_i (w_i * (x_i - mu_i) / sigma_i)
 *   Contribution:           c_i = w_i * (x_i - mu_i) / sigma_i
 *   Reconstruction:         z = b_0 + sum_i c_i
 *   Probability (Phishing): P = 1 / (1 + exp(-z))
 * 
 * Directional Interpretation:
 *   c_i > 0 : Positive log-odds contribution pushing toward PHISHING.
 *   c_i < 0 : Negative log-odds contribution pushing toward LEGITIMATE.
 * 
 * Properties:
 *   - Mathematically exact to floating-point precision (zero approximation error).
 *   - Deterministic: identical inputs yield identical ranked explanations.
 *   - Pure JavaScript, zero ML dependencies, browser and worker compatible.
 */

(function (global) {
    "use strict";

    var attributionParams = null;
    var isInitializing = false;
    var initPromise = null;

    /**
     * Load JSON parameters from extension directory or Node filesystem.
     */
    async function loadJsonMetadata(relativePath) {
        if (typeof process !== "undefined" && process.versions && process.versions.node) {
            try {
                var nodeFs = require("fs");
                var path = require("path");
                var fullPath = (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.getURL)
                    ? chrome.runtime.getURL(relativePath)
                    : relativePath;
                var content = nodeFs.readFileSync(fullPath, "utf-8");
                return JSON.parse(content);
            } catch (e) {
                // fall through to fetch
            }
        }
        if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.getURL) {
            var url = chrome.runtime.getURL(relativePath);
            var res = await fetch(url);
            if (!res.ok) throw new Error("Failed to fetch " + url + " (" + res.status + ")");
            return await res.json();
        } else if (typeof fetch !== "undefined") {
            var res2 = await fetch(relativePath);
            return await res2.json();
        } else {
            throw new Error("[FEDrA Attribution] No mechanism available to load JSON metadata.");
        }
    }

    /**
     * Initialize attribution parameters (idempotent).
     */
    async function initializeAttribution(paramsOverride) {
        if (paramsOverride) {
            attributionParams = paramsOverride;
            return attributionParams;
        }
        if (attributionParams) {
            return attributionParams;
        }
        if (isInitializing && initPromise) {
            return initPromise;
        }

        isInitializing = true;
        initPromise = (async function () {
            try {
                attributionParams = await loadJsonMetadata("models/attribution_parameters.json");
                isInitializing = false;
                return attributionParams;
            } catch (err) {
                isInitializing = false;
                initPromise = null;
                console.error("[FEDrA Attribution] Failed to load attribution parameters:", err);
                throw err;
            }
        })();

        return initPromise;
    }

    /**
     * Decompose and explain a logistic regression prediction.
     */
    function computeLinearAttribution(rawFeatures, modelParams, modalityName) {
        if (!rawFeatures || rawFeatures.length !== modelParams.feature_dim) {
            throw new Error(
                "[FEDrA Attribution] " + modalityName.toUpperCase() + " feature dimension mismatch: expected " +
                modelParams.feature_dim + ", got " + (rawFeatures ? rawFeatures.length : "null")
            );
        }

        var names = modelParams.feature_names;
        var mean = modelParams.scaler.mean;
        var scale = modelParams.scaler.scale;
        var coefs = modelParams.coefficients;
        var intercept = modelParams.intercept;
        var descriptions = modelParams.descriptions || {};

        var featureAttributions = [];
        var sumContributions = 0.0;

        for (var i = 0; i < modelParams.feature_dim; i++) {
            var rawVal = Number(rawFeatures[i]);
            var s = (scale[i] === 0 || !scale[i]) ? 1.0 : scale[i];
            var scaledVal = (rawVal - mean[i]) / s;
            var contribution = coefs[i] * scaledVal;
            sumContributions += contribution;

            var direction = contribution > 0 ? "phishing" : "legitimate";
            var featName = names[i];
            var descObj = descriptions[featName] || {};
            var desc = descObj[direction] || (direction === "phishing" ? "Elevated risk indicator: " + featName : "Normal/safe pattern: " + featName);

            featureAttributions.push({
                name: featName,
                raw_value: rawVal,
                scaled_value: scaledVal,
                coefficient: coefs[i],
                contribution: contribution,
                abs_contribution: Math.abs(contribution),
                direction: direction,
                description: desc
            });
        }

        var decisionScore = intercept + sumContributions;
        var phishingProb = 1.0 / (1.0 + Math.exp(-decisionScore));
        var predictedLabel = decisionScore > 0 ? 1 : 0;
        var prediction = predictedLabel === 1 ? "PHISHING" : "LEGITIMATE";

        // Sort by magnitude (absolute contribution)
        var rankedByImpact = featureAttributions.slice().sort(function (a, b) {
            return b.abs_contribution - a.abs_contribution;
        });

        // Top phishing contributors (c_i > 0, highest positive values)
        var topPhishing = featureAttributions
            .filter(function (f) { return f.contribution > 0; })
            .sort(function (a, b) { return b.contribution - a.contribution; });

        // Top legitimate contributors (c_i < 0, most negative values)
        var topLegitimate = featureAttributions
            .filter(function (f) { return f.contribution < 0; })
            .sort(function (a, b) { return a.contribution - b.contribution; });

        // Ranked human-readable reasons
        var rankedReasons = rankedByImpact.map(function (f) {
            return {
                feature: f.name,
                direction: f.direction,
                contribution: f.contribution,
                abs_contribution: f.abs_contribution,
                reason: f.description
            };
        });

        return {
            modality: modalityName,
            prediction: prediction,
            predicted_label: predictedLabel,
            phishing_probability: phishingProb,
            phishing_probability_pct: Math.round(phishingProb * 10000) / 100,
            decision_score: decisionScore,
            intercept: intercept,
            reconstructed_score: decisionScore,
            reconstruction_error: 0.0, // exactly matches decisionScore
            features: featureAttributions,
            top_phishing_contributors: topPhishing,
            top_legitimate_contributors: topLegitimate,
            ranked_reasons: rankedReasons
        };
    }

    /**
     * Explain URL baseline model prediction.
     * @param {Array<number>|Float32Array} rawFeatures22 - 22 canonical URL features
     * @returns {Promise<Object>} Attribution explanation object
     */
    async function explainUrlPrediction(rawFeatures22) {
        var t0 = performance.now();
        var params = await initializeAttribution();
        var result = computeLinearAttribution(rawFeatures22, params.url, "url");
        result.attribution_latency_ms = Math.round((performance.now() - t0) * 100) / 100;
        return result;
    }

    /**
     * Explain HTML baseline model prediction.
     * @param {Array<number>|Float32Array} rawFeatures12 - 12 canonical HTML features
     * @returns {Promise<Object>} Attribution explanation object
     */
    async function explainHtmlPrediction(rawFeatures12) {
        var t0 = performance.now();
        var params = await initializeAttribution();
        var result = computeLinearAttribution(rawFeatures12, params.html, "html");
        result.attribution_latency_ms = Math.round((performance.now() - t0) * 100) / 100;
        return result;
    }

    /**
     * Precomputed coordinate mapping for bilinear interpolation (7x7 -> 224x224).
     */
    var interpCoords = null;
    function getInterpCoords() {
        if (!interpCoords) {
            interpCoords = [];
            for (var i = 0; i < 224; i++) {
                var u = (i + 0.5) / 32.0 - 0.5;
                if (u < 0.0) u = 0.0;
                if (u > 6.0) u = 6.0;
                var i0 = Math.floor(u);
                var i1 = Math.min(i0 + 1, 6);
                var w = u - i0;
                interpCoords.push({ i0: i0, i1: i1, w: w });
            }
        }
        return interpCoords;
    }

    /**
     * Bilinear interpolation from 7x7 grid to 224x224 heatmap (align_corners=False).
     * @param {Float32Array|Array<number>} grid7x7 - 49 float values
     * @returns {Float32Array} 50,176 float values (224x224)
     */
    function interpolate7x7To224x224(grid7x7) {
        var coords = getInterpCoords();
        var out = new Float32Array(224 * 224);
        for (var y = 0; y < 224; y++) {
            var cy = coords[y];
            var y0 = cy.i0;
            var y1 = cy.i1;
            var wy = cy.w;
            var rowOffset = y * 224;
            var r0 = y0 * 7;
            var r1 = y1 * 7;
            for (var x = 0; x < 224; x++) {
                var cx = coords[x];
                var x0 = cx.i0;
                var x1 = cx.i1;
                var wx = cx.w;

                var v00 = grid7x7[r0 + x0];
                var v01 = grid7x7[r0 + x1];
                var v10 = grid7x7[r1 + x0];
                var v11 = grid7x7[r1 + x1];

                out[rowOffset + x] = (1.0 - wx) * (1.0 - wy) * v00 +
                                     wx * (1.0 - wy) * v01 +
                                     (1.0 - wx) * wy * v10 +
                                     wx * wy * v11;
            }
        }
        return out;
    }

    /**
     * Compute visual Grad-CAM heatmap from MobileNetV2 spatial features (1280x7x7).
     * Exact Grad-CAM Channel Weight: alpha_k = w_k / (SPATIAL_AREA * sigma_k) where SPATIAL_AREA = 49.
     * Activation Map: L_GradCAM(p) = ReLU(sum_k alpha_k * A_{k, p})
     * @param {Float32Array|Array<number>} spatialData - 62,720 float values (1280 x 7 x 7, channel-first)
     * @param {Object} visualParams - visual attribution metadata (containing gradcam_channel_weights)
     * @returns {Object} Grad-CAM heatmap explanation object
     */
    function computeVisualGradCam(spatialData, visualParams) {
        if (!spatialData || spatialData.length !== 1280 * 49) {
            throw new Error(
                "[FEDrA Attribution] Spatial features dimension mismatch: expected 62720 (1280x7x7), got " +
                (spatialData ? spatialData.length : "null")
            );
        }

        var weights = visualParams.gradcam_channel_weights;
        if (!weights || weights.length !== 1280) {
            throw new Error("[FEDrA Attribution] Visual Grad-CAM channel weights missing or invalid (expected 1280).");
        }

        // 1. Channel-weighted combination across 1280 channels: L(p) = ReLU(sum_k alpha_k * A_{k, p})
        var grid7x7 = new Float32Array(49);
        for (var k = 0; k < 1280; k++) {
            var w = weights[k];
            var offset = k * 49;
            for (var p = 0; p < 49; p++) {
                grid7x7[p] += w * spatialData[offset + p];
            }
        }

        // Apply ReLU
        for (var p2 = 0; p2 < 49; p2++) {
            if (grid7x7[p2] < 0.0) {
                grid7x7[p2] = 0.0;
            }
        }

        // 2. Bilinear upsampling to 224x224
        var heatmap224 = interpolate7x7To224x224(grid7x7);

        // 3. Min-Max Normalization to [0.0, 1.0]
        var minVal = Infinity;
        var maxVal = -Infinity;
        for (var i = 0; i < 50176; i++) {
            var v = heatmap224[i];
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }

        var denom = maxVal - minVal;
        var normalizedHeatmap = new Float32Array(50176);
        if (denom > 1e-8) {
            for (var j = 0; j < 50176; j++) {
                normalizedHeatmap[j] = (heatmap224[j] - minVal) / denom;
            }
        }

        // 4. Extract Peak Regions from 7x7 grid
        var peakRegions = [];
        var gridMin = Infinity;
        var gridMax = -Infinity;
        for (var gp = 0; gp < 49; gp++) {
            if (grid7x7[gp] < gridMin) gridMin = grid7x7[gp];
            if (grid7x7[gp] > gridMax) gridMax = grid7x7[gp];
        }
        var gridDenom = gridMax - gridMin;

        if (gridDenom > 1e-8) {
            for (var gy = 0; gy < 7; gy++) {
                for (var gx = 0; gx < 7; gx++) {
                    var rawCell = grid7x7[gy * 7 + gx];
                    var normScore = (rawCell - gridMin) / gridDenom;
                    if (normScore >= 0.5) {
                        peakRegions.push({
                            grid_y: gy,
                            grid_x: gx,
                            intensity: Math.round(normScore * 10000) / 10000,
                            bbox_224: [gx * 32, gy * 32, (gx + 1) * 32, (gy + 1) * 32]
                        });
                    }
                }
            }
            peakRegions.sort(function (a, b) { return b.intensity - a.intensity; });
        }

        // Build 2D 7x7 grid array for JSON serialization
        var grid2D = [];
        for (var r = 0; r < 7; r++) {
            var row = [];
            for (var c = 0; c < 7; c++) {
                row.push(grid7x7[r * 7 + c]);
            }
            grid2D.push(row);
        }

        return {
            modality: "visual",
            target_layer: visualParams.target_layer || "mobilenet_v2.features.18",
            spatial_grid_7x7: grid2D,
            normalized_heatmap_224x224: Array.from(normalizedHeatmap),
            heatmap_min: minVal,
            heatmap_max: maxVal,
            peak_regions: peakRegions.slice(0, 5),
            description: visualParams.description || "Visual activation heatmap indicating spatial regions that contribute toward phishing visual indicators."
        };
    }

    /**
     * Explain visual prediction via Grad-CAM from extracted spatial features.
     * @param {Float32Array|Array<number>} spatialFeatures - 62,720 spatial activations from MobileNetV2
     * @returns {Promise<Object>} Grad-CAM explanation object
     */
    async function explainVisualGradCam(spatialFeatures) {
        var t0 = performance.now();
        var params = await initializeAttribution();
        var result = computeVisualGradCam(spatialFeatures, params.visual);
        result.attribution_latency_ms = Math.round((performance.now() - t0) * 100) / 100;
        return result;
    }

    /**
     * Compact summary for a linear structured modality (URL / HTML).
     */
    function summarizeLinearModality(modalityExpl, topN) {
        if (!topN) topN = 3;
        if (!modalityExpl || !modalityExpl.features) {
            return {
                available: false,
                modality: modalityExpl && modalityExpl.modality ? modalityExpl.modality : "unknown",
                error_code: "MODALITY_EXPLANATION_UNAVAILABLE",
                description: "Explanation data for this modality is unavailable."
            };
        }

        var features = modalityExpl.features;
        var sorted = features.slice().sort(function (a, b) {
            return b.abs_contribution - a.abs_contribution;
        });

        var topFeatures = [];
        for (var i = 0; i < Math.min(topN, sorted.length); i++) {
            var f = sorted[i];
            var dir = f.contribution > 0 ? "phishing" : (f.contribution < 0 ? "legitimate" : "neutral");
            topFeatures.push({
                feature: f.name,
                raw_value: f.raw_value,
                scaled_value: f.scaled_value,
                attribution: f.contribution,
                abs_attribution: f.abs_contribution,
                direction: dir,
                description: f.description
            });
        }

        var posAttr = 0.0;
        var negAttr = 0.0;
        for (var j = 0; j < features.length; j++) {
            var c = features[j].contribution;
            if (c > 0) posAttr += c;
            else if (c < 0) negAttr += c;
        }

        return {
            available: true,
            modality: modalityExpl.modality,
            prediction: modalityExpl.prediction,
            predicted_label: modalityExpl.predicted_label,
            phishing_probability: modalityExpl.phishing_probability,
            phishing_probability_pct: modalityExpl.phishing_probability_pct,
            decision_score: modalityExpl.decision_score,
            top_features: topFeatures,
            total_positive_attribution: posAttr,
            total_negative_attribution: negAttr,
            attribution_magnitude: Math.abs(posAttr) + Math.abs(negAttr)
        };
    }

    /**
     * Compact summary for visual Grad-CAM modality.
     */
    function summarizeVisualModality(visualExpl, imagePrediction) {
        if (!visualExpl) {
            return {
                available: false,
                modality: "visual",
                error_code: "VISUAL_EXPLANATION_UNAVAILABLE",
                description: "Visual explanation data is unavailable."
            };
        }

        var grid = visualExpl.spatial_grid_7x7 || [];
        var activeCount = 0;
        for (var r = 0; r < grid.length; r++) {
            for (var c = 0; c < grid[r].length; c++) {
                if (grid[r][c] > 0) activeCount++;
            }
        }

        var heatMin = Number(visualExpl.heatmap_min || 0.0);
        var heatMax = Number(visualExpl.heatmap_max || 0.0);
        var heatMean = (heatMin + heatMax) / 2.0;

        var predLabel = imagePrediction ? imagePrediction.predicted_label : (activeCount > 0 ? 1 : 0);
        var predStr = imagePrediction ? imagePrediction.prediction : (predLabel === 1 ? "PHISHING" : "LEGITIMATE");
        var prob = imagePrediction ? imagePrediction.phishing_probability : 0.5;
        var probPct = imagePrediction ? imagePrediction.phishing_probability_pct : 50.0;

        return {
            available: true,
            modality: "visual",
            method: "Grad-CAM",
            prediction: predStr,
            predicted_label: predLabel,
            phishing_probability: prob,
            phishing_probability_pct: probPct,
            target_class: 1,
            target_label: "phishing",
            heatmap_width: 224,
            heatmap_height: 224,
            heatmap_min: heatMin,
            heatmap_max: heatMax,
            heatmap_mean: heatMean,
            peak_regions: visualExpl.peak_regions || [],
            active_cells_count: activeCount,
            description: "The heatmap identifies spatial regions receiving high visual importance for the phishing-class prediction."
        };
    }

    /**
     * Deterministic evaluation of cross-modal agreement across available modalities.
     */
    function evaluateCrossModalAgreement(urlSummary, htmlSummary, visualSummary) {
        var evaluated = [];
        var phishCount = 0;
        var legitCount = 0;

        if (urlSummary && urlSummary.available) {
            evaluated.push("url");
            if (urlSummary.predicted_label === 1 || urlSummary.prediction === "PHISHING") phishCount++;
            else legitCount++;
        }
        if (htmlSummary && htmlSummary.available) {
            evaluated.push("html");
            if (htmlSummary.predicted_label === 1 || htmlSummary.prediction === "PHISHING") phishCount++;
            else legitCount++;
        }
        if (visualSummary && visualSummary.available) {
            evaluated.push("visual");
            if (visualSummary.predicted_label === 1 || visualSummary.prediction === "PHISHING") phishCount++;
            else legitCount++;
        }

        var total = evaluated.length;
        var status = "UNAVAILABLE";
        var summaryText = "";

        if (total === 0) {
            status = "UNAVAILABLE";
            summaryText = "No modality predictions available to assess cross-modal agreement.";
        } else if (total === 3) {
            if (phishCount === 3) {
                status = "ALL_PHISHING";
                summaryText = "Strong multimodal agreement: All 3 modalities indicate phishing risk.";
            } else if (legitCount === 3) {
                status = "ALL_LEGITIMATE";
                summaryText = "Strong multimodal agreement: All 3 modalities indicate legitimate page patterns.";
            } else {
                status = "MIXED";
                summaryText = "Mixed multimodal evidence: " + phishCount + " modality/modalities indicate phishing and " + legitCount + " indicate legitimate.";
            }
        } else {
            if (phishCount === total) {
                status = "PARTIAL_PHISHING";
                summaryText = "Partial multimodal evidence: All " + total + " available modality/modalities indicate phishing risk (" + (3 - total) + " unavailable).";
            } else if (legitCount === total) {
                status = "PARTIAL_LEGITIMATE";
                summaryText = "Partial multimodal evidence: All " + total + " available modality/modalities indicate legitimate page patterns (" + (3 - total) + " unavailable).";
            } else {
                status = "PARTIAL_MIXED";
                summaryText = "Partial mixed multimodal evidence: " + phishCount + " indicate phishing, " + legitCount + " indicate legitimate (" + (3 - total) + " unavailable).";
            }
        }

        return {
            status: status,
            phishing_modalities_count: phishCount,
            legitimate_modalities_count: legitCount,
            total_available_modalities: total,
            modalities_evaluated: evaluated,
            summary_text: summaryText
        };
    }

    /**
     * Synthesize individual modality explanations into a unified Multimodal Explanation Object (Step 7.3).
     */
    function fuseMultimodalExplanations(options) {
        var opts = options || {};
        var topN = opts.topN || 3;
        var urlSum = summarizeLinearModality(opts.urlExplanation, topN);
        var htmlSum = summarizeLinearModality(opts.htmlExplanation, topN);
        var visSum = summarizeVisualModality(opts.visualExplanation, opts.imagePrediction);

        var agreement = evaluateCrossModalAgreement(urlSum, htmlSum, visSum);

        var finalPred = opts.finalPrediction || null;
        var finalPredObj = null;
        if (finalPred) {
            var finalClass = (
                finalPred.class === 1 ||
                finalPred.predicted_label === 1 ||
                finalPred.label === "PHISHING" ||
                finalPred.prediction === "PHISHING"
            ) ? 1 : 0;
            var finalLabel = (finalPred.label || finalPred.prediction)
                ? (finalPred.label || finalPred.prediction)
                : (finalClass === 1 ? "PHISHING" : "LEGITIMATE");
            var finalProb = typeof finalPred.phishing_probability === "number" ? finalPred.phishing_probability : 0.0;
            var finalProbPct = typeof finalPred.phishing_probability_pct === "number" ? finalPred.phishing_probability_pct : (Math.round(finalProb * 10000) / 100);
            var finalSource = finalPred.source ? finalPred.source : "fusion_model";

            finalPredObj = {
                class: finalClass,
                label: finalLabel,
                phishing_probability: finalProb,
                phishing_probability_pct: finalProbPct,
                source: finalSource
            };
        }

        return {
            schema_version: "1.0",
            explanation_type: "multimodal_evidence_summary",
            final_prediction: finalPredObj,
            cross_modal_agreement: agreement,
            modalities: {
                url: urlSum,
                html: htmlSum,
                visual: visSum
            },
            metadata: {
                top_n_features: topN,
                fusion_weights: {
                    url: 0.4,
                    html: 0.3,
                    visual: 0.3
                },
                disclaimer: "Multimodal explanation provides a structured summary of modality-specific evidence. Final decision is authoritative from the fusion classifier."
            }
        };
    }

    /**
     * Explain complete multimodal analysis downstream of prediction.
     * Safely executes URL, HTML, and Visual Grad-CAM explainability, and fuses into
     * schema version 1.0 explanation object. Never throws; returns safe error object on failure.
     * @param {Object} inputs - { urlFeatures, htmlFeatures, spatialFeatures, imagePrediction, finalPrediction, topN }
     * @returns {Promise<Object>} Unified Multimodal Explanation Object
     */
    async function explainMultimodalPipeline(inputs) {
        var t0 = performance.now();
        var inps = inputs || {};
        try {
            await initializeAttribution();

            // 1. URL Model Explanation
            var urlExpl = null;
            if (inps.urlFeatures && inps.urlFeatures.length === 22) {
                try {
                    urlExpl = await explainUrlPrediction(inps.urlFeatures);
                } catch (uErr) {
                    console.warn("[FEDrA Attribution] URL explanation error:", uErr);
                }
            }

            // 2. HTML Model Explanation
            var htmlExpl = null;
            if (inps.htmlFeatures && inps.htmlFeatures.length === 12) {
                try {
                    htmlExpl = await explainHtmlPrediction(inps.htmlFeatures);
                } catch (hErr) {
                    console.warn("[FEDrA Attribution] HTML explanation error:", hErr);
                }
            }

            // 3. Visual Grad-CAM Explanation
            var visExpl = null;
            if (inps.spatialFeatures && inps.spatialFeatures.length === 62720) {
                try {
                    visExpl = await explainVisualGradCam(inps.spatialFeatures);
                } catch (vErr) {
                    console.warn("[FEDrA Attribution] Visual Grad-CAM error:", vErr);
                }
            }

            // 4. Fuse into unified Multimodal Explanation Object
            var fused = fuseMultimodalExplanations({
                urlExplanation: urlExpl,
                htmlExplanation: htmlExpl,
                visualExplanation: visExpl,
                imagePrediction: inps.imagePrediction || null,
                finalPrediction: inps.finalPrediction || null,
                topN: inps.topN || 3
            });

            fused.metadata = fused.metadata || {};
            fused.metadata.explanation_latency_ms = Math.round((performance.now() - t0) * 100) / 100;
            return fused;
        } catch (err) {
            console.error("[FEDrA Attribution] Multimodal pipeline explanation error:", err);
            return {
                schema_version: "1.0",
                explanation_type: "multimodal_evidence_summary",
                available: false,
                error_code: "EXPLANATION_GENERATION_FAILED",
                description: "Failed to generate multimodal explanation: " + err.message,
                final_prediction: inps.finalPrediction || null,
                metadata: {
                    explanation_latency_ms: Math.round((performance.now() - t0) * 100) / 100
                }
            };
        }
    }

    /**
     * Explain complete structured modalities (URL + HTML).
     */
    async function explainStructuredModalities(inputs) {
        var t0 = performance.now();
        var explanations = {
            url: null,
            html: null,
            combined_latency_ms: 0
        };

        if (inputs.urlFeatures && inputs.urlFeatures.length === 22) {
            explanations.url = await explainUrlPrediction(inputs.urlFeatures);
        }
        if (inputs.htmlFeatures && inputs.htmlFeatures.length === 12) {
            explanations.html = await explainHtmlPrediction(inputs.htmlFeatures);
        }

        explanations.combined_latency_ms = Math.round((performance.now() - t0) * 100) / 100;
        return explanations;
    }

    // Export module
    var FedraAttribution = {
        initializeAttribution: initializeAttribution,
        explainUrlPrediction: explainUrlPrediction,
        explainHtmlPrediction: explainHtmlPrediction,
        explainVisualGradCam: explainVisualGradCam,
        explainStructuredModalities: explainStructuredModalities,
        explainMultimodalPipeline: explainMultimodalPipeline,
        summarizeLinearModality: summarizeLinearModality,
        summarizeVisualModality: summarizeVisualModality,
        evaluateCrossModalAgreement: evaluateCrossModalAgreement,
        fuseMultimodalExplanations: fuseMultimodalExplanations,
        computeLinearAttribution: computeLinearAttribution,
        computeVisualGradCam: computeVisualGradCam,
        interpolate7x7To224x224: interpolate7x7To224x224,
        getParams: function () { return attributionParams; }
    };

    if (typeof module !== "undefined" && module.exports) {
        module.exports = FedraAttribution;
    } else {
        global.FedraAttribution = FedraAttribution;
    }
})(typeof self !== "undefined" ? self : this);



