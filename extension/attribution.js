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
        explainStructuredModalities: explainStructuredModalities,
        computeLinearAttribution: computeLinearAttribution,
        getParams: function () { return attributionParams; }
    };

    if (typeof module !== "undefined" && module.exports) {
        module.exports = FedraAttribution;
    } else {
        global.FedraAttribution = FedraAttribution;
    }
})(typeof self !== "undefined" ? self : this);
