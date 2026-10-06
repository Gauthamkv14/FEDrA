/**
 * inference.js — FEDrA In-Browser ONNX Inference Module (Step 6.3)
 * ===============================================================
 * Responsibilities:
 * 1. Configures ONNX Runtime Web (WASM execution provider).
 * 2. Caches and reuses InferenceSessions for:
 *    - URL Baseline (url_baseline.onnx: 22 -> 2)
 *    - HTML Baseline (html_baseline.onnx: 12 -> 2)
 *    - MobileNetV2 GAP Feature Extractor (mobilenet_v2_visual.onnx: [1,3,224,224] -> [1,1280])
 *    - Image Baseline (image_baseline.onnx: 1280 -> 2)
 *    - Fusion MLP Classifier (fusion_model.onnx: 1314 -> 2)
 * 3. Preprocesses browser tab screenshots (Resize 256 -> CenterCrop 224 -> ImageNet RGB Normalization -> NCHW).
 * 4. Preprocesses and scales multimodal features (URL x 0.4, HTML x 0.3, Visual x 0.3 -> 1314 dims).
 * 5. Executes fast client-side inference across all modalities and computes final multimodal verdict.
 * 6. Fails safely without throwing unhandled exceptions.
 */

(function (global) {
    "use strict";

    var urlSession = null;
    var htmlSession = null;
    var mobilenetSession = null;
    var imageSession = null;
    var fusionSession = null;
    var modalityScalers = null;

    var isInitializing = false;
    var initPromise = null;
    var initLatencyMs = 0;

    /**
     * Configure ONNX Runtime Web environment.
     */
    function configureOrtEnv() {
        if (typeof ort === "undefined") {
            throw new Error("[FEDrA Inference] onnxruntime-web library (ort) is not loaded.");
        }
        if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.getURL && typeof process === "undefined") {
            ort.env.wasm.wasmPaths = chrome.runtime.getURL("libs/onnxruntime-web/");
        }
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.proxy = false;
    }

    /**
     * Load an ONNX model array buffer from the extension directory.
     */
    async function loadModelBuffer(relativePath) {
        if (typeof process !== "undefined" && process.versions && process.versions.node) {
            try {
                var nodeFs = require("fs");
                var fullPath = (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.getURL)
                    ? chrome.runtime.getURL(relativePath)
                    : relativePath;
                var nodeBuf = nodeFs.readFileSync(fullPath);
                return nodeBuf.buffer.slice(nodeBuf.byteOffset, nodeBuf.byteOffset + nodeBuf.byteLength);
            } catch (e) {
                console.warn("[FEDrA Inference] Node fs read error, falling back to fetch:", e.message);
            }
        }
        if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.getURL) {
            var url = chrome.runtime.getURL(relativePath);
            var res = await fetch(url);
            if (!res.ok) {
                throw new Error("Failed to fetch model at " + url + " (HTTP " + res.status + ")");
            }
            return await res.arrayBuffer();
        } else if (typeof fetch !== "undefined") {
            var res = await fetch(relativePath);
            return await res.arrayBuffer();
        } else {
            throw new Error("No mechanism available to load model buffer.");
        }
    }

    /**
     * Load JSON metadata file (such as modality_scalers.json).
     */
    async function loadJsonMetadata(relativePath) {
        if (typeof process !== "undefined" && process.versions && process.versions.node) {
            try {
                var nodeFs = require("fs");
                var fullPath = (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.getURL)
                    ? chrome.runtime.getURL(relativePath)
                    : relativePath;
                var content = nodeFs.readFileSync(fullPath, "utf-8");
                return JSON.parse(content);
            } catch (e) {
                console.warn("[FEDrA Inference] Node fs read error for JSON, falling back to fetch:", e.message);
            }
        }
        if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.getURL) {
            var url = chrome.runtime.getURL(relativePath);
            var res = await fetch(url);
            if (!res.ok) {
                throw new Error("Failed to fetch JSON at " + url + " (HTTP " + res.status + ")");
            }
            return await res.json();
        } else if (typeof fetch !== "undefined") {
            var res = await fetch(relativePath);
            return await res.json();
        } else {
            throw new Error("No mechanism available to load JSON metadata.");
        }
    }

    /**
     * Initialize and cache ONNX sessions for URL, HTML, MobileNetV2, Image Baseline, and Fusion models.
     * Guaranteed to be idempotent; concurrent callers share the same initialization Promise.
     */
    async function initializeInference() {
        if (urlSession && htmlSession && mobilenetSession && imageSession && fusionSession && modalityScalers) {
            return {
                status: "ready",
                cached: true,
                initLatencyMs: initLatencyMs
            };
        }

        if (isInitializing && initPromise) {
            return initPromise;
        }

        isInitializing = true;
        initPromise = (async function () {
            var t0 = performance.now();
            configureOrtEnv();

            try {
                // 1. URL Model Session
                if (!urlSession) {
                    var urlBuf = await loadModelBuffer("models/url_baseline.onnx");
                    urlSession = await ort.InferenceSession.create(urlBuf, {
                        executionProviders: ["wasm"]
                    });
                    console.log("[FEDrA Inference] URL baseline session created successfully.");
                }

                // 2. HTML Model Session
                if (!htmlSession) {
                    var htmlBuf = await loadModelBuffer("models/html_baseline.onnx");
                    htmlSession = await ort.InferenceSession.create(htmlBuf, {
                        executionProviders: ["wasm"]
                    });
                    console.log("[FEDrA Inference] HTML baseline session created successfully.");
                }

                // 3. MobileNetV2 Visual Feature Extractor Session
                if (!mobilenetSession) {
                    var mv2Buf = await loadModelBuffer("models/mobilenet_v2_visual.onnx");
                    mobilenetSession = await ort.InferenceSession.create(mv2Buf, {
                        executionProviders: ["wasm"]
                    });
                    console.log("[FEDrA Inference] MobileNetV2 visual session created successfully.");
                }

                // 4. Image Baseline Classifier Session
                if (!imageSession) {
                    var imgBuf = await loadModelBuffer("models/image_baseline.onnx");
                    imageSession = await ort.InferenceSession.create(imgBuf, {
                        executionProviders: ["wasm"]
                    });
                    console.log("[FEDrA Inference] Image baseline session created successfully.");
                }

                // 5. Fusion MLP Classifier Session
                if (!fusionSession) {
                    var fusionBuf = await loadModelBuffer("models/fusion_model.onnx");
                    fusionSession = await ort.InferenceSession.create(fusionBuf, {
                        executionProviders: ["wasm"]
                    });
                    console.log("[FEDrA Inference] Fusion MLP session created successfully.");
                }

                // 6. Modality Scalers & Weights Metadata
                if (!modalityScalers) {
                    modalityScalers = await loadJsonMetadata("models/modality_scalers.json");
                    console.log("[FEDrA Inference] Modality scalers loaded successfully.");
                }

                initLatencyMs = Math.round(performance.now() - t0);
                isInitializing = false;
                return {
                    status: "ready",
                    cached: false,
                    initLatencyMs: initLatencyMs
                };
            } catch (err) {
                isInitializing = false;
                initPromise = null;
                console.error("[FEDrA Inference] Session initialization error:", err);
                throw err;
            }
        })();

        return initPromise;
    }

    /**
     * Run local in-browser URL ONNX inference on a 22-dimensional raw feature vector.
     * @param {Array<number>|Float32Array} rawFeatures22 - 22 canonical URL features
     * @returns {Promise<Object>} Formatted prediction object
     */
    async function runUrlInference(rawFeatures22) {
        if (!rawFeatures22 || rawFeatures22.length !== 22) {
            throw new Error("[FEDrA Inference] Invalid URL features: expected 22 elements, got " + (rawFeatures22 ? rawFeatures22.length : "null"));
        }

        var t0 = performance.now();
        await initializeInference();

        var inputTensor = new ort.Tensor("float32", new Float32Array(rawFeatures22), [1, 22]);
        var results = await urlSession.run({ X: inputTensor });

        var tInference = performance.now() - t0;
        var label = Number(results.label.data[0]);
        var probs = Array.from(results.probabilities.data); // [prob_legit, prob_phish]
        var probPhish = probs[1];

        return {
            predicted_label: label,
            prediction: label === 1 ? "PHISHING" : "LEGITIMATE",
            phishing_probability: probPhish,
            phishing_probability_pct: Math.round(probPhish * 10000) / 100,
            probabilities: probs,
            inference_time_ms: Math.round(tInference * 100) / 100,
            model: "url_baseline.onnx",
            source: "browser_onnx_wasm"
        };
    }

    /**
     * Run local in-browser HTML ONNX inference on a 12-dimensional raw feature vector.
     * @param {Array<number>|Float32Array} rawFeatures12 - 12 canonical HTML features
     * @returns {Promise<Object>} Formatted prediction object
     */
    async function runHtmlInference(rawFeatures12) {
        if (!rawFeatures12 || rawFeatures12.length !== 12) {
            throw new Error("[FEDrA Inference] Invalid HTML features: expected 12 elements, got " + (rawFeatures12 ? rawFeatures12.length : "null"));
        }

        var t0 = performance.now();
        await initializeInference();

        var inputTensor = new ort.Tensor("float32", new Float32Array(rawFeatures12), [1, 12]);
        var results = await htmlSession.run({ X: inputTensor });

        var tInference = performance.now() - t0;
        var label = Number(results.label.data[0]);
        var probs = Array.from(results.probabilities.data); // [prob_legit, prob_phish]
        var probPhish = probs[1];

        return {
            predicted_label: label,
            prediction: label === 1 ? "PHISHING" : "LEGITIMATE",
            phishing_probability: probPhish,
            phishing_probability_pct: Math.round(probPhish * 10000) / 100,
            probabilities: probs,
            inference_time_ms: Math.round(tInference * 100) / 100,
            model: "html_baseline.onnx",
            source: "browser_onnx_wasm"
        };
    }

    /**
     * Preprocess a screenshot image (Data URL, Blob, or ImageBitmap) to a [1, 3, 224, 224] NCHW float32 Tensor.
     * Implements Resize(256) -> CenterCrop(224) -> ImageNet Normalization.
     */
    async function preprocessScreenshot(screenshotSource) {
        var t0 = performance.now();
        var bitmap = null;

        if (typeof ImageBitmap !== "undefined" && screenshotSource instanceof ImageBitmap) {
            bitmap = screenshotSource;
        } else if (typeof screenshotSource === "string" && screenshotSource.startsWith("data:")) {
            var blob;
            if (typeof fetch !== "undefined") {
                try {
                    var res = await fetch(screenshotSource);
                    blob = await res.blob();
                } catch (_) {
                    blob = null;
                }
            }
            if (!blob && typeof atob !== "undefined") {
                var parts = screenshotSource.split(",");
                var b64 = parts.length > 1 ? parts[1] : parts[0];
                var bin = atob(b64);
                var arr = new Uint8Array(bin.length);
                for (var j = 0; j < bin.length; j++) arr[j] = bin.charCodeAt(j);
                blob = new Blob([arr], { type: "image/png" });
            }
            if (typeof createImageBitmap !== "undefined") {
                bitmap = await createImageBitmap(blob);
            }
        } else if (typeof Blob !== "undefined" && screenshotSource instanceof Blob) {
            if (typeof createImageBitmap !== "undefined") {
                bitmap = await createImageBitmap(screenshotSource);
            }
        }

        if (!bitmap) {
            throw new Error("[FEDrA Inference] Unable to decode screenshot source into ImageBitmap.");
        }

        // Calculate aspect-preserving Resize(256) -> CenterCrop(224)
        var W0 = bitmap.width;
        var H0 = bitmap.height;
        var minDim = Math.min(W0, H0);
        var srcSize = (224 * minDim) / 256;
        var srcX = (W0 - srcSize) / 2;
        var srcY = (H0 - srcSize) / 2;

        var canvas;
        if (typeof OffscreenCanvas !== "undefined") {
            canvas = new OffscreenCanvas(224, 224);
        } else if (typeof document !== "undefined" && document.createElement) {
            canvas = document.createElement("canvas");
            canvas.width = 224;
            canvas.height = 224;
        } else {
            throw new Error("[FEDrA Inference] No Canvas implementation available for image preprocessing.");
        }

        var ctx = canvas.getContext("2d", { willReadFrequently: true });
        ctx.drawImage(bitmap, srcX, srcY, srcSize, srcSize, 0, 0, 224, 224);
        var imgData = ctx.getImageData(0, 0, 224, 224).data;

        // ImageNet RGB Normalization: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        var numPixels = 224 * 224;
        var floatArr = new Float32Array(3 * numPixels);
        var offsetG = numPixels;
        var offsetB = numPixels * 2;

        for (var i = 0; i < numPixels; i++) {
            var pxIdx = i * 4;
            var r = imgData[pxIdx + 0] / 255.0;
            var g = imgData[pxIdx + 1] / 255.0;
            var b = imgData[pxIdx + 2] / 255.0;

            floatArr[i] = (r - 0.485) / 0.229;
            floatArr[offsetG + i] = (g - 0.456) / 0.224;
            floatArr[offsetB + i] = (b - 0.406) / 0.225;
        }

        var tPrep = performance.now() - t0;
        var tensor = new ort.Tensor("float32", floatArr, [1, 3, 224, 224]);
        return {
            tensor: tensor,
            preprocess_time_ms: Math.round(tPrep * 100) / 100
        };
    }

    /**
     * Run local in-browser MobileNetV2 ONNX visual feature extraction.
     * @param {string|Blob|ImageBitmap|ort.Tensor} screenshotSource - Screenshot data or preprocessed tensor
     * @returns {Promise<Object>} 1280-dimensional visual embedding vector and metadata
     */
    async function runVisualInference(screenshotSource) {
        if (!screenshotSource) {
            throw new Error("[FEDrA Inference] Missing screenshot source for visual feature extraction.");
        }

        await initializeInference();

        var inputTensor;
        var tPrepMs = 0;

        if (screenshotSource instanceof ort.Tensor) {
            inputTensor = screenshotSource;
        } else {
            var prepResult = await preprocessScreenshot(screenshotSource);
            inputTensor = prepResult.tensor;
            tPrepMs = prepResult.preprocess_time_ms;
        }

        var t0 = performance.now();
        var results = await mobilenetSession.run({ image_input: inputTensor });
        var tInference = performance.now() - t0;

        var embeddingData = Array.from(results.visual_embedding.data);

        return {
            visual_embedding: embeddingData,
            embedding_dim: embeddingData.length,
            preprocess_time_ms: tPrepMs,
            inference_time_ms: Math.round(tInference * 100) / 100,
            model: "mobilenet_v2_visual.onnx",
            source: "browser_onnx_wasm"
        };
    }

    /**
     * Run local in-browser Image baseline ONNX inference on a 1280-dimensional visual embedding.
     * (StandardScaler is embedded inside the ONNX pipeline).
     * @param {Array<number>|Float32Array} visualEmbedding - 1280 raw visual embedding features
     * @returns {Promise<Object>} Formatted prediction object
     */
    async function runImageBaselineInference(visualEmbedding) {
        if (!visualEmbedding || visualEmbedding.length !== 1280) {
            throw new Error("[FEDrA Inference] Invalid visual embedding: expected 1280 elements, got " + (visualEmbedding ? visualEmbedding.length : "null"));
        }

        var t0 = performance.now();
        await initializeInference();

        var inputTensor = new ort.Tensor("float32", new Float32Array(visualEmbedding), [1, 1280]);
        var results = await imageSession.run({ X: inputTensor });

        var tInference = performance.now() - t0;
        var label = Number(results.label.data[0]);
        var probs = Array.from(results.probabilities.data); // [prob_legit, prob_phish]
        var probPhish = probs[1];

        return {
            predicted_label: label,
            prediction: label === 1 ? "PHISHING" : "LEGITIMATE",
            phishing_probability: probPhish,
            phishing_probability_pct: Math.round(probPhish * 10000) / 100,
            probabilities: probs,
            inference_time_ms: Math.round(tInference * 100) / 100,
            model: "image_baseline.onnx",
            source: "browser_onnx_wasm"
        };
    }

    /**
     * Preprocess and concatenate URL, HTML, and Visual features according to the fusion model contract:
     * [scale(url)*0.4, scale(html)*0.3, scale(visual)*0.3] -> Float32Array(1314)
     */
    function preprocessFusionInput(urlFeatures, htmlFeatures, visualEmbedding, scalersObj) {
        if (!urlFeatures || urlFeatures.length !== 22) {
            throw new Error("[FEDrA Inference] Invalid URL features: expected 22, got " + (urlFeatures ? urlFeatures.length : "null"));
        }
        if (!htmlFeatures || htmlFeatures.length !== 12) {
            throw new Error("[FEDrA Inference] Invalid HTML features: expected 12, got " + (htmlFeatures ? htmlFeatures.length : "null"));
        }
        if (!visualEmbedding || visualEmbedding.length !== 1280) {
            throw new Error("[FEDrA Inference] Invalid visual embedding: expected 1280, got " + (visualEmbedding ? visualEmbedding.length : "null"));
        }
        if (!scalersObj || !scalersObj.scalers || !scalersObj.modality_weights) {
            throw new Error("[FEDrA Inference] Modality scalers metadata not loaded.");
        }

        var weights = scalersObj.modality_weights;
        var scalers = scalersObj.scalers;
        var fused = new Float32Array(1314);
        var offset = 0;

        // 1. URL (22)
        var urlMean = scalers.url.mean;
        var urlScale = scalers.url.scale;
        var wUrl = weights.url; // 0.4
        for (var i = 0; i < 22; i++) {
            var s = (urlScale[i] === 0 || !urlScale[i]) ? 1.0 : urlScale[i];
            fused[offset + i] = ((urlFeatures[i] - urlMean[i]) / s) * wUrl;
        }
        offset += 22;

        // 2. HTML (12)
        var htmlMean = scalers.html.mean;
        var htmlScale = scalers.html.scale;
        var wHtml = weights.html; // 0.3
        for (var j = 0; j < 12; j++) {
            var sHtml = (htmlScale[j] === 0 || !htmlScale[j]) ? 1.0 : htmlScale[j];
            fused[offset + j] = ((htmlFeatures[j] - htmlMean[j]) / sHtml) * wHtml;
        }
        offset += 12;

        // 3. Visual (1280)
        var visMean = scalers.visual.mean;
        var visScale = scalers.visual.scale;
        var wVis = weights.visual; // 0.3
        for (var k = 0; k < 1280; k++) {
            var sVis = (visScale[k] === 0 || !visScale[k]) ? 1.0 : visScale[k];
            fused[offset + k] = ((visualEmbedding[k] - visMean[k]) / sVis) * wVis;
        }

        return fused;
    }

    /**
     * Run local in-browser Fusion MLP ONNX inference on concatenated & scaled multimodal features (1314 dims).
     * @param {Array<number>|Float32Array} urlFeatures - 22 raw URL features
     * @param {Array<number>|Float32Array} htmlFeatures - 12 raw HTML features
     * @param {Array<number>|Float32Array} visualEmbedding - 1280 raw visual embedding features
     * @returns {Promise<Object>} Formatted fusion prediction object
     */
    async function runFusionInference(urlFeatures, htmlFeatures, visualEmbedding) {
        var t0 = performance.now();
        await initializeInference();

        var fusedInput = preprocessFusionInput(urlFeatures, htmlFeatures, visualEmbedding, modalityScalers);
        var tPrepMs = Math.round((performance.now() - t0) * 100) / 100;

        var tInf0 = performance.now();
        var inputTensor = new ort.Tensor("float32", fusedInput, [1, 1314]);
        var results = await fusionSession.run({ X: inputTensor });
        var tInference = performance.now() - tInf0;

        var label = Number(results.label.data[0]);
        var probs = Array.from(results.probabilities.data); // [prob_legit, prob_phish]
        var probPhish = probs[1];

        return {
            predicted_label: label,
            prediction: label === 1 ? "PHISHING" : "LEGITIMATE",
            phishing_probability: probPhish,
            phishing_probability_pct: Math.round(probPhish * 10000) / 100,
            probabilities: probs,
            preprocess_time_ms: tPrepMs,
            inference_time_ms: Math.round(tInference * 100) / 100,
            model: "fusion_model.onnx",
            source: "browser_onnx_wasm"
        };
    }

    /**
     * Run complete end-to-end browser-local multimodal inference pipeline.
     */
    async function runFullBrowserPipeline(inputs) {
        var t0 = performance.now();
        var results = {
            url: null,
            html: null,
            visual_embedding: null,
            image: null,
            fusion: null,
            final_verdict: null,
            final_phishing_prob_pct: null,
            timings: {},
            success: false
        };

        try {
            await initializeInference();

            // 1. URL Model
            if (inputs.urlFeatures && inputs.urlFeatures.length === 22) {
                results.url = await runUrlInference(inputs.urlFeatures);
            }

            // 2. HTML Model
            if (inputs.htmlFeatures && inputs.htmlFeatures.length === 12) {
                results.html = await runHtmlInference(inputs.htmlFeatures);
            }

            // 3. Visual Feature Extractor
            var visualEmb = inputs.visualEmbedding || null;
            if (!visualEmb && inputs.screenshot) {
                var visRes = await runVisualInference(inputs.screenshot);
                visualEmb = visRes.visual_embedding;
                results.visual_extraction = visRes;
            }
            results.visual_embedding = visualEmb;

            // 4. Image Baseline Model
            if (visualEmb && visualEmb.length === 1280) {
                results.image = await runImageBaselineInference(visualEmb);
            }

            // 5. Fusion MLP Model
            if (inputs.urlFeatures && inputs.htmlFeatures && visualEmb) {
                results.fusion = await runFusionInference(inputs.urlFeatures, inputs.htmlFeatures, visualEmb);
                results.final_verdict = results.fusion.prediction;
                results.final_phishing_prob_pct = results.fusion.phishing_probability_pct;
                results.success = true;
            } else if (results.url) {
                // Fallback to URL-only if visual/HTML missing
                results.final_verdict = results.url.prediction;
                results.final_phishing_prob_pct = results.url.phishing_probability_pct;
                results.success = true;
            }

            results.timings.total_browser_pipeline_ms = Math.round((performance.now() - t0) * 100) / 100;
            return results;
        } catch (err) {
            console.error("[FEDrA Inference] Full browser pipeline error:", err);
            results.error = err.message;
            results.timings.total_browser_pipeline_ms = Math.round((performance.now() - t0) * 100) / 100;
            return results;
        }
    }

    // Export module
    var FedraInference = {
        initializeInference: initializeInference,
        runUrlInference: runUrlInference,
        runHtmlInference: runHtmlInference,
        preprocessScreenshot: preprocessScreenshot,
        runVisualInference: runVisualInference,
        runImageBaselineInference: runImageBaselineInference,
        preprocessFusionInput: preprocessFusionInput,
        runFusionInference: runFusionInference,
        runFullBrowserPipeline: runFullBrowserPipeline,
        getModalityScalers: function () { return modalityScalers; },
        isReady: function () {
            return urlSession !== null &&
                htmlSession !== null &&
                mobilenetSession !== null &&
                imageSession !== null &&
                fusionSession !== null &&
                modalityScalers !== null;
        },
        getInitLatencyMs: function () { return initLatencyMs; }
    };

    if (typeof module !== "undefined" && module.exports) {
        module.exports = FedraInference;
    } else {
        global.FedraInference = FedraInference;
    }
})(typeof self !== "undefined" ? self : this);

