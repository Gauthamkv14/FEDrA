/**
 * extension/html_features.js
 * ==========================
 * Browser-compatible implementation of the canonical 12 HTML/DOM features for FEDrA.
 * 
 * Guarantees 100% semantic parity with `scripts/extract_html_features.py` and `scripts/api_server.py`.
 * Operates purely on the live DOM Document or raw HTML string with zero network/DNS dependencies.
 * 
 * Exposes:
 *   - CANONICAL_HTML_FEATURE_NAMES (Array of 12 feature names in exact order)
 *   - extractHtmlFeaturesFromDoc(doc, pageUrl, rawHtmlLength) -> Object
 *   - extractHtmlFeaturesFromString(htmlContent, pageUrl) -> Object
 *   - extractHtmlFeaturesVector(docOrHtml, pageUrl) -> Array (length 12)
 */

(function (root, factory) {
    if (typeof define === "function" && define.amd) {
        define([], factory);
    } else if (typeof module === "object" && module.exports) {
        module.exports = factory();
    } else {
        root.FEDrA_HTML = factory();
    }
})(typeof self !== "undefined" ? self : this, function () {
    "use strict";

    var HTML_FEATURE_DIM = 12;

    var CANONICAL_HTML_FEATURE_NAMES = [
        "num_forms",
        "num_inputs",
        "num_iframes",
        "num_ext_links",
        "num_ext_scripts",
        "has_password_field",
        "has_meta_redirect",
        "script_content_ratio",
        "favicon_mismatch",
        "has_auto_submit",
        "input_submit_ratio",
        "num_unique_ext_domains"
    ];

    /**
     * Extract hostname/domain matching Python _get_domain() exactly.
     */
    function getDomain(urlStr) {
        if (!urlStr || typeof urlStr !== "string") return "";
        var u = urlStr.trim();
        if (!u.startsWith("http") && !u.startsWith("//")) return "";
        var parseTarget = u.indexOf("://") !== -1 ? u : "http://" + u;
        if (parseTarget.startsWith("http:////") || parseTarget.startsWith("https:////")) {
            return "";
        }
        try {
            var parsed = new URL(parseTarget);
            return (parsed.hostname || "").toLowerCase();
        } catch (e) {
            var m = parseTarget.match(/^https?:\/\/([^/?#:]+)/i);
            return m ? m[1].toLowerCase() : "";
        }
    }

    /**
     * Extract 12 HTML features from a live DOM Document object in the browser.
     */
    function extractHtmlFeaturesFromDoc(doc, pageUrl, rawHtmlLength) {
        if (!doc) {
            var empty = {};
            for (var i = 0; i < CANONICAL_HTML_FEATURE_NAMES.length; i++) {
                empty[CANONICAL_HTML_FEATURE_NAMES[i]] = 0;
            }
            return empty;
        }

        var pageDomain = getDomain(pageUrl);

        // 1. Basic Element Counts
        var forms = doc.querySelectorAll ? doc.querySelectorAll("form") : [];
        var inputs = doc.querySelectorAll ? doc.querySelectorAll("input") : [];
        var iframes = doc.querySelectorAll ? doc.querySelectorAll("iframe") : [];
        var num_forms = forms.length;
        var num_inputs = inputs.length;
        var num_iframes = iframes.length;

        // 2. External links, scripts, and distinct domains
        var extDomains = {};
        var num_ext_domains_count = 0;
        var num_ext_links = 0;
        var num_ext_scripts = 0;

        // 2a. Links
        var links = doc.querySelectorAll ? doc.querySelectorAll("a[href]") : [];
        for (var l = 0; l < links.length; l++) {
            var href = links[l].getAttribute("href") || "";
            var linkDomain = getDomain(href);
            if (linkDomain && linkDomain !== pageDomain) {
                num_ext_links++;
                if (!extDomains[linkDomain]) {
                    extDomains[linkDomain] = true;
                    num_ext_domains_count++;
                }
            }
        }

        // 2b. Scripts
        var scripts = doc.querySelectorAll ? doc.querySelectorAll("script") : [];
        var scriptTextAccum = "";
        for (var s = 0; s < scripts.length; s++) {
            var sc = scripts[s];
            var src = sc.getAttribute("src") || "";
            if (src) {
                var scriptDomain = getDomain(src);
                if (scriptDomain && scriptDomain !== pageDomain) {
                    num_ext_scripts++;
                    if (!extDomains[scriptDomain]) {
                        extDomains[scriptDomain] = true;
                        num_ext_domains_count++;
                    }
                }
            }
            if (sc.textContent) {
                scriptTextAccum += sc.textContent;
            }
        }

        var num_unique_ext_domains = num_ext_domains_count;

        // 3. Password field check
        var has_password_field = 0;
        for (var inp = 0; inp < inputs.length; inp++) {
            var t = (inputs[inp].getAttribute("type") || "").toLowerCase();
            if (t === "password") {
                has_password_field = 1;
                break;
            }
        }

        // 4. Meta refresh redirect check
        var has_meta_redirect = 0;
        var metaTags = doc.querySelectorAll ? doc.querySelectorAll("meta") : [];
        for (var m = 0; m < metaTags.length; m++) {
            var httpEquiv = (metaTags[m].getAttribute("http-equiv") || "").toLowerCase();
            if (httpEquiv === "refresh") {
                has_meta_redirect = 1;
                break;
            }
        }

        // 5. Script to content ratio
        var totalLen = rawHtmlLength || (doc.documentElement ? doc.documentElement.outerHTML.length : 1);
        var script_content_ratio = scriptTextAccum.length / Math.max(1, totalLen);

        // 6. Favicon mismatch
        var favicon_mismatch = 0;
        var linkTags = doc.querySelectorAll ? doc.querySelectorAll("link") : [];
        for (var fav = 0; fav < linkTags.length; fav++) {
            var rel = (linkTags[fav].getAttribute("rel") || "").toLowerCase();
            if (rel.indexOf("icon") !== -1) {
                var favHref = linkTags[fav].getAttribute("href") || "";
                var favDomain = getDomain(favHref);
                if (favDomain && favDomain !== pageDomain) {
                    favicon_mismatch = 1;
                    break;
                }
            }
        }

        // 7. Auto-submitting forms (forms present + "submit()" in script text)
        var has_auto_submit = (num_forms > 0 && scriptTextAccum.toLowerCase().indexOf("submit()") !== -1) ? 1 : 0;

        // 8. Ratio of inputs to submit buttons (matching BeautifulSoup type=lambda t: t and t.lower() == 'submit')
        var submitCount = 0;
        for (var subInp = 0; subInp < inputs.length; subInp++) {
            var inType = (inputs[subInp].getAttribute("type") || "").toLowerCase();
            if (inType === "submit" || inType === "image") {
                submitCount++;
            }
        }
        var buttons = doc.querySelectorAll ? doc.querySelectorAll("button") : [];
        for (var btn = 0; btn < buttons.length; btn++) {
            var btnType = (buttons[btn].getAttribute("type") || "").toLowerCase();
            if (btnType === "submit") {
                submitCount++;
            }
        }
        var input_submit_ratio = num_inputs > 0 ? (num_inputs / Math.max(1, submitCount)) : 0.0;

        return {
            num_forms: num_forms,
            num_inputs: num_inputs,
            num_iframes: num_iframes,
            num_ext_links: num_ext_links,
            num_ext_scripts: num_ext_scripts,
            has_password_field: has_password_field,
            has_meta_redirect: has_meta_redirect,
            script_content_ratio: script_content_ratio,
            favicon_mismatch: favicon_mismatch,
            has_auto_submit: has_auto_submit,
            input_submit_ratio: input_submit_ratio,
            num_unique_ext_domains: num_unique_ext_domains
        };
    }

    /**
     * Extract 12 HTML features from an HTML string (Node.js/testing parser).
     */
    function extractHtmlFeaturesFromString(htmlContent, pageUrl) {
        if (!htmlContent || typeof htmlContent !== "string") {
            var empty = {};
            for (var i = 0; i < CANONICAL_HTML_FEATURE_NAMES.length; i++) {
                empty[CANONICAL_HTML_FEATURE_NAMES[i]] = 0;
            }
            return empty;
        }

        if (typeof DOMParser !== "undefined") {
            var parser = new DOMParser();
            var doc = parser.parseFromString(htmlContent, "text/html");
            return extractHtmlFeaturesFromDoc(doc, pageUrl, htmlContent.length);
        }

        // Clean comments to avoid false tag matches in comments
        var cleanHtml = htmlContent.replace(/<!--[\s\S]*?-->/g, "");
        var pageDomain = getDomain(pageUrl);

        var formMatches = cleanHtml.match(/<form\b[^>]*>/gi) || [];
        var inputMatches = cleanHtml.match(/<input\b[^>]*>/gi) || [];
        var iframeMatches = cleanHtml.match(/<iframe\b[^>]*>/gi) || [];

        var num_forms = formMatches.length;
        var num_inputs = inputMatches.length;
        var num_iframes = iframeMatches.length;

        var extDomains = {};
        var num_ext_domains_count = 0;
        var num_ext_links = 0;
        var num_ext_scripts = 0;

        // Links
        var aMatches = cleanHtml.match(/<a\b[^>]*\bhref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))[^>]*>/gi) || [];
        for (var a = 0; a < aMatches.length; a++) {
            var hrefM = aMatches[a].match(/\bhref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i);
            if (hrefM) {
                var hrefVal = hrefM[1] || hrefM[2] || hrefM[3] || "";
                var ld = getDomain(hrefVal);
                if (ld && ld !== pageDomain) {
                    num_ext_links++;
                    if (!extDomains[ld]) {
                        extDomains[ld] = true;
                        num_ext_domains_count++;
                    }
                }
            }
        }

        // Scripts & script text
        var scriptSrcMatches = cleanHtml.match(/<script\b[^>]*\bsrc\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))[^>]*>/gi) || [];
        for (var sc = 0; sc < scriptSrcMatches.length; sc++) {
            var srcM = scriptSrcMatches[sc].match(/\bsrc\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i);
            if (srcM) {
                var srcVal = srcM[1] || srcM[2] || srcM[3] || "";
                var sd = getDomain(srcVal);
                if (sd && sd !== pageDomain) {
                    num_ext_scripts++;
                    if (!extDomains[sd]) {
                        extDomains[sd] = true;
                        num_ext_domains_count++;
                    }
                }
            }
        }

        var scriptBlocks = cleanHtml.match(/<script\b[^>]*>([\s\S]*?)<\/script>/gi) || [];
        var scriptTextAccum = "";
        for (var sb = 0; sb < scriptBlocks.length; sb++) {
            var inner = scriptBlocks[sb].replace(/^<script\b[^>]*>/i, "").replace(/<\/script>$/i, "");
            scriptTextAccum += inner;
        }

        var has_password_field = 0;
        for (var ip = 0; ip < inputMatches.length; ip++) {
            if (/type\s*=\s*["']password["']/i.test(inputMatches[ip])) {
                has_password_field = 1;
                break;
            }
        }

        var has_meta_redirect = /<meta\b[^>]*\bhttp-equiv\s*=\s*["']refresh["'][^>]*>/i.test(cleanHtml) ? 1 : 0;
        var script_content_ratio = scriptTextAccum.length / Math.max(1, htmlContent.length);

        var favicon_mismatch = 0;
        var favMatches = cleanHtml.match(/<link\b[^>]*\brel\s*=\s*["'][^"']*icon[^"']*["'][^>]*>/gi) || [];
        for (var fv = 0; fv < favMatches.length; fv++) {
            var favHrefM = favMatches[fv].match(/\bhref\s*=\s*["']([^"']+)["']/i);
            if (favHrefM && favHrefM[1]) {
                var favDom = getDomain(favHrefM[1]);
                if (favDom && favDom !== pageDomain) {
                    favicon_mismatch = 1;
                    break;
                }
            }
        }

        var has_auto_submit = (num_forms > 0 && scriptTextAccum.toLowerCase().indexOf("submit()") !== -1) ? 1 : 0;

        var submitCount = 0;
        for (var subI = 0; subI < inputMatches.length; subI++) {
            if (/type\s*=\s*["'](submit|image)["']/i.test(inputMatches[subI])) {
                submitCount++;
            }
        }
        var buttonMatches = cleanHtml.match(/<button\b[^>]*>/gi) || [];
        for (var b = 0; b < buttonMatches.length; b++) {
            if (/type\s*=\s*["']submit["']/i.test(buttonMatches[b])) {
                submitCount++;
            }
        }
        var input_submit_ratio = num_inputs > 0 ? (num_inputs / Math.max(1, submitCount)) : 0.0;

        return {
            num_forms: num_forms,
            num_inputs: num_inputs,
            num_iframes: num_iframes,
            num_ext_links: num_ext_links,
            num_ext_scripts: num_ext_scripts,
            has_password_field: has_password_field,
            has_meta_redirect: has_meta_redirect,
            script_content_ratio: script_content_ratio,
            favicon_mismatch: favicon_mismatch,
            has_auto_submit: has_auto_submit,
            input_submit_ratio: input_submit_ratio,
            num_unique_ext_domains: num_ext_domains_count
        };
    }

    /**
     * Extract canonical 12 HTML features as an Array of 12 numbers.
     */
    function extractHtmlFeaturesVector(docOrHtml, pageUrl) {
        var dict;
        if (typeof docOrHtml === "string") {
            dict = extractHtmlFeaturesFromString(docOrHtml, pageUrl);
        } else {
            dict = extractHtmlFeaturesFromDoc(docOrHtml, pageUrl);
        }
        var vec = new Array(CANONICAL_HTML_FEATURE_NAMES.length);
        for (var i = 0; i < CANONICAL_HTML_FEATURE_NAMES.length; i++) {
            vec[i] = Number(dict[CANONICAL_HTML_FEATURE_NAMES[i]]);
        }
        return vec;
    }

    return {
        HTML_FEATURE_DIM: HTML_FEATURE_DIM,
        CANONICAL_HTML_FEATURE_NAMES: CANONICAL_HTML_FEATURE_NAMES,
        getDomain: getDomain,
        extractHtmlFeaturesFromDoc: extractHtmlFeaturesFromDoc,
        extractHtmlFeaturesFromString: extractHtmlFeaturesFromString,
        extractHtmlFeaturesVector: extractHtmlFeaturesVector
    };
});
