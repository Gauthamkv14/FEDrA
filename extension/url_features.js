/**
 * extension/url_features.js
 * =========================
 * Browser-compatible implementation of the canonical 22 URL features for FEDrA.
 * 
 * Guarantees 100% semantic parity with `scripts/url_features.py` (Schema v1).
 * Operates purely on the URL string with zero network/DNS/WHOIS lookups.
 * 
 * Exposes:
 *   - CANONICAL_URL_FEATURE_NAMES (Array of 22 feature names in exact order)
 *   - extractCanonicalUrlFeaturesDict(url) -> Object
 *   - extractCanonicalUrlFeaturesVector(url) -> Array (length 22)
 */

(function (root, factory) {
    if (typeof define === "function" && define.amd) {
        define([], factory);
    } else if (typeof module === "object" && module.exports) {
        module.exports = factory();
    } else {
        root.FEDrA_URL = factory();
    }
})(typeof self !== "undefined" ? self : this, function () {
    "use strict";

    var URL_SCHEMA_VERSION = "v1";
    var URL_FEATURE_DIM = 22;

    var CANONICAL_URL_FEATURE_NAMES = [
        "url_len",
        "num_subdomains",
        "has_ip",
        "is_https",
        "count_at",
        "count_dash",
        "count_double_slash",
        "domain_entropy",
        "num_params",
        "has_suspicious_params",
        "brand_in_domain",
        "brand_in_subdomain",
        "has_typosquat",
        "has_punycode",
        "excessive_subdomains",
        "is_suspicious_tld",
        "free_hosting",
        "is_shortener",
        "excessive_hyphens",
        "has_nonstandard_port",
        "high_entropy",
        "long_url"
    ];

    var SUSPICIOUS_QUERY_WORDS = ["login", "redirect", "verify", "secure"];

    var SUSPICIOUS_TLDS = {
        "xyz": true, "tk": true, "ml": true, "ga": true, "cf": true,
        "gq": true, "top": true, "click": true, "work": true, "loan": true,
        "men": true, "date": true, "racing": true, "party": true, "trade": true,
        "kim": true, "country": true, "stream": true, "download": true, "gdn": true,
        "bid": true, "accountant": true, "faith": true, "review": true,
        "science": true, "win": true
    };

    var FREE_HOSTING_DOMAINS = [
        "000webhostapp.com", "weebly.com", "wixsite.com", "wordpress.com",
        "blogspot.com", "netlify.app", "github.io", "glitch.me",
        "firebaseapp.com", "web.app", "surge.sh", "pages.dev"
    ];

    var URL_SHORTENERS = [
        "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
        "buff.ly", "is.gd", "short.io", "rebrand.ly"
    ];

    var BRAND_KEYWORDS = [
        "paypal", "amazon", "apple", "google", "microsoft", "facebook",
        "instagram", "netflix", "dropbox", "linkedin", "twitter", "ebay",
        "wellsfargo", "chase", "citibank", "bankofamerica", "irs",
        "dhl", "fedex", "usps", "whatsapp", "telegram"
    ];

    var TYPO_PATTERNS = [
        ["0", "o"], ["1", "l"], ["3", "e"], ["4", "a"], ["5", "s"]
    ];

    // Standard two-part ccTLD suffixes matching Public Suffix List / tldextract
    var MULTI_PART_TLDS = {
        "ac.in": true, "ac.uk": true, "co.id": true, "co.il": true, "co.in": true,
        "co.jp": true, "co.kr": true, "co.nz": true, "co.th": true, "co.uk": true,
        "co.za": true, "co.zw": true, "com.ar": true, "com.au": true, "com.br": true,
        "com.cn": true, "com.co": true, "com.do": true, "com.eg": true, "com.es": true,
        "com.gr": true, "com.hk": true, "com.hr": true, "com.my": true, "com.mu": true,
        "com.mx": true, "com.ng": true, "com.np": true, "com.pe": true, "com.ph": true,
        "com.pk": true, "com.pl": true, "com.pt": true, "com.ro": true, "com.ru": true,
        "com.sa": true, "com.sg": true, "com.tr": true, "com.tw": true, "com.ua": true,
        "com.uy": true, "com.ve": true, "com.vn": true, "edu.au": true, "edu.cn": true,
        "edu.hk": true, "edu.in": true, "edu.my": true, "edu.ng": true, "edu.pk": true,
        "edu.sg": true, "edu.tw": true, "gov.au": true, "gov.br": true, "gov.cn": true,
        "gov.hk": true, "gov.in": true, "gov.ng": true, "gov.pk": true, "gov.sg": true,
        "gov.uk": true, "gov.za": true, "net.au": true, "net.br": true, "net.cn": true,
        "net.hk": true, "net.in": true, "net.my": true, "net.nz": true, "net.pk": true,
        "net.ru": true, "net.sg": true, "net.tr": true, "net.tw": true, "net.ua": true,
        "net.za": true, "org.au": true, "org.br": true, "org.cn": true, "org.hk": true,
        "org.in": true, "org.my": true, "org.nz": true, "org.pk": true, "org.ru": true,
        "org.sg": true, "org.tr": true, "org.tw": true, "org.ua": true, "org.uk": true,
        "org.ve": true, "org.za": true, "tec.br": true, "web.id": true
    };

    /**
     * Compute Shannon entropy of a string (base 2).
     */
    function shannonEntropy(s) {
        if (!s || typeof s !== "string" || s.length === 0) {
            return 0.0;
        }
        var counts = {};
        var len = s.length;
        for (var i = 0; i < len; i++) {
            var c = s[i];
            counts[c] = (counts[c] || 0) + 1;
        }
        var entropy = 0.0;
        for (var ch in counts) {
            if (Object.prototype.hasOwnProperty.call(counts, ch)) {
                var p = counts[ch] / len;
                entropy -= p * Math.log2(p);
            }
        }
        return entropy;
    }

    /**
     * Parse URL into hostname, port, pathname, search matching urllib.parse.urlparse.
     */
    function parseUrlComponents(rawUrl) {
        var urlStr = rawUrl;
        if (urlStr.indexOf("://") === -1) {
            urlStr = "http://" + urlStr;
        }
        var hostname = "";
        var port = null;
        var query = "";

        try {
            var u = new URL(urlStr);
            hostname = u.hostname || "";
            if (u.port) {
                port = parseInt(u.port, 10);
            }
            query = u.search ? u.search.substring(1) : "";
        } catch (e) {
            var hostMatch = urlStr.match(/^https?:\/\/([^/?#:]+)(?::(\d+))?([^?#]*)(?:\?([^#]*))?/i);
            if (hostMatch) {
                hostname = hostMatch[1] || "";
                port = hostMatch[2] ? parseInt(hostMatch[2], 10) : null;
                query = hostMatch[4] || "";
            }
        }
        return { hostname: hostname, port: port, query: query };
    }

    /**
     * Extract (subdomain, domain, suffix) matching Python tldextract semantics.
     */
    function extractTldParts(rawUrl) {
        var parsed = parseUrlComponents(rawUrl);
        var hostname = (parsed.hostname || "").toLowerCase();

        if (!hostname) {
            return { subdomain: "", domain: "", suffix: "", hostname: "" };
        }

        // IPv4 check: if host is raw IP, domain is the IP, suffix and subdomain are empty
        var ipMatch = hostname.match(/^(\d{1,3}\.){3}\d{1,3}$/);
        if (ipMatch) {
            return { subdomain: "", domain: hostname, suffix: "", hostname: hostname };
        }

        var parts = hostname.split(".");
        var subdomain = "";
        var domain = "";
        var suffix = "";

        if (parts.length >= 3) {
            var lastTwo = parts.slice(-2).join(".");
            if (MULTI_PART_TLDS[lastTwo]) {
                suffix = lastTwo;
                domain = parts[parts.length - 3] || "";
                subdomain = parts.slice(0, -3).join(".");
                return { subdomain: subdomain, domain: domain, suffix: suffix, hostname: hostname };
            }
        }

        if (parts.length >= 2) {
            suffix = parts[parts.length - 1];
            domain = parts[parts.length - 2];
            subdomain = parts.slice(0, -2).join(".");
        } else {
            domain = hostname;
            suffix = "";
            subdomain = "";
        }

        return { subdomain: subdomain, domain: domain, suffix: suffix, hostname: hostname };
    }

    /**
     * Count non-overlapping occurrences of substring in string.
     */
    function countOccurrences(str, sub) {
        if (!str || !sub) return 0;
        var count = 0;
        var pos = 0;
        while ((pos = str.indexOf(sub, pos)) !== -1) {
            count++;
            pos += sub.length;
        }
        return count;
    }

    /**
     * Extract canonical 22 URL features as an Object dictionary.
     */
    function extractCanonicalUrlFeaturesDict(url) {
        if (!url || typeof url !== "string") {
            var empty = {};
            for (var i = 0; i < CANONICAL_URL_FEATURE_NAMES.length; i++) {
                empty[CANONICAL_URL_FEATURE_NAMES[i]] = 0;
            }
            return empty;
        }

        var parsedComp = parseUrlComponents(url);
        var tld = extractTldParts(url);
        var hostname = tld.hostname;
        var domain = tld.domain;
        var subdomain = tld.subdomain;
        var suffix = tld.suffix;

        var fullHost = hostname.toLowerCase();
        var domainLower = domain.toLowerCase();
        var subdomainLower = subdomain.toLowerCase();

        // 0. URL length
        var url_len = url.length;

        // 1. Number of subdomains
        var num_subdomains = 0;
        if (subdomain) {
            var subParts = subdomain.split(".");
            for (var sp = 0; sp < subParts.length; sp++) {
                if (subParts[sp]) num_subdomains++;
            }
        }

        // 2. IP address in domain
        var has_ip = (/^(\d{1,3}\.){3}\d{1,3}$/).test(domain) ? 1 : 0;

        // 3. HTTPS scheme
        var is_https = url.toLowerCase().startsWith("https") ? 1 : 0;

        // 4. Count of '@'
        var count_at = countOccurrences(url, "@");

        // 5. Count of '-'
        var count_dash = countOccurrences(url, "-");

        // 6. Count of '//' (excluding protocol)
        var count_double_slash = url.indexOf("://") !== -1 ?
            Math.max(0, countOccurrences(url, "//") - 1) : countOccurrences(url, "//");

        // 7. Domain entropy (domain + suffix token)
        var domainToken = suffix ? (domain + "." + suffix) : domain;
        var domain_entropy = shannonEntropy(domainToken);

        // 8. Query parameter count & 9. Suspicious query parameters (matching urllib.parse.parse_qsl)
        var num_params = 0;
        var has_suspicious_params = 0;
        if (parsedComp.query && parsedComp.query.length > 0) {
            var pairs = parsedComp.query.split("&");
            for (var p = 0; p < pairs.length; p++) {
                if (pairs[p]) {
                    var eqIdx = pairs[p].indexOf("=");
                    if (eqIdx !== -1) {
                        var rawKey = pairs[p].substring(0, eqIdx);
                        var rawVal = pairs[p].substring(eqIdx + 1);
                        // urllib.parse.parse_qsl default keep_blank_values=False ignores empty values (e.g. key=)
                        if (rawVal.length > 0) {
                            num_params++;
                            var key = "";
                            try { key = decodeURIComponent(rawKey).toLowerCase(); } catch (e) { key = rawKey.toLowerCase(); }
                            for (var s = 0; s < SUSPICIOUS_QUERY_WORDS.length; s++) {
                                if (key.indexOf(SUSPICIOUS_QUERY_WORDS[s]) !== -1) {
                                    has_suspicious_params = 1;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // 10. Brand keyword in domain (impersonation check)
        var brand_in_domain = 0;
        for (var b = 0; b < BRAND_KEYWORDS.length; b++) {
            var bk = BRAND_KEYWORDS[b];
            if (domainLower.indexOf(bk) !== -1 && domainLower !== bk) {
                brand_in_domain = 1;
                break;
            }
        }

        // 11. Brand keyword in subdomain
        var brand_in_subdomain = 0;
        for (var bsub = 0; bsub < BRAND_KEYWORDS.length; bsub++) {
            if (subdomainLower.indexOf(BRAND_KEYWORDS[bsub]) !== -1) {
                brand_in_subdomain = 1;
                break;
            }
        }

        // 12. Typosquatting (exact Python logic: b.replace(orig, rep) == domainLower)
        var has_typosquat = 0;
        for (var bt = 0; bt < BRAND_KEYWORDS.length; bt++) {
            var brandStr = BRAND_KEYWORDS[bt];
            for (var tp = 0; tp < TYPO_PATTERNS.length; tp++) {
                var orig = TYPO_PATTERNS[tp][0];
                var rep = TYPO_PATTERNS[tp][1];
                if (brandStr.split(orig).join(rep) === domainLower) {
                    has_typosquat = 1;
                    break;
                }
            }
            if (has_typosquat) break;
        }

        // 13. Punycode / IDN homograph
        var has_punycode = fullHost.indexOf("xn--") !== -1 ? 1 : 0;

        // 14. Excessive subdomains (> 3 levels)
        var excessive_subdomains = num_subdomains > 3 ? 1 : 0;

        // 15. Suspicious TLD
        var is_suspicious_tld = (suffix && SUSPICIOUS_TLDS[suffix.toLowerCase()]) ? 1 : 0;

        // 16. Free hosting platform
        var free_hosting = 0;
        for (var fh = 0; fh < FREE_HOSTING_DOMAINS.length; fh++) {
            if (fullHost.indexOf(FREE_HOSTING_DOMAINS[fh]) !== -1) {
                free_hosting = 1;
                break;
            }
        }

        // 17. URL shortener service
        var is_shortener = 0;
        for (var us = 0; us < URL_SHORTENERS.length; us++) {
            if (fullHost.indexOf(URL_SHORTENERS[us]) !== -1) {
                is_shortener = 1;
                break;
            }
        }

        // 18. Excessive hyphens (>= 3 in domain)
        var excessive_hyphens = countOccurrences(domain, "-") >= 3 ? 1 : 0;

        // 19. Non-standard port
        var port = parsedComp.port;
        var has_nonstandard_port = (port !== null && port !== 80 && port !== 443 && port !== 8080) ? 1 : 0;

        // 20. High entropy threshold (> 3.8)
        var high_entropy = domain_entropy > 3.8 ? 1 : 0;

        // 21. Long URL threshold (> 100 characters)
        var long_url = url_len > 100 ? 1 : 0;

        return {
            url_len: url_len,
            num_subdomains: num_subdomains,
            has_ip: has_ip,
            is_https: is_https,
            count_at: count_at,
            count_dash: count_dash,
            count_double_slash: count_double_slash,
            domain_entropy: domain_entropy,
            num_params: num_params,
            has_suspicious_params: has_suspicious_params,
            brand_in_domain: brand_in_domain,
            brand_in_subdomain: brand_in_subdomain,
            has_typosquat: has_typosquat,
            has_punycode: has_punycode,
            excessive_subdomains: excessive_subdomains,
            is_suspicious_tld: is_suspicious_tld,
            free_hosting: free_hosting,
            is_shortener: is_shortener,
            excessive_hyphens: excessive_hyphens,
            has_nonstandard_port: has_nonstandard_port,
            high_entropy: high_entropy,
            long_url: long_url,
            // Metadata
            _domain: domain,
            _subdomain: subdomain,
            _suffix: suffix,
            _hostname: hostname
        };
    }

    /**
     * Extract canonical 22 URL features as an Array of 22 numbers.
     */
    function extractCanonicalUrlFeaturesVector(url) {
        var dict = extractCanonicalUrlFeaturesDict(url);
        var vec = new Array(CANONICAL_URL_FEATURE_NAMES.length);
        for (var i = 0; i < CANONICAL_URL_FEATURE_NAMES.length; i++) {
            vec[i] = Number(dict[CANONICAL_URL_FEATURE_NAMES[i]]);
        }
        return vec;
    }

    return {
        URL_SCHEMA_VERSION: URL_SCHEMA_VERSION,
        URL_FEATURE_DIM: URL_FEATURE_DIM,
        CANONICAL_URL_FEATURE_NAMES: CANONICAL_URL_FEATURE_NAMES,
        shannonEntropy: shannonEntropy,
        extractTldParts: extractTldParts,
        extractCanonicalUrlFeaturesDict: extractCanonicalUrlFeaturesDict,
        extractCanonicalUrlFeaturesVector: extractCanonicalUrlFeaturesVector
    };
});
