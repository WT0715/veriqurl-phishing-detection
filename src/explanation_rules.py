import re
import math
from urllib.parse import urlparse

# ============================================================
# Thresholds for risk levels based on phishing probability
# ============================================================
PHISHING_THRESHOLD = 0.50
HIGH_RISK_THRESHOLD = 0.80

# ============================================================
# 1. Suspicious words used ONLY for model-compatible RF features
# ============================================================
# Keep this list the same as the old RF version to avoid changing
# the feature distribution used by the trained Random Forest model.
MODEL_SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "bank",
    "signin", "confirm", "password", "paypal", "free", "bonus",
    "win", "urgent", "click", "limited", "security", "webscr"
]


# ============================================================
# 2. Expanded suspicious words for explanation only
# ============================================================
# These words are heuristic lexical indicators inspired by URL-based
# phishing detection literature. They are NOT standalone proof of phishing.
EXPLANATION_SUSPICIOUS_WORDS = sorted(set([
    # Account / credential related
    "login", "signin", "logout", "signout", "account", "myaccount",
    "password", "submit", "client", "server",

    # Verification / security / urgency related
    "verify", "verification", "confirm", "update", "secure", "secured",
    "security", "securewebsession", "urgent", "limited", "required",
    "resolution", "suspend", "recovery", "restore", "temporary",

    # Financial / payment / brand-abuse related
    "bank", "banking", "billing", "paypal", "webscr", "ebayisapi",
    "refund", "dispute",

    # Reward / bait related
    "free", "bonus", "win", "lucky", "giveaway", "click",

    # Hosting / platform / web resource related
    "wordpress", "wp", "alibaba", "dropbox", "admin", "includes",
    "themes", "plugins", "content", "site", "images", "js", "css",
    "view", "browser", "review",

    # Free-hosting / suspicious hosting style words from literature
    "000webhostapp", "webhostapp", "webservis", "webspace",
    "webnode", "servico", "redirectme"
]))


# ============================================================
# 3. Brand words used for explanation only
# ============================================================
# This is a small practical list. It is not a full brand database.
# The goal is to explain obvious brand-abuse patterns such as
# secure-paypal-login-example.com or example.com/paypal/login.
BRAND_WORDS = sorted(set([
    "paypal", "google", "microsoft", "apple", "amazon", "facebook",
    "instagram", "whatsapp", "netflix", "dropbox", "wordpress",
    "alibaba", "ebay", "maybank", "cimb", "publicbank", "rhb",
    "bankislam", "hongleong", "tng", "touchngo", "shopee",
    "lazada", "dhl", "fedex"
]))


# ============================================================
# 4. URL shorteners
# ============================================================
SHORTENER_DOMAINS = sorted(set([
    "bit.ly", "bitly.com", "tinyurl.com", "tiny.cc", "goo.gl",
    "t.co", "ow.ly", "is.gd", "buff.ly", "cutt.ly", "rebrand.ly",
    "shorturl.at", "s.id"
]))


# ============================================================
# 5. Common TLD words used to detect TLD-in-path tricks
# ============================================================
COMMON_TLDS = sorted(set([
    "com", "net", "org", "info", "biz", "xyz", "top", "online",
    "site", "club", "shop", "store", "app", "live", "me", "co",
    "my", "id", "uk", "us"
]))


SPECIAL_DELIMITERS = [
    "~", "`", "!", "^", "*", "(", ")", "[", "]", "{", "}",
    "\"", "'", ";", ",", ">", "<", "|"
]

OTHER_DELIMITERS = ["+", "$", "=", "&", ":", "#", "%"]


def normalize_url_for_parsing(url: str) -> str:
    """
    Add http:// if the user enters a URL without a scheme.
    This helps urlparse correctly identify the domain.
    """
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        return "http://" + url
    return url


def get_parsed_url(url: str):
    """
    Return a parsed URL object.
    """
    normalized_url = normalize_url_for_parsing(url)
    return urlparse(normalized_url)


def get_domain(url: str) -> str:
    """
    Extract domain / netloc from URL.
    """
    parsed = get_parsed_url(url)
    domain = parsed.netloc.lower()

    # Remove username/password if present
    if "@" in domain:
        domain = domain.split("@")[-1]

    # Remove port if present
    if ":" in domain:
        domain = domain.split(":")[0]

    return domain


def get_domain_tokens(domain: str) -> list:
    """
    Split domain into tokens using dots and hyphens.
    """
    return [token for token in re.split(r"[.\-]", domain.lower()) if token]


def get_path_tokens(path: str) -> list:
    """
    Split path into tokens using common URL separators.
    """
    return [token for token in re.split(r"[\/_\-.\?=&%+#]", path.lower()) if token]


def get_registered_domain_approx(domain: str) -> str:
    """
    Approximate registered domain using the last two labels.
    Example:
        secure.paypal-login.com -> paypal-login.com

    This is simple and lightweight. It does not require tldextract.
    """
    parts = [p for p in domain.split(".") if p]
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return domain


def count_digits(text: str) -> int:
    return sum(char.isdigit() for char in text)


def count_special_characters(text: str) -> int:
    return sum(not char.isalnum() for char in text)


def has_ip_address(url: str) -> int:
    pattern = r"(?:\d{1,3}\.){3}\d{1,3}"
    return int(bool(re.search(pattern, url)))


def has_email_address(url: str) -> int:
    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    return int(bool(re.search(pattern, url)))


def shannon_entropy(text: str) -> float:
    """
    Calculate Shannon entropy.
    Higher entropy can indicate random-looking or obfuscated strings.
    """
    if not text:
        return 0.0

    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1

    entropy = 0.0
    length = len(text)

    for count in freq.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return entropy


def get_matched_words(text: str, word_list: list) -> list:
    """
    Find matched words inside a URL string.
    """
    text_lower = text.lower()
    return [word for word in word_list if word in text_lower]


def count_suspicious_words_for_model(text: str) -> int:
    """
    Count suspicious words using the old RF-compatible list.
    """
    return len(get_matched_words(text, MODEL_SUSPICIOUS_WORDS))


def get_matched_explanation_words(text: str) -> list:
    """
    Find suspicious words using the expanded explanation list.
    """
    return get_matched_words(text, EXPLANATION_SUSPICIOUS_WORDS)


def get_matched_brand_words(text: str) -> list:
    """
    Find brand-like words inside the URL.
    """
    return get_matched_words(text, BRAND_WORDS)


def is_shortener_domain(domain: str) -> int:
    """
    Check whether the domain is a known URL shortener.
    """
    domain = domain.lower()
    return int(domain in SHORTENER_DOMAINS or any(domain.endswith("." + d) for d in SHORTENER_DOMAINS))


def has_tld_in_path(path: str) -> int:
    """
    Detect patterns like /paypal.com/login or /bank.net/verify.
    """
    path_lower = path.lower()
    for tld in COMMON_TLDS:
        if f".{tld}" in path_lower:
            return 1
    return 0


def has_redirect_like_double_slash(url: str) -> int:
    """
    Detect extra // after the scheme area.
    Example:
        http://example.com//login
        http://example.com/redirect//paypal.com
    """
    parsed = get_parsed_url(url)
    path_and_query = parsed.path + ("?" + parsed.query if parsed.query else "")
    return int("//" in path_and_query)


def has_repeated_characters(text: str, repeat_threshold: int = 4) -> int:
    """
    Detect repeated characters such as aaa, 1111, ----.
    """
    pattern = r"(.)\1{" + str(repeat_threshold - 1) + r",}"
    return int(bool(re.search(pattern, text)))


def brand_in_unusual_host(domain: str, matched_brands: list) -> int:
    """
    Flag brand words in unusual domain structures.

    Example flagged:
        secure-paypal-login.com
        paypal-verification-alert.net

    Example not flagged:
        paypal.com
        www.paypal.com
    """
    if not matched_brands:
        return 0

    registered_domain = get_registered_domain_approx(domain)

    for brand in matched_brands:
        official_pattern_1 = f"{brand}.com"
        official_pattern_2 = f"www.{brand}.com"

        if domain == official_pattern_1 or domain == official_pattern_2:
            continue

        # If brand appears but domain is not simply brand.com, treat as unusual.
        if brand in registered_domain or brand in domain:
            return 1

    return 0


def extract_model_features(url: str) -> dict:
    """
    Extract the original RF-compatible feature set.

    IMPORTANT:
    Use this for the trained Random Forest model.
    Do not add or remove columns here unless you plan to retrain RF.
    """
    parsed = get_parsed_url(url)

    domain = get_domain(url)
    path = parsed.path
    query = parsed.query

    features = {
        "url_length": len(url),
        "domain_length": len(domain),
        "path_length": len(path),
        "query_length": len(query),
        "digit_count": count_digits(url),
        "special_char_count": count_special_characters(url),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "slash_count": url.count("/"),
        "question_mark_count": url.count("?"),
        "equal_count": url.count("="),
        "at_count": url.count("@"),
        "ampersand_count": url.count("&"),
        "underscore_count": url.count("_"),
        "tilde_count": url.count("~"),
        "percent_count": url.count("%"),
        "subdomain_count": max(0, domain.count(".") - 1),
        "has_https": int(parsed.scheme == "https"),
        "has_ip_address": has_ip_address(url),
        "suspicious_word_count": count_suspicious_words_for_model(url),
    }

    return features


def extract_explanation_indicators(url: str) -> dict:
    """
    Extract expanded indicators used for explanations and awareness tips.

    These are not automatically used by RF unless you retrain the model.
    """
    parsed = get_parsed_url(url)

    domain = get_domain(url)
    path = parsed.path
    query = parsed.query
    registered_domain = get_registered_domain_approx(domain)

    domain_tokens = get_domain_tokens(domain)
    path_tokens = get_path_tokens(path)

    matched_suspicious_words = get_matched_explanation_words(url)
    matched_brand_words = get_matched_brand_words(url)

    path_delimiter_count = sum(path.count(ch) for ch in SPECIAL_DELIMITERS + OTHER_DELIMITERS)
    longest_host_token_len = max([len(t) for t in domain_tokens], default=0)
    longest_path_token_len = max([len(t) for t in path_tokens], default=0)

    indicators = {
        "scheme": parsed.scheme,
        "domain": domain,
        "registered_domain": registered_domain,
        "path": path,
        "query": query,

        "url_length": len(url),
        "domain_length": len(domain),
        "path_length": len(path),
        "query_length": len(query),

        "digit_count": count_digits(url),
        "domain_digit_count": count_digits(domain),
        "path_digit_count": count_digits(path),

        "special_char_count": count_special_characters(url),
        "dot_count": url.count("."),
        "domain_dot_count": domain.count("."),
        "path_dot_count": path.count("."),

        "hyphen_count": url.count("-"),
        "domain_hyphen_count": domain.count("-"),
        "path_hyphen_count": path.count("-"),

        "slash_count": url.count("/"),
        "question_mark_count": url.count("?"),
        "equal_count": url.count("="),
        "at_count": url.count("@"),
        "ampersand_count": url.count("&"),
        "underscore_count": url.count("_"),
        "tilde_count": url.count("~"),
        "percent_count": url.count("%"),

        "subdomain_count": max(0, domain.count(".") - 1),
        "has_https": int(parsed.scheme == "https"),
        "has_ip_address": has_ip_address(url),
        "has_email_address": has_email_address(url),
        "has_shortener_domain": is_shortener_domain(domain),
        "has_tld_in_path": has_tld_in_path(path),
        "has_redirect_like_double_slash": has_redirect_like_double_slash(url),
        "has_repeated_characters": has_repeated_characters(url),

        "url_entropy": shannon_entropy(url),
        "path_entropy": shannon_entropy(path),

        "path_delimiter_count": path_delimiter_count,
        "longest_host_token_len": longest_host_token_len,
        "longest_path_token_len": longest_path_token_len,

        "matched_suspicious_words": matched_suspicious_words,
        "suspicious_word_count": len(matched_suspicious_words),

        "matched_brand_words": matched_brand_words,
        "brand_word_count": len(matched_brand_words),
        "brand_in_unusual_host": brand_in_unusual_host(domain, matched_brand_words),
        "brand_in_path": int(any(brand in path.lower() for brand in BRAND_WORDS)),

        "url_to_path_ratio": len(url) / max(1, len(path)),
    }

    return indicators


def get_risk_level(phishing_probability: float) -> str:
    """
    Convert phishing probability into a simple risk level.
    """
    if phishing_probability >= HIGH_RISK_THRESHOLD:
        return "High Risk"
    elif phishing_probability >= PHISHING_THRESHOLD:
        return "Suspicious"
    else:
        return "Low Risk"


def generate_explanations(url: str, indicators: dict) -> list:
    """
    Generate user-friendly explanations from URL indicators.
    """
    reasons = []

    if indicators["has_https"] == 0:
        reasons.append("The URL uses HTTP instead of HTTPS.")

    if indicators["has_ip_address"] == 1:
        reasons.append("The URL contains an IP address instead of a normal domain name.")

    if indicators["has_email_address"] == 1:
        reasons.append("The URL contains an email address, which is unusual and may be used for deception.")

    if indicators["at_count"] > 0:
        reasons.append("The URL contains the '@' symbol, which may hide the real destination domain.")

    if indicators["has_shortener_domain"] == 1:
        reasons.append("The URL uses a known shortening service, which can hide the final destination.")

    if indicators["has_redirect_like_double_slash"] == 1:
        reasons.append("The URL contains an extra double slash in the path, which may indicate redirection or obfuscation.")

    if indicators["brand_in_unusual_host"] == 1:
        brands = ", ".join(indicators["matched_brand_words"][:5])
        reasons.append(
            f"The URL contains brand-related words in an unusual domain structure: {brands}."
        )

    if indicators["brand_in_path"] == 1:
        brands = ", ".join(indicators["matched_brand_words"][:5])
        reasons.append(
            f"The URL contains brand-related words in the path: {brands}."
        )

    if indicators["suspicious_word_count"] > 0:
        words = ", ".join(indicators["matched_suspicious_words"][:8])
        reasons.append(
            f"The URL contains suspicious or suggestive words: {words}."
        )

    if indicators["domain_hyphen_count"] >= 2:
        reasons.append("The domain contains multiple hyphens, which may indicate brand imitation or obfuscation.")

    elif indicators["hyphen_count"] >= 3:
        reasons.append("The URL contains several hyphens, which may be used to imitate a legitimate service.")

    if indicators["digit_count"] >= 6:
        reasons.append("The URL contains many digits, which may make the link look random or deceptive.")

    if indicators["domain_digit_count"] >= 3:
        reasons.append("The domain contains several digits, which is uncommon for many legitimate brand domains.")

    if indicators["url_length"] >= 100:
        reasons.append("The URL is very long, which may hide the real destination or suspicious path.")

    elif indicators["url_length"] >= 60:
        reasons.append("The URL is relatively long, which may indicate a complex or misleading structure.")

    if indicators["path_length"] >= 40:
        reasons.append("The URL has a long path section, which may hide suspicious pages deeper in the link.")

    if indicators["query_length"] >= 40:
        reasons.append("The URL has a long query string, which may contain tracking, redirection, or obfuscated parameters.")

    if indicators["subdomain_count"] >= 2:
        reasons.append("The URL contains multiple subdomains, which may be used to make the link look trustworthy.")

    if indicators["path_dot_count"] >= 2:
        reasons.append("The path contains multiple dots, which may indicate a hidden external domain or file-like deception.")

    if indicators["has_tld_in_path"] == 1:
        reasons.append("The path appears to contain a top-level domain string such as '.com' or '.net', which may be used to imitate another website.")

    if indicators["path_delimiter_count"] >= 3:
        reasons.append("The path or parameters contain several unusual delimiters, which may indicate obfuscation.")

    if indicators["url_entropy"] >= 4.8:
        reasons.append("The URL has high character randomness, which may indicate an automatically generated or obfuscated link.")

    if indicators["path_entropy"] >= 4.2 and indicators["path_length"] >= 20:
        reasons.append("The path section has high randomness, which may indicate an encoded or suspicious path.")

    if indicators["longest_host_token_len"] >= 25:
        reasons.append("The hostname contains an unusually long token, which may be used to hide the real domain.")

    if indicators["longest_path_token_len"] >= 30:
        reasons.append("The path contains an unusually long token, which may be used to hide suspicious content.")

    if indicators["has_repeated_characters"] == 1:
        reasons.append("The URL contains repeated characters, which may indicate an unusual or automatically generated pattern.")

    if not reasons:
        reasons.append("No strong suspicious lexical pattern was detected from the extracted URL indicators.")

    return reasons


def generate_recommendation(risk_level: str) -> str:
    """
    Generate user action recommendation based on risk level.
    """
    if risk_level == "High Risk":
        return "Do not open the link or enter any personal, banking, or login information."
    elif risk_level == "Suspicious":
        return "Verify the domain carefully before proceeding, and avoid entering sensitive information."
    else:
        return "The URL appears low risk based on lexical indicators, but you should still verify the website before trusting it."


def generate_awareness_tip(url: str, indicators: dict) -> str:
    """
    Generate one awareness tip based on the most relevant indicator.
    """
    if indicators["brand_in_unusual_host"] == 1:
        return "A familiar brand name inside a strange domain does not mean the website is official. Always check the real registered domain."

    if indicators["has_shortener_domain"] == 1:
        return "Shortened URLs can hide the final destination. Be careful when the sender or context is not trusted."

    if indicators["suspicious_word_count"] > 0:
        return "Be careful when a URL combines words like login, verify, secure, update, or account with an unusual domain."

    if indicators["has_https"] == 0:
        return "A missing HTTPS connection can be a warning sign, but HTTPS alone does not guarantee that a website is legitimate."

    if indicators["subdomain_count"] >= 2:
        return "Attackers may use long or confusing subdomains to make a URL look related to a trusted brand."

    if indicators["has_ip_address"] == 1:
        return "Legitimate public services usually use readable domain names rather than raw IP addresses."

    if indicators["url_length"] >= 60:
        return "Very long URLs can hide the true destination. Focus on the registered domain before clicking."

    return "Always check the full domain name carefully, not just familiar words appearing inside the URL."


def build_explanation_result(url: str, phishing_probability: float) -> dict:
    """
    Build a complete explanation package for either RF or TCN prediction output.

    Input:
        url
        phishing_probability from model

    Output:
        risk level, reasons, recommendation, awareness tip, indicators
    """
    indicators = extract_explanation_indicators(url)
    risk_level = get_risk_level(phishing_probability)
    reasons = generate_explanations(url, indicators)
    recommendation = generate_recommendation(risk_level)
    awareness_tip = generate_awareness_tip(url, indicators)

    return {
        "risk_level": risk_level,
        "reasons": reasons,
        "recommendation": recommendation,
        "awareness_tip": awareness_tip,
        "indicators": indicators,
    }