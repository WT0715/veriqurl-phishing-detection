import re
import joblib
import pandas as pd
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "bank",
    "signin", "confirm", "password", "paypal", "free", "bonus",
    "win", "urgent", "click", "limited", "security", "webscr"
]

def count_digits(text: str) -> int:
    return sum(char.isdigit() for char in text)

def count_special_characters(text: str) -> int:
    return sum(not char.isalnum() for char in text)

def count_suspicious_words(text: str) -> int:
    text_lower = text.lower()
    return sum(word in text_lower for word in SUSPICIOUS_WORDS)

def has_ip_address(url: str) -> int:
    pattern = r"(?:\d{1,3}\.){3}\d{1,3}"
    return int(bool(re.search(pattern, url)))

def extract_url_features(url: str) -> dict:
    parsed = urlparse(url)

    domain = parsed.netloc
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
        "suspicious_word_count": count_suspicious_words(url),
    }

    return features

def get_risk_level(phishing_probability: float) -> str:
    """
    Convert phishing probability into a simple risk level
    """
    if phishing_probability >= 0.80:
        return "High Risk"
    elif phishing_probability >= 0.50:
        return "Suspicious"
    else:
        return "Low Risk"
    
def load_model(model_path: str):
    return joblib.load(model_path)

def predict_url(url: str, model):
    features = extract_url_features(url)

    feature_df = pd.DataFrame([features])

    prediction = model.predict(feature_df)[0]
    probabilities = model.predict_proba(feature_df)[0]

    benign_probability = probabilities[0]
    phishing_probability = probabilities[1]

    predicted_label = "phishing" if prediction == 1 else "benign"
    risk_level = get_risk_level(phishing_probability)

    return {
        "url": url,
        "predicted_label": predicted_label,
        "benign_probability": benign_probability,
        "phishing_probability": phishing_probability,
        "risk_level": risk_level,
        "features": features
    }

def main():
    model_path = "models/random_forest_model.pkl"

    print("Loading trained Random Forest model...")
    model = load_model(model_path)
    print("Model loaded successfully.\n")

    url = input("Enter a URL to check: ").strip()

    result = predict_url(url, model)

    print("\n===== Prediction Result =====")
    print("URL:", result["url"])
    print("Predicted label:", result["predicted_label"])
    print(f"Benign probability:   {result['benign_probability']:.4f}")
    print(f"Phishing probability: {result['phishing_probability']:.4f}")
    print("Risk level:", result["risk_level"])

    print("\n===== Extracted Features =====")
    for key, value in result["features"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()