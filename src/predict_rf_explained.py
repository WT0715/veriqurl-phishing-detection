import re
import joblib
import pandas as pd
from urllib.parse import urlparse
from explanation_rules import (
    extract_model_features,
    build_explanation_result
)


    
def load_model(model_path: str):
    """
    Load the trained Random Forest modelfrom disk.
    """
    return joblib.load(model_path)

def predict_url(url: str, model):
    features = extract_model_features(url)

    feature_df = pd.DataFrame([features])

    prediction = model.predict(feature_df)[0]
    probabilities = model.predict_proba(feature_df)[0]

    benign_probability = probabilities[0]
    phishing_probability = probabilities[1]

    predicted_label = "phishing" if prediction == 1 else "benign"

    explanation = build_explanation_result(url, phishing_probability)

    return {
        "url": url,
        "predicted_label": predicted_label,
        "benign_probability": benign_probability,
        "phishing_probability": phishing_probability,
        "risk_level": explanation["risk_level"],
        "features": features,
        "indicators": explanation["indicators"],
        "reasons": explanation["reasons"],
        "recommendation": explanation["recommendation"],
        "awareness_tip": explanation["awareness_tip"],
    }

def main():
    model_path = "models/random_forest_model.pkl"

    print("Loading trained Random Forest model...")
    model = load_model(model_path)
    print("Model loaded successfully.\n")

    url = input("Enter a URL to check: ").strip()

    result = predict_url(url, model)

    print("\n===== Prediction Result =====")
    print("Model: Random Forest")
    print("URL:", result["url"])
    print("Predicted label:", result["predicted_label"])
    print(f"Benign probability:   {result['benign_probability']:.4f}")
    print(f"Phishing probability: {result['phishing_probability']:.4f}")
    print("Risk level:", result["risk_level"])

    print("\n===== Why It Was Flagged =====")
    for i, reason in enumerate(result["reasons"], start=1):
        print(f"{i}. {reason}")

    print("\n===== Recommendation =====")
    print(result["recommendation"])

    print("\n===== Awareness Tip =====")
    print(result["awareness_tip"])

    print("\n===== RF Model Features =====")
    for key, value in result["features"].items():
        print(f"{key}: {value}")

    print("\n===== Explanation Indicators =====")
    for key, value in result["indicators"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()