import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from explanation_rules import (
    extract_model_features,
    build_explanation_result
)


# =========================
# Paths
# =========================

RF_MODEL_PATH = "models/random_forest_model.pkl"

TCN_DATA_DIR = "data/processed/tcn"
TCN_MODEL_PATH = "models/tcn_model.pt"
VOCAB_PATH = os.path.join(TCN_DATA_DIR, "char_vocab.json")


# =========================
# TCN settings
# Must match train_tcn.py
# =========================

MAX_LEN = 200
EMBED_DIM = 32
NUM_CHANNELS = 64
KERNEL_SIZE = 3
DROPOUT = 0.2


# =========================
# TCN Model Definition
# Must match train_tcn.py
# =========================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_channels, kernel_size, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)

        self.tcn = nn.Sequential(
            TemporalBlock(embed_dim, num_channels, kernel_size, dilation=1, dropout=dropout),
            TemporalBlock(num_channels, num_channels, kernel_size, dilation=2, dropout=dropout),
            TemporalBlock(num_channels, num_channels, kernel_size, dilation=4, dropout=dropout)
        )

        self.fc = nn.Linear(num_channels, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.tcn(x)

        # Global max pooling over sequence length.
        x = torch.max(x, dim=2).values

        x = self.fc(x)
        return x.squeeze(1)


# =========================
# Loading functions
# =========================

def load_rf_model():
    """
    Load trained Random Forest model.
    """
    return joblib.load(RF_MODEL_PATH)


def load_char_vocab():
    """
    Load character vocabulary used by TCN.
    """
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tcn_model(vocab_size: int, device):
    """
    Build TCN architecture and load trained weights.
    """
    model = TCNClassifier(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_channels=NUM_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT
    ).to(device)

    model.load_state_dict(torch.load(TCN_MODEL_PATH, map_location=device))
    model.eval()

    return model


# =========================
# Prediction functions
# =========================

def predict_rf(url: str, rf_model):
    """
    Predict URL using Random Forest.

    RF uses handcrafted lexical features.
    """
    features = extract_model_features(url)
    feature_df = pd.DataFrame([features])

    prediction = rf_model.predict(feature_df)[0]
    probabilities = rf_model.predict_proba(feature_df)[0]

    benign_probability = probabilities[0]
    phishing_probability = probabilities[1]

    predicted_label = "phishing" if prediction == 1 else "benign"
    explanation = build_explanation_result(url, phishing_probability)

    return {
        "model": "Random Forest",
        "predicted_label": predicted_label,
        "benign_probability": benign_probability,
        "phishing_probability": phishing_probability,
        "risk_level": explanation["risk_level"],
        "features": features,
    }


def encode_url_for_tcn(url: str, char_to_idx: dict, max_len: int = 200):
    """
    Convert URL string into fixed-length character index sequence.
    """
    encoded = [char_to_idx.get(ch, 0) for ch in str(url)]

    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        encoded += [0] * (max_len - len(encoded))

    return np.array(encoded, dtype=np.int64)


def predict_tcn(url: str, tcn_model, char_to_idx: dict, device):
    """
    Predict URL using TCN.

    TCN uses character-level sequence input.
    """
    encoded_url = encode_url_for_tcn(url, char_to_idx, MAX_LEN)
    input_tensor = torch.from_numpy(encoded_url).unsqueeze(0).long().to(device)

    with torch.no_grad():
        logit = tcn_model(input_tensor)
        phishing_probability = torch.sigmoid(logit).item()

    benign_probability = 1.0 - phishing_probability
    predicted_label = "phishing" if phishing_probability >= 0.5 else "benign"
    explanation = build_explanation_result(url, phishing_probability)

    return {
        "model": "TCN",
        "predicted_label": predicted_label,
        "benign_probability": benign_probability,
        "phishing_probability": phishing_probability,
        "risk_level": explanation["risk_level"],
        "encoded_sequence_length": len(encoded_url),
        "non_padding_length": int(np.count_nonzero(encoded_url)),
    }


def get_final_decision(rf_result: dict, tcn_result: dict):
    """
    Produce a simple combined decision from RF and TCN.

    This does not replace the models.
    It is only used for demo-level comparison.
    """
    rf_prob = rf_result["phishing_probability"]
    tcn_prob = tcn_result["phishing_probability"]

    avg_prob = (rf_prob + tcn_prob) / 2

    if avg_prob >= 0.80:
        final_risk = "High Risk"
    elif avg_prob >= 0.50:
        final_risk = "Suspicious"
    else:
        final_risk = "Low Risk"

    final_label = "phishing" if avg_prob >= 0.5 else "benign"

    agreement = "Yes" if rf_result["predicted_label"] == tcn_result["predicted_label"] else "No"

    return {
        "average_phishing_probability": avg_prob,
        "final_label": final_label,
        "final_risk": final_risk,
        "models_agree": agreement,
    }


def print_model_result(result: dict):
    """
    Print one model result in a consistent format.
    """
    print(f"\n===== {result['model']} Result =====")
    print("Predicted label:", result["predicted_label"])
    print(f"Benign probability:   {result['benign_probability']:.4f}")
    print(f"Phishing probability: {result['phishing_probability']:.4f}")
    print("Risk level:", result["risk_level"])


def main():
    print("Loading models...")

    rf_model = load_rf_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    char_to_idx = load_char_vocab()
    vocab_size = len(char_to_idx)

    tcn_model = load_tcn_model(vocab_size, device)

    print("Models loaded successfully.")
    print(f"TCN device: {device}\n")

    url = input("Enter a URL to check: ").strip()

    rf_result = predict_rf(url, rf_model)
    tcn_result = predict_tcn(url, tcn_model, char_to_idx, device)

    shared_explanation = build_explanation_result(
        url,
        max(rf_result["phishing_probability"], tcn_result["phishing_probability"])
    )

    final_decision = get_final_decision(rf_result, tcn_result)

    print("\n==============================")
    print(" URL Phishing Detection Result")
    print("==============================")
    print("URL:", url)

    print_model_result(rf_result)
    print_model_result(tcn_result)

    print("\n===== Combined Decision =====")
    print("Models agree:", final_decision["models_agree"])
    print(f"Average phishing probability: {final_decision['average_phishing_probability']:.4f}")
    print("Final label:", final_decision["final_label"])
    print("Final risk level:", final_decision["final_risk"])

    print("\n===== Shared Explanation =====")
    for i, reason in enumerate(shared_explanation["reasons"], start=1):
        print(f"{i}. {reason}")

    print("\n===== Awareness Guidance =====")
    for i, item in enumerate(shared_explanation["awareness_guidance"], start=1):
        print(f"{i}. {item}")

    print("\n===== RF Model Features =====")
    for key, value in rf_result["features"].items():
        print(f"{key}: {value}")

    print("\n===== TCN Sequence Info =====")
    print("Fixed sequence length:", tcn_result["encoded_sequence_length"])
    print("Non-padding character count:", tcn_result["non_padding_length"])


if __name__ == "__main__":
    main()