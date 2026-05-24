# Embedding → TemporalBlock → TemporalBlock → TemporalBlock → Global Max Pooling → Linear

import os
import json
import numpy as np
import torch
import torch.nn as nn

from explanation_rules import build_explanation_result


DATA_DIR = "data/processed/tcn"
MODEL_PATH = "models/tcn_model.pt"
VOCAB_PATH = os.path.join(DATA_DIR, "char_vocab.json")

MAX_LEN = 200
EMBED_DIM = 32
NUM_CHANNELS = 64
KERNEL_SIZE = 3
DROPOUT = 0.2


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
        x = self.embedding(x)          # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)          # (batch, embed_dim, seq_len)
        x = self.tcn(x)                # (batch, num_channels, seq_len)

        # Global max pooling:
        # use the strongest learned URL pattern found anywhere in the sequence.
        x = torch.max(x, dim=2).values # (batch, num_channels)

        x = self.fc(x)                 # (batch, 1)
        return x.squeeze(1)

def load_char_vocab(vocab_path: str):
    """
    Load character-to-index mapping used during TCN training.
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def encode_url(url: str, char_to_idx: dict, max_len: int = 200):
    """
    Convert a URL string into a fixed-length integer sequence.

    Example:
    https://www.google.com -> [12, 25, 25, 19, 24, ..., 0, 0, 0]

    Each character is converted into its training-time index.
    Unknown characters become 0.
    Short URLs are padded with 0.
    Long URLs are truncated to max_len.
    """
    encoded = [char_to_idx.get(ch, 0) for ch in str(url)]

    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        encoded += [0] * (max_len - len(encoded))

    return np.array(encoded, dtype=np.int64)

def load_tcn_model(model_path: str, vocab_size: int, device):
    """
    Build the TCN architecture and load trained weights.
    """
    model = TCNClassifier(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_channels=NUM_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model

def predict_url(url: str, model, char_to_idx: dict, device):
    """
    Predict whether a URL is benign or phishing using the trained TCN model.

    This function:
    1. Encodes the URL into a character sequence.
    2. Runs the TCN model.
    3. Converts the model output into phishing probability.
    4. Uses the shared explanation layer to generate user-friendly explanations.
    """
    encoded_url = encode_url(url, char_to_idx, MAX_LEN)
    input_tensor = torch.from_numpy(encoded_url).unsqueeze(0).long().to(device)

    with torch.no_grad():
        logit = model(input_tensor)
        phishing_probability = torch.sigmoid(logit).item()

    benign_probability = 1.0 - phishing_probability
    predicted_label = "phishing" if phishing_probability >= 0.5 else "benign"

    explanation = build_explanation_result(url, phishing_probability)

    return {
        "url": url,
        "predicted_label": predicted_label,
        "benign_probability": benign_probability,
        "phishing_probability": phishing_probability,
        "risk_level": explanation["risk_level"],
        "reasons": explanation["reasons"],
        "recommendation": explanation["recommendation"],
        "awareness_tip": explanation["awareness_tip"],
        "indicators": explanation["indicators"],
        "encoded_sequence_length": len(encoded_url),
        "non_padding_length": int(np.count_nonzero(encoded_url)),
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading trained TCN model...")
    print(f"Using device: {device}")

    char_to_idx = load_char_vocab(VOCAB_PATH)
    vocab_size = len(char_to_idx)

    model = load_tcn_model(MODEL_PATH, vocab_size, device)

    print("Model loaded successfully.\n")

    url = input("Enter a URL to check: ").strip()

    result = predict_url(url, model, char_to_idx, device)

    print("\n===== Prediction Result =====")
    print("Model: TCN")
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

    print("\n===== TCN Sequence Info =====")
    print("Fixed sequence length:", result["encoded_sequence_length"])
    print("Non-padding character count:", result["non_padding_length"])

    print("\n===== Explanation Indicators =====")
    for key, value in result["indicators"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()