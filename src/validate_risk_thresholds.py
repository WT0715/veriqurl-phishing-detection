import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Allow importing project modules when running from project root
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from predict_compare import (
    load_rf_model,
    load_char_vocab,
    load_tcn_model,
)


# =========================
# Paths
# =========================

VAL_FEATURES_PATH = ROOT_DIR / "data" / "processed" / "features" / "val_features.csv"
VAL_X_PATH = ROOT_DIR / "data" / "processed" / "tcn" / "val_X.npy"
VAL_Y_PATH = ROOT_DIR / "data" / "processed" / "tcn" / "val_y.npy"

RESULTS_DIR = ROOT_DIR / "results"
THRESHOLD_OUTPUT_PATH = RESULTS_DIR / "risk_threshold_validation.csv"
DISTRIBUTION_OUTPUT_PATH = RESULTS_DIR / "risk_threshold_distribution.csv"


# =========================
# Settings
# =========================

CANDIDATE_HIGH_RISK_THRESHOLDS = [0.60, 0.70, 0.80, 0.90]
BATCH_SIZE = 512


def get_rf_validation_probabilities(rf_model):
    """
    Get Random Forest phishing probabilities on the validation set.

    The RF model uses handcrafted URL features.
    """
    val_features = pd.read_csv(VAL_FEATURES_PATH)

    if "label" not in val_features.columns:
        raise ValueError("val_features.csv must contain a 'label' column.")

    y_true = val_features["label"].astype(int).values
    X_val = val_features.drop(columns=["label"])

    rf_probs = rf_model.predict_proba(X_val)[:, 1]

    return rf_probs, y_true


def get_tcn_validation_probabilities(tcn_model, device):
    """
    Get TCN phishing probabilities on the validation set.

    The TCN model uses encoded character-level URL sequences.
    """
    X_val = np.load(VAL_X_PATH)
    y_true = np.load(VAL_Y_PATH).astype(int)

    all_probs = []

    tcn_model.eval()

    with torch.no_grad():
        for start_idx in range(0, len(X_val), BATCH_SIZE):
            batch = X_val[start_idx:start_idx + BATCH_SIZE]
            batch_tensor = torch.tensor(batch, dtype=torch.long).to(device)

            logits = tcn_model(batch_tensor)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

            all_probs.extend(probs)

    return np.array(all_probs), y_true


def evaluate_thresholds(y_true, combined_probs):
    """
    Evaluate candidate High Risk thresholds.

    Binary phishing decision remains fixed at 0.50.
    High Risk threshold is tested separately to decide when a phishing
    prediction becomes a stronger high-confidence warning.
    """
    rows = []
    distribution_rows = []

    base_binary_pred = (combined_probs >= 0.50).astype(int)

    base_accuracy = accuracy_score(y_true, base_binary_pred)
    base_precision = precision_score(y_true, base_binary_pred, zero_division=0)
    base_recall = recall_score(y_true, base_binary_pred, zero_division=0)
    base_f1 = f1_score(y_true, base_binary_pred, zero_division=0)

    total_samples = len(y_true)
    total_phishing = int(np.sum(y_true == 1))
    total_benign = int(np.sum(y_true == 0))

    for threshold in CANDIDATE_HIGH_RISK_THRESHOLDS:
        high_risk_mask = combined_probs >= threshold
        suspicious_mask = (combined_probs >= 0.50) & (combined_probs < threshold)
        low_risk_mask = combined_probs < 0.50

        high_risk_count = int(np.sum(high_risk_mask))
        suspicious_count = int(np.sum(suspicious_mask))
        low_risk_count = int(np.sum(low_risk_mask))

        high_risk_true_phishing = int(np.sum((high_risk_mask) & (y_true == 1)))
        high_risk_false_positive = int(np.sum((high_risk_mask) & (y_true == 0)))

        suspicious_true_phishing = int(np.sum((suspicious_mask) & (y_true == 1)))
        suspicious_benign = int(np.sum((suspicious_mask) & (y_true == 0)))

        low_risk_phishing = int(np.sum((low_risk_mask) & (y_true == 1)))
        low_risk_benign = int(np.sum((low_risk_mask) & (y_true == 0)))

        if high_risk_count > 0:
            high_risk_precision = high_risk_true_phishing / high_risk_count
        else:
            high_risk_precision = 0.0

        if total_phishing > 0:
            high_risk_recall = high_risk_true_phishing / total_phishing
        else:
            high_risk_recall = 0.0

        rows.append({
            "candidate_high_risk_threshold": threshold,
            "base_binary_accuracy_at_0_50": base_accuracy,
            "base_binary_precision_at_0_50": base_precision,
            "base_binary_recall_at_0_50": base_recall,
            "base_binary_f1_at_0_50": base_f1,
            "high_risk_precision": high_risk_precision,
            "high_risk_recall": high_risk_recall,
            "high_risk_count": high_risk_count,
            "high_risk_true_phishing": high_risk_true_phishing,
            "high_risk_false_positive": high_risk_false_positive,
            "suspicious_count": suspicious_count,
            "suspicious_true_phishing": suspicious_true_phishing,
            "suspicious_benign": suspicious_benign,
            "low_risk_count": low_risk_count,
            "low_risk_phishing": low_risk_phishing,
            "low_risk_benign": low_risk_benign,
            "total_samples": total_samples,
            "total_phishing": total_phishing,
            "total_benign": total_benign,
        })

        distribution_rows.append({
            "candidate_high_risk_threshold": threshold,
            "low_risk_count": low_risk_count,
            "suspicious_count": suspicious_count,
            "high_risk_count": high_risk_count,
            "low_risk_phishing": low_risk_phishing,
            "suspicious_true_phishing": suspicious_true_phishing,
            "high_risk_true_phishing": high_risk_true_phishing,
            "high_risk_false_positive": high_risk_false_positive,
        })

    return pd.DataFrame(rows), pd.DataFrame(distribution_rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    rf_model = load_rf_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    char_to_idx = load_char_vocab()
    tcn_model = load_tcn_model(len(char_to_idx), device)

    print("Getting RF validation probabilities...")
    rf_probs, y_rf = get_rf_validation_probabilities(rf_model)

    print("Getting TCN validation probabilities...")
    tcn_probs, y_tcn = get_tcn_validation_probabilities(tcn_model, device)

    if len(y_rf) != len(y_tcn):
        raise ValueError("RF validation labels and TCN validation labels have different lengths.")

    if not np.array_equal(y_rf, y_tcn):
        raise ValueError(
            "RF validation labels and TCN validation labels are not in the same order. "
            "Please check val_features.csv and val_y.npy."
        )

    y_true = y_rf

    combined_probs = (rf_probs + tcn_probs) / 2

    threshold_df, distribution_df = evaluate_thresholds(y_true, combined_probs)

    threshold_df.to_csv(THRESHOLD_OUTPUT_PATH, index=False)
    distribution_df.to_csv(DISTRIBUTION_OUTPUT_PATH, index=False)

    print("\n=== Risk Threshold Validation Results ===")
    print(threshold_df.to_string(index=False))

    print("\nSaved threshold validation to:")
    print(THRESHOLD_OUTPUT_PATH)

    print("\nSaved risk distribution to:")
    print(DISTRIBUTION_OUTPUT_PATH)


if __name__ == "__main__":
    main()