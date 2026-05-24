import os
import pandas as pd

RESULTS_DIR = "results"

RF_METRICS_PATH = os.path.join(RESULTS_DIR, "rf_metrics.csv")
TCN_METRICS_PATH = os.path.join(RESULTS_DIR, "tcn_metrics.csv")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "model_comparison.csv")

def normalize_columns(df):
    df = df.copy()
    df.columns = [col.strip().lower() for col in df.columns]

    rename_map = {
        "f1": "f1_score",
        "f1-score": "f1_score",
        "f1 score": "f1_score",
        "model_name": "model",
        "training_time": "training_time_s",
        "inference_time": "testing_time_s",
        "per_sample_time": "per_sample_time_s",
    }

    df = df.rename(columns=rename_map)
    return df

def add_lightweight_metrics(df):
    df = df.copy()

    if "per_sample_time_s" in df.columns and "per_sample_latency_ms" not in df.columns:
        df["per_sample_latency_ms"] = df["per_sample_time_s"] * 1000

    if "accuracy" in df.columns and "training_time_s" in df.columns:
        df["tes"] = df.apply(
            lambda row: row["accuracy"] / row["training_time_s"]
            if pd.notnull(row["training_time_s"]) and row["training_time_s"] > 0
            else None,
            axis=1
        )

    if "accuracy" in df.columns and "testing_time_s" in df.columns:
        df["ies"] = df.apply(
            lambda row: row["accuracy"] / row["testing_time_s"]
            if pd.notnull(row["testing_time_s"]) and row["testing_time_s"] > 0
            else None,
            axis=1
        )

    if "accuracy" in df.columns and "per_sample_latency_ms" in df.columns:
        df["rtde"] = df.apply(
            lambda row: row["accuracy"] / row["per_sample_latency_ms"]
            if pd.notnull(row["per_sample_latency_ms"]) and row["per_sample_latency_ms"] > 0
            else None,
            axis=1
        )

    return df

def main():
    if not os.path.exists(RF_METRICS_PATH):
        raise FileNotFoundError(f"Cannot find {RF_METRICS_PATH}")

    if not os.path.exists(TCN_METRICS_PATH):
        raise FileNotFoundError(f"Cannot find {TCN_METRICS_PATH}")

    rf_df = pd.read_csv(RF_METRICS_PATH)
    tcn_df = pd.read_csv(TCN_METRICS_PATH)

    rf_df = normalize_columns(rf_df)
    tcn_df = normalize_columns(tcn_df)

    # Use Test set only for fair final comparison
    if "dataset" in rf_df.columns:
        rf_df = rf_df[rf_df["dataset"].str.lower() == "test"]

    if "dataset" in tcn_df.columns:
        tcn_df = tcn_df[tcn_df["dataset"].str.lower() == "test"]

    if "model" not in rf_df.columns:
        rf_df.insert(0, "model", "Random Forest")

    if "model" not in tcn_df.columns:
        tcn_df.insert(0, "model", "TCN")

    rf_df = add_lightweight_metrics(rf_df)
    tcn_df = add_lightweight_metrics(tcn_df)

    keep_cols = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "training_time_s",
        "testing_time_s",
        "per_sample_time_s",
        "per_sample_latency_ms",
        "tes",
        "ies",
        "rtde",
    ]

    optional_cols = [
        "model_size_mb",
        "num_parameters",
        "pred_positive_rate",
    ]

    for col in optional_cols:
        if col in rf_df.columns or col in tcn_df.columns:
            if col not in rf_df.columns:
                rf_df[col] = None
            if col not in tcn_df.columns:
                tcn_df[col] = None
            keep_cols.append(col)

    comparison_df = pd.concat(
        [rf_df[keep_cols], tcn_df[keep_cols]],
        ignore_index=True
    )

    comparison_df.to_csv(OUTPUT_PATH, index=False)

    print("\n===== Final Model Comparison with Lightweight Metrics =====")
    print(comparison_df)

    print(f"\nSaved comparison table to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()