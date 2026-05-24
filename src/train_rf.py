import time
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def load_feature_data(file_path: str) -> pd.DataFrame:
    """
    Load feature-based daataset from CSV.
    """
    df = pd.read_csv(file_path)
    return df

def split_features_and_label(df: pd.DataFrame):
    """
    Seperate input features (x) and target label (y).
    """
    x = df.drop(columns=["label"])
    y = df["label"]
    return x, y

def train_random_forest(x_train, y_train):
    """
    Train a Random Forest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    start_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()

    training_time = end_time - start_time
    return model, training_time

def evaluate_model(model, x, y, dataset_name="Dataset"):
    """
    Evaluate model performance on one dataset.
    """
    start_time = time.time()
    y_pred = model.predict(x)
    end_time = time.time()

    inference_time = end_time - start_time
    per_sample_time = inference_time / len(x)
    per_sample_latency_ms = per_sample_time * 1000

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"===== {dataset_name} Results =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Total inference time: {inference_time:.4f} seconds")
    print(f"Per-sample inference time: {per_sample_latency_ms:.8f} ms")
    print()
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print()

    metrics = {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "inference_time": inference_time,
        "per_sample_time": per_sample_time,
        "per_sample_latency_ms": per_sample_latency_ms
    }

    return metrics

def save_metrics(metrics_list, output_path: str):
    """
    Save evaluation metrics into a CSV file.
    """
    df_metrics = pd.DataFrame(metrics_list)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(output_path, index=False)

def main():
    train_path = "data/processed/features/train_features.csv"
    val_path = "data/processed/features/val_features.csv"
    test_path = "data/processed/features/test_features.csv"

    print("Loading feature datasets...")
    train_df = load_feature_data(train_path)
    val_df = load_feature_data(val_path)
    test_df = load_feature_data(test_path)

    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)
    print()

    print("Splitting features and labels...")
    X_train, y_train = split_features_and_label(train_df)
    X_val, y_val = split_features_and_label(val_df)
    X_test, y_test = split_features_and_label(test_df)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print()

    print("Training Random Forest model...")
    model, training_time = train_random_forest(X_train, y_train)
    print(f"Training completed in {training_time:.4f} seconds")
    print()

    metrics_list = []

    val_metrics = evaluate_model(model, X_val, y_val, dataset_name="Validation")
    val_metrics["training_time"] = training_time
    metrics_list.append(val_metrics)

    test_metrics = evaluate_model(model, X_test, y_test, dataset_name="Test")
    test_metrics["training_time"] = training_time
    metrics_list.append(test_metrics)

    output_metrics_path = "results/rf_metrics.csv"
    save_metrics(metrics_list, output_metrics_path)

    print(f"Metrics saved to: {output_metrics_path}")

if __name__ == "__main__":
    main()