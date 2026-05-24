import joblib 
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

def load_feature_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def split_features_and_label(df: pd.DataFrame):
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y


def train_random_forest(X, y):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

def save_model(model, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

def main():
    train_path = "data/processed/features/train_features.csv"
    model_path = "models/random_forest_model.pkl"

    print("Loading training data...")
    train_df = load_feature_data(train_path)

    print("Splitting features and labels...")
    X_train, y_train = split_features_and_label(train_df)

    print("Training Random Forest model...")
    model = train_random_forest(X_train, y_train)

    print("Saving model...")
    save_model(model, model_path)

    print(f"Done. Model saved to: {model_path}")


if __name__ == "__main__":
    main()