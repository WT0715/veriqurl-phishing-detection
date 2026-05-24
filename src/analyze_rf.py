import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

def load_feature_data(file_path: str) -> pd.DataFrame:
    """
    Load feature dataset from CSV
    """
    return pd.read_csv(file_path)

def split_features_and_label(df: pd.DataFrame):
    """
    Seperate X and y
    """
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y

def train_random_forest(X, y):
    """
    Train Random Forest model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

def get_feature_importance(model, feature_names):
    """
    Create a dataframe of feature importance scores.
    """
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    ).reset_index(drop=True)

    return importance_df

def save_feature_importance(df: pd.DataFrame, output_path: str):
    """
    Save feature importance results
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

def main():
    train_path = "data/processed/features/train_features.csv"
    output_path = "results/rf_feature_importance.csv"

    print("Loading training feature dataset...")
    train_df = load_feature_data(train_path)
    print("Train shape: ", train_df.shape)
    print()

    print("Splitting features and label...")
    X_train, y_train = split_features_and_label(train_df)
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print()

    print("Training Random Forest model...")
    model = train_random_forest(X_train, y_train)
    print("Training done.")
    print()

    print("Calculating feature importance...")
    importance_df = get_feature_importance(model, X_train.columns)

    print("Top 10 most important features: ")
    print(importance_df.head(10))
    print()

    save_feature_importance(importance_df, output_path)
    print(f"Feature importance saved to: {output_path}")

if __name__ == "__main__":
    main()