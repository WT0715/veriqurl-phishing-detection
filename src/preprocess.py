import pandas as pd
from pathlib import Path

def load_kaggle_data(file_path: str) -> pd.DataFrame:
    """
    Read the raw Kaggle phishing datraset.
    """
    df = pd.read_csv(file_path)
    return df

def select_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns needed for model training:
    - URL
    - label
    """
    df = df[["URL", "label"]].copy()
    return df

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns into a consistent format.
    """
    df = df.rename(columns={"URL": "url"})
    return df

def convert_label_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert text labels into numeic labels.

    Mapping:
    - benign -> 0
    - phishing -> 1
    """
    label_mapping = {
        "benign": 0,
        "phishing": 1
    }

    df["label"] = df["label"].map(label_mapping)
    return df

def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing url or label
    """
    df = df.dropna(subset=["url", "label"]).copy()
    return df

def remove_duplicate_urls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicated URLs to avoid repeated samples.
    """
    df = df.drop_duplicates(subset=["url"]).copy()
    return df

def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the cleaned dataset to CSV.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

def main():
    input_path = "data/raw/phishing_simple.csv"
    output_path = "data/processed/kaggle_cleaned.csv"

    print("Loading raw dataset...")
    df = load_kaggle_data(input_path)
    print("Original shape: ", df.shape)

    print("Selecting required columns...")
    df = select_required_columns(df)
    print("Shape after selecting columns:", df.shape)

    print("Renaming columns...")
    df = rename_columns(df)
    print("Columns:", df.columns.tolist())

    print("Converting labels to numeric...")
    df = convert_label_to_numeric(df)
    print("Label counts after conversion:")
    print(df["label"].value_counts(dropna=False))

    print("Removing missing values...")
    df = remove_missing_values(df)
    print("Shape after removing missing values:", df.shape)

    print("Removing duplicate URLs...")
    df = remove_duplicate_urls(df)
    print("Final shape after removing duplicates:", df.shape)

    print("Saving cleaned dataset...")
    save_cleaned_data(df, output_path)

    print(f"Done. Cleaned file saved to: {output_path}")
    print("\nFirst 5 rows:")
    print(df.head())

if __name__ == "__main__":
    main()