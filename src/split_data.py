import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_cleaned_data(file_path: str) -> pd.DataFrame:
    """
    Load the cleaned dataset from CSV.
    """
    df = pd.read_csv(file_path)
    return df

def split_dataset(df: pd.DataFrame, random_state: int = 42):
    """
    Split dataset into:
    - train 70%
    - validation 15%
    - test 15%

    Stratified split is used so label distribution stays similar.
    """

    # First split: train 70%, temp 30%
    train_df, temp_df = train_test_split(
        df,
        test_size = 0.30,
        stratify = df["label"],
        random_state = random_state
    )
    
    # Second split: temp 30% -> val 15% + test 15%
    val_df, test_df = train_test_split(
        temp_df,
        test_size = 0.50,
        stratify = temp_df["label"],
        random_state = random_state
    )

    return train_df, val_df, test_df

def save_split_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str):
    """
    Save train, validation, and test sets into csv files.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

def print_split_info(name: str, df: pd.DataFrame):
    """
    Print shape and label distribution for one split.
    """
    print(f"===== {name} =====")
    print("Shape:", df.shape)
    print("Label counts:")
    print(df["label"].value_counts())
    print("Label proportions:")
    print(df["label"].value_counts(normalize=True))
    print()

def main():
    input_path = "data/processed/kaggle_cleaned.csv"
    output_dir = "data/processed/splits"

    print("Loading cleaned dataset...")
    df = load_cleaned_data(input_path)
    print("Full dataset shape:", df.shape)
    print()

    print("Splitting dataset into train / val / test...")
    train_df, val_df, test_df = split_dataset(df)

    print_split_info("TRAIN", train_df)
    print_split_info("VALIDATION", val_df)
    print_split_info("TEST", test_df)

    print("Saving split files...")
    save_split_data(train_df, val_df, test_df, output_dir)

    print(f"Done. Files saved in: {output_dir}")


if __name__ == "__main__":
    main()