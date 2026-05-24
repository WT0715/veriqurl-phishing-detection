import re
import pandas as pd
from urllib.parse import urlparse
from pathlib import Path

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "bank", 
    "signin", "confirm", "password", "paypal", "free", "bonus",
    "win", "urgent", "click", "limited", "security", "webscr"
]

def count_digits(text: str) -> int:
    return sum(char.isdigit() for char in text)

def count_special_characters(text: str) -> int:
    return sum(not char.isalnum() for char in text)

def count_suspicious_words(text: str) -> int:
    text_lower = text.lower()
    return sum(word in text_lower for word in SUSPICIOUS_WORDS)

def has_ip_adress(url: str) -> int:
    """
    Check whether the uRL contains as IPv4 address.
    Example: http://192.168.1.1/login
    """
    pattern = r"(?:\d{1,3}\.){3}\d{1,3}"
    return int(bool(re.search(pattern, url)))

def extract_url_features(url: str) -> dict:
    """
    Extract handcrafted lexical features from a single URL.
    """
    parsed = urlparse(url)

    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    features = {
        "url_length": len(url),
        "domain_length": len(domain),
        "path_length": len(path),
        "query_length": len(query),
        "digit_count": count_digits(url),
        "special_char_count": count_special_characters(url),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "slash_count": url.count("/"),
        "question_mark_count": url.count("?"),
        "equal_count": url.count("="),
        "at_count": url.count("@"),
        "ampersand_count": url.count("&"),
        "underscore_count": url.count("_"),
        "tilde_count": url.count("~"),
        "percent_count": url.count("%"),
        "subdomain_count": max(0, domain.count(".") -1),
        "has_https": int(parsed.scheme == "https"),
        "has_ip_address": has_ip_adress(url),
        "suspicious_word_count": count_suspicious_words(url),
    }

    return features

def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a dataframe with column [url, label]
    into a dataframe of external features + label.
    """
    feature_rows = []

    for _, row in df.iterrows():
        url = str(row["url"])
        label = int(row['label'])

        features = extract_url_features(url)
        features["label"] = label
        feature_rows.append(features)

    feature_df = pd.DataFrame(feature_rows)
    return feature_df

def save_featured_data(df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

def main():
    input_files = {
        "train": "data/processed/splits/train.csv",
        "val": "data/processed/splits/val.csv",
        "test": "data/processed/splits/test.csv"
    }

    output_files = {
        "train": "data/processed/features/train_features.csv",
        "val": "data/processed/features/val_features.csv",
        "test": "data/processed/features/test_features.csv"
    }

    for split_name in ["train", "val", "test"]:
        print(f"Processing {split_name} set...")
        df = pd.read_csv(input_files[split_name])
        print("Input shape: ", df.shape)

        feature_df = transform_dataframe(df)
        print("Output shape:", feature_df.shape)
        print("First 5 rows:")
        print(feature_df.head())

        save_featured_data(feature_df, output_files[split_name])
        print(f"Saved to: {output_files[split_name]}")
        print("-" * 50)

if __name__ == "__main__":
    main()