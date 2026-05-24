import os
import json
import pandas as pd
import numpy as np

MAX_LEN = 200
PAD_VALUE = 0

TRAIN_PATH = "data/processed/splits/train.csv"
VAL_PATH = "data/processed/splits/val.csv"
TEST_PATH = "data/processed/splits/test.csv"

OUTPUT_DIR = "data/processed/tcn"
VOCAB_PATH = os.path.join(OUTPUT_DIR, "char_vocab.json")


def build_char_vocab(urls):
    chars = set()
    for url in urls:
        for ch in str(url):
            chars.add(ch)

    chars = sorted(list(chars))

    char_to_idx = {ch: idx + 1 for idx, ch in enumerate(chars)}
    return char_to_idx


def encode_url(url, char_to_idx, max_len=200):
    encoded = [char_to_idx.get(ch, 0) for ch in str(url)]

    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        encoded += [PAD_VALUE] * (max_len - len(encoded))

    return encoded


def process_split(input_path, output_path, char_to_idx, max_len=200):
    df = pd.read_csv(input_path)

    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{input_path} must contain 'url' and 'label' columns.")

    X = np.array([encode_url(url, char_to_idx, max_len) for url in df["url"]])
    y = df["label"].values

    np.save(os.path.join(output_path, os.path.basename(input_path).replace(".csv", "_X.npy")), X)
    np.save(os.path.join(output_path, os.path.basename(input_path).replace(".csv", "_y.npy")), y)

    print(f"Saved {input_path} -> X shape: {X.shape}, y shape: {y.shape}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if "url" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError("train.csv must contain 'url' and 'label' columns.")

    all_train_urls = train_df["url"].astype(str).tolist()
    char_to_idx = build_char_vocab(all_train_urls)

    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(char_to_idx, f, ensure_ascii=False, indent=2)

    print(f"Character vocabulary saved to {VOCAB_PATH}")
    print(f"Vocabulary size: {len(char_to_idx)}")

    process_split(TRAIN_PATH, OUTPUT_DIR, char_to_idx, MAX_LEN)
    process_split(VAL_PATH, OUTPUT_DIR, char_to_idx, MAX_LEN)
    process_split(TEST_PATH, OUTPUT_DIR, char_to_idx, MAX_LEN)

    print("TCN data preparation completed.")


if __name__ == "__main__":
    main()