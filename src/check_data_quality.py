import pandas as pd

# 1. Read Datasets
kaggle_path = "data/raw/phishing_simple.csv"
phiusiil_path = "data/external/PhiUSIIL_Phishing_URL_Dataset.csv"

df_kaggle = pd.read_csv(kaggle_path)
df_phiusiil = pd.read_csv(phiusiil_path)

# 2. Show basic label information
print("===== Kaggle Label Info =====")
print("Unique labels:", df_kaggle["label"].unique()) # see the values in the "label" column
print("Label counts:")
print(df_kaggle["label"].value_counts())
print()

print("===== PhiUSIIL Label Info =====")
print("Unique labels:", df_phiusiil["label"].unique())
print("Label counts:")
print(df_phiusiil["label"].value_counts())
print()

# 3. Check for missing values
print("===== Missing Values =====")
print("Kaggle missing URL:", df_kaggle["URL"].isna().sum())
print("Kaggle missing label:", df_kaggle["label"].isna().sum())
print("PhiUSIIL missing URL:", df_phiusiil["URL"].isna().sum())
print("PhiUSIIL missing label:", df_phiusiil["label"].isna().sum())
print()

# 4. Check duplicate URLs inside each dataset
print("===== Duplicate URLs Inside Each Dataset =====")
print("Kaggle duplicated URLs:", df_kaggle["URL"].duplicated().sum())
print("PhiUSIIL duplicated URLs:", df_phiusiil["URL"].duplicated().sum())
print()

# 5. Check overlap between Kaggle and PhiUSIIL datasets
kaggle_urls = set(df_kaggle["URL"].dropna().astype(str))
phiusiil_urls = set(df_phiusiil["URL"].dropna().astype(str))

overlap = kaggle_urls.intersection(phiusiil_urls)

print("===== Overlap Between Datasets =====")
print("Number of overlapping URLs:", len(overlap))
print("Kaggle unique URLs:", len(kaggle_urls))
print("PhiUSIIL unique URLs:", len(phiusiil_urls))

if len(overlap) > 0:
    print("Sample overlapping URLs:")
    for i, url in enumerate(list(overlap)[:10]):
        print(f"{i+1}. {url}")