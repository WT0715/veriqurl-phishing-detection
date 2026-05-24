import pandas as pd

# 1. Read Kaaggle dataset
kaggle_path = "data/raw/phishing_simple.csv"
df_kaggle = pd.read_csv(kaggle_path)

# 2. Read PhiUSIIL dataset
phiusiil_path = "data/external/PhiUSIIL_Phishing_URL_Dataset.csv"
df_phiusiil = pd.read_csv(phiusiil_path)

# 3. Display basic information about the Kaggle datasets
print("===== Kaggle Dataset =====")
print("Shape:", df_kaggle.shape)          # how many rows and columns
print("Columns:", df_kaggle.columns.tolist())  # all column names
print(df_kaggle.head())                  # first 5 rows
print()

# 4. Display basic information about the PhiUSIIL dataset
print("===== PhiUSIIL Dataset =====")
print("Shape:", df_phiusiil.shape)
print("Columns:", df_phiusiil.columns.tolist())
print(df_phiusiil.head())
print()