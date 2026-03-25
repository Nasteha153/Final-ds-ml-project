import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

# ==============================
# 1) LOAD DATA
# ==============================
CSV_PATH = "dataset/Car-price-prediction.csv"
df = pd.read_csv(CSV_PATH)

df = df.drop(columns=["name"])

print("\n=== INITIAL HEAD ===")
print(df.head())

print("\n=== INITIAL INFO ===")
print(df.info())

print("\n=== INITIAL MISSING VALUES ===")
print(df.isnull().sum())

# ==============================
# 2) CLEAN TARGET
# ==============================
df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")

# ==============================
# 3) HANDLE MISSING VALUES
# ==============================
df["km_driven"] = df["km_driven"].fillna(df["km_driven"].median())
df["fuel"] = df["fuel"].fillna(df["fuel"].mode()[0])
df["transmission"] = df["transmission"].fillna(df["transmission"].mode()[0])
df["owner"] = df["owner"].fillna(df["owner"].mode()[0])

# 🔥 IMPORTANT: owner ka dhig number
df["owner"] = df["owner"].astype("category").cat.codes

# ==============================
# 4) REMOVE DUPLICATES
# ==============================
before = df.shape
df = df.drop_duplicates()
after = df.shape
print(f"\nDropped duplicates: {before} → {after}")

# ==============================
# 5) OUTLIER HANDLING
# ==============================
def iqr_fun(series, k=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

low_price, high_price = iqr_fun(df["selling_price"])
low_km, high_km = iqr_fun(df["km_driven"])

df["selling_price"] = df["selling_price"].clip(lower=low_price, upper=high_price)
df["km_driven"] = df["km_driven"].clip(lower=low_km, upper=high_km)

# ==============================
# 6) FEATURE ENGINEERING
# ==============================
CURRENT_YEAR = 2025
df["car_age"] = CURRENT_YEAR - df["year"]

# ==============================
# 7) ENCODING (NO owner here!)
# ==============================
df = pd.get_dummies(df, columns=["fuel", "seller_type", "transmission"], drop_first=True)

# ==============================
# 8) SCALING
# ==============================
dont_scale = {"selling_price"}
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

features_to_scale = [c for c in numeric_cols if c not in dont_scale]

scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# ==============================
# 9) SAVE FILES
# ==============================
os.makedirs("models", exist_ok=True)

joblib.dump(scaler, "models/car_scaler.pkl")

TRAIN_COLUMNS = df.drop(columns=["selling_price"]).columns.tolist()
json.dump(TRAIN_COLUMNS, open("models/train_columns.json", "w"))

# ==============================
# FINAL CHECK
# ==============================
print("\n=== FINAL HEAD ===")
print(df.head())

print("\n=== FINAL INFO ===")
print(df.info())

print("\n=== FINAL MISSING VALUES ===")
print(df.isnull().sum())

# ==============================
# SAVE CLEAN DATA
# ==============================
df.to_csv("dataset/clean_car_data.csv", index=False)

print("\n✅ Data preprocessing completed!")