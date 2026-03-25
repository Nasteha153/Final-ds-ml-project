import json
import joblib
import pandas as pd

CURRENT_YEAR = 2026

# Load saved artifacts
TRAIN_COLUMNS = json.load(open("models/train_columns.json"))
SCALER = joblib.load("models/car_scaler.pkl")


def prepare_features_from_raw(record: dict) -> pd.DataFrame:
    """
    Convert raw car input into model-ready features
    (same preprocessing as training).
    """

    # 1) Get raw inputs
    year = int(record.get("year", CURRENT_YEAR))
    km = float(record.get("km_driven", 0))
    owner = float(record.get("owner", 0))
    fuel = str(record.get("fuel", "Petrol"))
    seller = str(record.get("seller_type", "Individual"))
    trans = str(record.get("transmission", "Manual"))

    # 2) Feature engineering
    car_age = CURRENT_YEAR - year

    # 3) Create empty row
    row = {col: 0.0 for col in TRAIN_COLUMNS}

    # 4) Fill numeric features
    for name, val in [
        ("year", year),
        ("km_driven", km),
        ("owner", owner),
        ("car_age", car_age),
    ]:
        if name in row:
            row[name] = float(val)

    # 5) One-hot encoding (IMPORTANT)
    # fuel
    fuel_col = f"fuel_{fuel}"
    if fuel_col in row:
        row[fuel_col] = 1.0

    # seller_type
    seller_col = f"seller_type_{seller}"
    if seller_col in row:
        row[seller_col] = 1.0

    # transmission
    trans_col = f"transmission_{trans}"
    if trans_col in row:
        row[trans_col] = 1.0

    # 6) Convert to DataFrame (1 row)
    df_one = pd.DataFrame([row], columns=TRAIN_COLUMNS)

    # 7) Apply scaling (same as training)
    if hasattr(SCALER, "feature_names_in_"):
        cols_to_scale = list(SCALER.feature_names_in_)
        df_one[cols_to_scale] = SCALER.transform(df_one[cols_to_scale])

    return df_one