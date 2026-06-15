"""
model.py  (v2 — Delhi Edition)
-------------------------------
Trains a Decision Tree to predict bin overflow.

WHAT CHANGED FROM v1:
  - Feature columns updated: 'latitude' and 'longitude' replace old 'x' / 'y'
  - Added 'area_type_encoded' — encodes commercial / market / residential as
    numbers so the Decision Tree can use it
  - Everything else (train_test_split, accuracy_score, predict_proba) is
    identical to v1 — keeping it simple!
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Map text area types to numbers the model can read
AREA_TYPE_MAP = {
    "residential": 0,
    "commercial":  1,
    "market":      2,
}

# These are the exact column names the model will train on
FEATURE_COLUMNS = [
    "fill_level",
    "latitude",
    "longitude",
    "past_overflow_count",
    "area_type_encoded",   # NEW — encodes commercial / market / residential
]


def _encode_area_type(df):
    """
    Adds an 'area_type_encoded' column to the DataFrame.
    Converts text like 'market' → 2 so the Decision Tree can use it.
    """
    df = df.copy()
    df["area_type_encoded"] = df["area_type"].map(AREA_TYPE_MAP).fillna(0).astype(int)
    return df


def train_model(df):
    """
    Trains a Decision Tree classifier on the Delhi bin dataset.

    Parameters:
        df : DataFrame from data_generator.generate_bin_data()

    Returns:
        model    : Trained DecisionTreeClassifier
        accuracy : Float accuracy on the held-out test set
        X_test   : Test feature matrix
        y_test   : True labels for test set
    """

    # Encode area type before splitting
    df_enc = _encode_area_type(df)

    X = df_enc[FEATURE_COLUMNS]   # Features
    y = df_enc["overflow"]         # Target label

    # 80% train / 20% test split — same as v1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Decision Tree — max_depth=4 keeps it simple and explainable
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy, X_test, y_test


def predict_overflow(model, df):
    """
    Predicts overflow probability for every bin.

    Parameters:
        model : Trained model from train_model()
        df    : Full bin DataFrame

    Returns:
        results_df : df + 'overflow_prob' and 'predicted' columns
    """

    df_enc = _encode_area_type(df)
    X_all  = df_enc[FEATURE_COLUMNS]

    # Column index 1 = probability of class 1 (overflow)
    proba     = model.predict_proba(X_all)[:, 1]
    predicted = (proba >= 0.5).astype(int)

    results_df = df.copy()
    results_df["overflow_prob"] = proba.round(3)
    results_df["predicted"]     = predicted

    return results_df


# Quick test
if __name__ == "__main__":
    from data_generator import generate_bin_data

    df = generate_bin_data()
    model, acc, X_test, y_test = train_model(df)
    print(f"Accuracy: {acc * 100:.2f}%")

    results = predict_overflow(model, df)
    print(results[["bin_id", "location_name", "fill_level",
                   "overflow_prob", "predicted"]].head(12).to_string())
