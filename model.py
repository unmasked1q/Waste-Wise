import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

AREA_TYPE_MAP = {
    "residential": 0,
    "commercial":  1,
    "market":      2,
}

FEATURE_COLUMNS = ["fill_level","latitude","longitude","past_overflow_count","area_type_encoded",   ]

def _encode_area_type(df):
    df = df.copy()
    df["area_type_encoded"] = df["area_type"].map(AREA_TYPE_MAP).fillna(0).astype(int)
    return df


def train_model(df):
    df_enc = _encode_area_type(df)

    X =df_enc[FEATURE_COLUMNS]   
    y = df_enc["overflow"]        

    # 80-20 train test split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Decision Tree 
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy, X_test, y_test


def predict_overflow(model, df):
    df_enc = _encode_area_type(df)
    X_all  = df_enc[FEATURE_COLUMNS]

    proba     = model.predict_proba(X_all)[:, 1]
    predicted = (proba >= 0.5).astype(int)

    results_df = df.copy()
    results_df["overflow_prob"] = proba.round(3)
    results_df["predicted"]     = predicted

    return results_df

if __name__ == "__main__":
    from data_generator import generate_bin_data

    df = generate_bin_data()
    model, acc, X_test, y_test = train_model(df)
    print(f"Accuracy: {acc * 100:.2f}%")

    results = predict_overflow(model, df)
    print(results[["bin_id", "location_name", "fill_level",
                   "overflow_prob", "predicted"]].head(12).to_string())
