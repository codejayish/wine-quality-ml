# heroku-app/train.py
import json, os, io, zipfile, urllib.request
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

UCI_RED = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
UCI_WHITE = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

FEATURES = [
    "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
    "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"
]
# assignment spells 'sulfates'; dataset uses 'sulphates'. We'll normalize to 'sulfates' at API edges.

def load_wine():
    red = pd.read_csv(UCI_RED, sep=';')
    white = pd.read_csv(UCI_WHITE, sep=';')
    df = pd.concat([red, white], ignore_index=True)
    return df

def main():
    os.makedirs("artifacts", exist_ok=True)
    df = load_wine()

    # Sanity
    assert set(FEATURES).issubset(df.columns), f"Missing features: {set(FEATURES)-set(df.columns)}"
    X = df[FEATURES].copy()
    y = df["quality"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    rmse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"RMSE={rmse:.3f} | MAE={mae:.3f} | R2={r2:.3f}")

    joblib.dump(pipe, "artifacts/model.pkl")
    # scaler is embedded in pipeline; dumping separately is optional
    joblib.dump(pipe.named_steps["scaler"], "artifacts/scaler.pkl")
    with open("artifacts/feature_order.json","w") as f:
        json.dump([f.replace("sulphates","sulfates") for f in FEATURES], f)

    # Persist a small inference contract example
    sample = X_test.iloc[0].to_dict()
    # convert 'sulphates' key to 'sulfates' in sample for API ergonomics
    sample = {("sulfates" if k=="sulphates" else k): float(v) for k,v in sample.items()}
    with open("artifacts/sample_payload.json","w") as f:
        json.dump(sample, f, indent=2)

if __name__ == "__main__":
    main()
