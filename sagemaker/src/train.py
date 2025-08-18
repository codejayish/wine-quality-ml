# sagemaker/src/train.py
import argparse, os, json
import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-col", type=str, default="quality")
    return parser.parse_args()

def main():
    args = parse_args()

    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    train_path = os.path.join(train_dir, "train.csv")
    df = pd.read_csv(train_path)

    # Normalize spelling
    df = df.rename(columns={"sulphates": "sulfates"})
    features = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
        "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulfates", "alcohol"
    ]

    X = df[features].copy()
    y = df[args.target_col].astype(float)

    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    pipe.fit(X, y)

    # Save model and feature order
    joblib.dump(pipe, os.path.join(model_dir, "model.pkl"))
    with open(os.path.join(model_dir, "feature_order.json"), "w") as f:
        json.dump(features, f)

if __name__ == "__main__":
    main()
