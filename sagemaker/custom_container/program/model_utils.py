import joblib
import numpy as np

MODEL_PATH = "/opt/ml/model/model.pkl"

def save_model(model):
    joblib.dump(model, MODEL_PATH)

def load_model():
    return joblib.load(MODEL_PATH)

def predict(model, X):
    return model.predict(X).tolist()
