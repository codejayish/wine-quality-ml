# sagemaker/src/inference.py
import json, os
import numpy as np, joblib

MODEL = None
FEATURES = None

def model_fn(model_dir):
    global MODEL, FEATURES
    MODEL = joblib.load(os.path.join(model_dir, "model.pkl"))
    with open(os.path.join(model_dir, "feature_order.json")) as f:
        FEATURES = json.load(f)
    return MODEL

def input_fn(request_body, content_type):
    data = json.loads(request_body)
    payload = data.get("features", data)

    # Accept both "sulphates" and "sulfates"
    if "sulphates" in payload and "sulfates" not in payload:
        payload["sulfates"] = payload["sulphates"]

    x = [float(payload[name]) for name in FEATURES]
    return np.array(x, dtype=float).reshape(1, -1)

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, accept):
    return json.dumps({"prediction": float(prediction[0])})
