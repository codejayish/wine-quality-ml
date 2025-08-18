from flask import Flask, request, jsonify
import numpy as np
from model_utils import load_model, predict

app = Flask(__name__)

print("Loading model...")
model = load_model()
print("Model loaded!")

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(force=True)
    X = np.array(data["inputs"])
    preds = predict(model, X)
    return jsonify({"predictions": preds})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
