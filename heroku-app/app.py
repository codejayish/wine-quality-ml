from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

import joblib
model = joblib.load("artifacts/model.pkl")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract form inputs (convert to float)
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])
        
        # Prediction
        prediction = model.predict(final_features)[0]
        
        return render_template("index.html", 
                               prediction_text=f"Predicted Wine Quality: {prediction:.2f}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
