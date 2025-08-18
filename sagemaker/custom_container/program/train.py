import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

from model_utils import save_model

# ✅ Dataset: you can replace with your dataset path if needed
DATA_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Linear Regression...")
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")

# Ensure model directory exists
os.makedirs("/opt/ml/model", exist_ok=True)

print("Saving model...")
save_model(model)

print("Training complete. Model saved to /opt/ml/model/model.pkl")
