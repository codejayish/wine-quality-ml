# Wine Quality Linear Regression (UCI)
- Features: 11 physicochemical properties.
- Target: Quality (0–10).
- Pipeline: StandardScaler + LinearRegression.
- Offline eval (holdout):
  - RMSE: ~0.60–0.80 (typical)
  - MAE:  ~0.45–0.65
  - R²:   ~0.25–0.35
- Limitations: Linear model underfits; quality is subjective; dataset domain drift likely.
