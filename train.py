"""
train.py
----------
Chota synthetic dataset banata hai (house size -> price), Linear Regression
train karta hai, MLflow pe params/metrics/model log karta hai, aur model
ko disk pe save karta hai taake FastAPI usko serve kar sake.

Run:
    python train.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# 1) MLflow tracking setup
# ---------------------------------------------------------------------------
# Local run ke liye default: ./mlruns folder mein track hoga.
# DagsHub jaisa free hosted MLflow use karna ho to environment variables se
# override kar dein (README mein steps diye hain):
#   MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("house-price-linear-regression")


def generate_data(n_samples: int = 60, seed: int = 42) -> pd.DataFrame:
    """Chota synthetic dataset: house size (sq. ft.) -> price (in lakh PKR)."""
    rng = np.random.default_rng(seed)
    size_sqft = rng.uniform(500, 3500, n_samples)
    # true relationship + thora random noise
    price_lakh = 5 + 0.04 * size_sqft + rng.normal(0, 8, n_samples)
    return pd.DataFrame({"size_sqft": size_sqft, "price_lakh": price_lakh})


def train():
    df = generate_data()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/house_prices.csv", index=False)

    X = df[["size_sqft"]]
    y = df["price_lakh"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)

        # ---- log params & metrics ----
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("coefficient", model.coef_[0])
        mlflow.log_metric("intercept", model.intercept_)

        # ---- log model as MLflow artifact ----
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"RMSE: {rmse:.3f} | R2: {r2:.3f}")

        # ---- save plain model file for FastAPI to load directly ----
        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/model.pkl")
        print("Model saved to model/model.pkl")


if __name__ == "__main__":
    train()
