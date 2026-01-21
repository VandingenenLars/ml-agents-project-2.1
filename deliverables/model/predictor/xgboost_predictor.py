"""
xgboost_predictor.py

Description:
    Train XGBoost models to predict:
    1. Training success (binary classifier)
    2. Training duration (regressor for successful runs)
    3. Algorithm efficiency (binary classifier with weighted function for successful runs)

Usage:
    Run as main script: python xgboost_predictor.py

Author:
    Andreas Constantinou
Date:
    2025-11-25 - 2026-01-16
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

# Config
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

DATA_DIR = PROJECT_ROOT / "deliverables" / "data"
FEATURES_DIR = DATA_DIR / "features"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = DATA_DIR / "models/xgboost"
PREDICTIONS_DIR = DATA_DIR / "predictions/xgboost"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


def load_feature_sets():
    X_train = pd.read_csv(FEATURES_DIR / "X_train.csv")
    X_test = pd.read_csv(FEATURES_DIR / "X_test.csv")
    y_train = pd.read_csv(FEATURES_DIR / "y_train.csv")
    y_test = pd.read_csv(FEATURES_DIR / "y_test.csv")

    run_ids_train = pd.read_csv(FEATURES_DIR / "run_ids_train.csv")
    run_ids_test = pd.read_csv(FEATURES_DIR / "run_ids_test.csv")

    return X_train, X_test, y_train, y_test, run_ids_train, run_ids_test

def train_ram_regressor(X_train, y_train, X_test, y_test, target_col):
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=34
    )
    y_train_col = y_train[target_col].copy()
    y_test_col = y_test[target_col].copy()

    upper = y_train_col.quantile(0.995)
    y_train_col = y_train_col.clip(upper=upper)

    y_train_log = np.log1p(y_train_col)
    model.fit(X_train, y_train_log)

    preds_log = model.predict(X_test)
    preds_mb = np.expm1(preds_log)

    mae = mean_absolute_error(y_test_col, preds_mb)
    rmse = np.sqrt(mean_squared_error(y_test_col, preds_mb))
    print(f"{target_col} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    model.save_model(MODELS_DIR / f"{target_col}_regressor.json")
    pd.DataFrame(preds_mb, columns=[f"pred_{target_col}"]).to_csv(
        PREDICTIONS_DIR / f"{target_col}_predictions.csv", index=False
    )
    return preds_mb

def train_performance_regressor(X_train, y_train, X_test, y_test, target_col):
    # Only successful runs
    mask_train = y_train["reached_threshold"] == 1
    mask_test = y_test["reached_threshold"] == 1

    X_train_s = X_train.loc[mask_train]
    y_train_s = y_train.loc[mask_train, target_col]
    X_test_s = X_test.loc[mask_test]
    y_test_s = y_test.loc[mask_test, target_col]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=34
    )
    model.fit(X_train_s, y_train_s)
    preds = model.predict(X_test_s)

    mae = mean_absolute_error(y_test_s, preds)
    rmse = np.sqrt(mean_squared_error(y_test_s, preds))
    print(f"{target_col} (successful runs) - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    model.save_model(MODELS_DIR / f"{target_col}_regressor.json")
    pd.DataFrame(preds, columns=[f"pred_{target_col}"]).to_csv(
        PREDICTIONS_DIR / f"{target_col}_predictions.csv", index=False
    )
    return preds, X_test_s.index

def train_threshold_classifier(X_train, y_train, X_test, y_test):
    clf = XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=34,
    )
    clf.fit(X_train, y_train["reached_threshold"])
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test["reached_threshold"], preds)
    print(f"Threshold success accuracy: {acc:.3f}")

    clf.save_model(MODELS_DIR / "threshold_classifier.json")
    pd.DataFrame(preds, columns=["pred_reached_threshold"]).to_csv(
        PREDICTIONS_DIR / "threshold_classifier_predictions.csv", index=False
    )
    return preds, probs


def main():
    X_train, X_test, y_train, y_test, run_ids_train, run_ids_test = load_feature_sets()
    summary = pd.DataFrame({"run_id": run_ids_test["run_id"]})

    # RAM regressors
    for col in ["avg_ram_usage_mb", "peak_ram_usage_mb"]:
        preds = train_ram_regressor(X_train, y_train, X_test, y_test, col)
        summary[f"pred_{col}"] = preds

    # Threshold classifier
    threshold_preds, threshold_probs = train_threshold_classifier(X_train, y_train, X_test, y_test)
    summary["pred_reached_threshold"] = threshold_preds
    summary["pred_reached_threshold_prob"] = threshold_probs

    # Iteration & seconds regressors (only successful runs)
    for col in ["iterations_to_threshold", "seconds_to_threshold"]:
        preds, idx = train_performance_regressor(X_train, y_train, X_test, y_test, col)
        summary.loc[idx, f"pred_{col}"] = preds

    summary.to_csv(PREDICTIONS_DIR / "summary.csv", index=False)
    print(f"\nAll models saved to {MODELS_DIR} and predictions + summary saved to {PREDICTIONS_DIR}.")

if __name__ == "__main__":
    main()
