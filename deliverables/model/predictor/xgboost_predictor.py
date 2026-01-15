"""
xgboost_predictor.py

Description:
    Describe what this module does.

Usage:
    Example of how to use this module.

Author:
    Andreas Constantinou
Date:
    2025-11-25
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

# Config
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

DATA_DIR = PROJECT_ROOT / "deliverables" / "data"
FEATURES_DIR = DATA_DIR / "features"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = DATA_DIR / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


def load_feature_sets():
    X_train = pd.read_csv(FEATURES_DIR / "X_train.csv")
    X_test = pd.read_csv(FEATURES_DIR / "X_test.csv")
    y_train = pd.read_csv(FEATURES_DIR / "y_train.csv")
    y_test = pd.read_csv(FEATURES_DIR / "y_test.csv")
    return X_train, X_test, y_train, y_test


def encode_categorical(df):
    for col in df.select_dtypes(include=["object"]).columns:
        if col != "run_id":
            df[col] = df[col].astype("category")
    return df


def train_regressor(name, X_train, y_train, X_test, y_test):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        enable_categorical=True
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"\n{name} MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    model_path = MODELS_DIR / f"{name}.json"
    model.save_model(model_path)

    pred_path = PREDICTIONS_DIR / f"{name}_predictions.csv"
    pd.DataFrame(preds, columns=[f"pred_{name}"]).to_csv(pred_path, index=False)

    return preds


def compute_efficiency(df, weights=(1/3, 1/3, 1/3)):
    T_norm = (df['training_time_to_target_sec'] - df['training_time_to_target_sec'].min()) / \
             (df['training_time_to_target_sec'].max() - df['training_time_to_target_sec'].min())
    I_norm = (df['iterations_to_target'] - df['iterations_to_target'].min()) / \
             (df['iterations_to_target'].max() - df['iterations_to_target'].min())
    R_norm = (df['peak_ram_usage_mb'] - df['peak_ram_usage_mb'].min()) / \
             (df['peak_ram_usage_mb'].max() - df['peak_ram_usage_mb'].min())
    alpha, beta, gamma = weights
    return alpha*T_norm + beta*I_norm + gamma*R_norm


def train_efficiency_classifier(X_train, X_test, y_train, y_test, weights=(1/3, 1/3, 1/3)):
    y_train_eff = compute_efficiency(y_train, weights)
    y_test_eff = compute_efficiency(y_test, weights)

    threshold = y_train_eff.median()
    y_train_cls = (y_train_eff < threshold).astype(int)
    y_test_cls = (y_test_eff < threshold).astype(int)

    clf = XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True
    )
    clf.fit(X_train, y_train_cls)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test_cls, preds)

    print(f"\nEfficiency Classifier Accuracy: {acc:.4f}")

    model_path = MODELS_DIR / "efficiency_classifier.json"
    clf.save_model(model_path)

    pred_path = PREDICTIONS_DIR / "efficiency_classifier_predictions.csv"
    pd.DataFrame(preds, columns=["pred_efficiency_class"]).to_csv(pred_path, index=False)

    return preds


def main():
    X_train, X_test, y_train, y_test = load_feature_sets()

    # Keep a copy of X_test with run_id for summary.csv
    summary_df = X_test.copy()

    # Encode categorical columns in XGBoost-friendly format
    X_train = encode_categorical(X_train).drop(columns=["run_id"], errors="ignore")
    X_test_encoded = encode_categorical(X_test).drop(columns=["run_id"], errors="ignore")

    targets = [
        "training_time_to_target_sec",
        "iterations_to_target",
        "peak_ram_usage_mb",
        "avg_ram_usage_mb",
    ]

    for target in targets:
        preds = train_regressor(
            name=f"regressor_{target}",
            X_train=X_train,
            y_train=y_train[target],
            X_test=X_test_encoded,
            y_test=y_test[target],
        )
        summary_df[f"pred_{target}"] = preds

    eff_preds = train_efficiency_classifier(X_train, X_test_encoded, y_train, y_test)
    summary_df["pred_efficiency_class"] = eff_preds

    # Save summary.csv
    summary_path = PREDICTIONS_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nAll models saved to {MODELS_DIR} and predictions + summary saved to {PREDICTIONS_DIR}.")


if __name__ == "__main__":
    main()
