"""
feature_extractor.py

Description:
    The feature extractor module is responsible for taking in processed data and returning a feature matrix and a
    feature summary that can be used for training machine learning predictive models.

Usage:
    Input: Preprocessed .csv file
    Output: Feature data sets (X_train, X_test, y_train, y_test)

Author:
    Andreas
Date:
    2025-11-10
"""

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Config
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


DATA_DIR = PROJECT_ROOT / "deliverables" / "data"
CSV_PATH = "processed_data.csv"
PROCESSED_PATH = DATA_DIR / "processed" / "processed_data.csv"
FEATURES_DIR = DATA_DIR / "features"

TEST_SIZE = 0.2
RANDOM_STATE = 34


def load_data(filepath: str) -> pd.DataFrame:
    """Load processed CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed data not found at: {filepath}")
    return pd.read_csv(filepath)


def extract_features_and_targets(df: pd.DataFrame):
    """
    Split dataframe into features (X) and targets (y).
    """
    # !!! Update this list if schema changes !!!
    target_columns = [
        "iterations_to_target",
        "peak_ram_usage_mb",
        "avg_ram_usage_mb",
    ]

    feature_columns = [c for c in df.columns if c not in target_columns]
    X = df[feature_columns]
    y = df[target_columns]

    return X, y


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns and scale numeric features."""

    # Encode categorical variables (one-hot)
    X = pd.get_dummies(X, columns=["game_type", "algorithm"], drop_first=True)

    # Scale numeric columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    return X


def save_datasets(X_train, X_test, y_train, y_test):
    """Save output to destination folder."""
    os.makedirs(FEATURES_DIR, exist_ok=True)

    X_train.to_csv(os.path.join(FEATURES_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(FEATURES_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(FEATURES_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(FEATURES_DIR, "y_test.csv"), index=False)

    print(f" Features saved in {FEATURES_DIR}")


def main():
    print(" Running Feature Extractor...")

    df = load_data(PROCESSED_PATH)
    X, y = extract_features_and_targets(df)
    X = preprocess_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    save_datasets(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()

