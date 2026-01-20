import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

# Config
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

DATA_DIR = PROJECT_ROOT / "deliverables" / "data"
FEATURES_DIR = DATA_DIR / "features"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = DATA_DIR / "random_forest" / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions/random_forest"

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

def train_ram_regressor(name,X_train, X_test, y_train, y_test, target):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=2,
        min_samples_split=5,
        max_features="sqrt",
        random_state=67
    )
    y_train_col = y_train[target].copy()
    y_test_col = y_test[target].copy()

    upper = y_train_col.quantile(0.995)
    y_train_col = y_train_col.clip(upper=upper)

    y_train_log = np.log1p(y_train_col)
    model.fit(X_train, y_train_log)

    preds_log = model.predict(X_test)
    preds_mb = np.expm1(preds_log)

    mae = mean_absolute_error(y_test_col, preds_mb)
    rmse = np.sqrt(mean_squared_error(y_test_col, preds_mb))
    print(f"{target} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    model_path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model,model_path)

    pd.DataFrame(preds_mb, columns=[f"pred_{target}"]).to_csv(
        PREDICTIONS_DIR / f"{target}_predictions.csv", index=False
    )
    return preds_mb

def train_performance_regressor(name,X_train, y_train, X_test, y_test, target):
    mask_train = y_train["reached_threshold"] == 1
    mask_test = y_test["reached_threshold"] == 1

    X_train_s = X_train.loc[mask_train]
    y_train_s = y_train.loc[mask_train, target]
    X_test_s = X_test.loc[mask_test]
    y_test_s = y_test.loc[mask_test, target]

    if len(y_train_s) == 0:
        print(f"{name}: No data to train with, skipping.")
        return [np.nan] * len(X_test)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=2,
        min_samples_split=5,
        max_features="sqrt",
        random_state=67
    )
    model.fit(X_train_s, y_train_s)
    preds = model.predict(X_test_s)

    mae = mean_absolute_error(y_test_s, preds)
    rmse = np.sqrt(mean_squared_error(y_test_s, preds))

    print(f"\n{name} MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    model_path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model,model_path)

    pred_path = PREDICTIONS_DIR / f"{name}_predictions.csv"
    pd.DataFrame(preds, columns=[f"pred_{name}"]).to_csv(pred_path, index=False)

    return preds, X_test_s.index

def train_threshold_classifier(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=2,
        min_samples_split=5,
        max_features='sqrt',
        random_state=67
    )
    clf.fit(X_train, y_train["reached_threshold"])
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test["reached_threshold"], preds)
    print(f"\nThreshold success accuracy: {acc:.4f}")

    model_path = MODELS_DIR / f"threshold_success_classifier.pkl"
    joblib.dump(clf,model_path)

    pred_path = PREDICTIONS_DIR / f"threshold_classifier_predictions.csv"
    pd.DataFrame(preds, columns=[f"preds_threshold_success"]).to_csv(pred_path, index=False)

    return preds, probs

def main():
    X_train, X_test, y_train, y_test, run_ids_train, run_ids_test = load_feature_sets()
    summary_df = pd.DataFrame({"run_id": run_ids_test["run_id"]})
    print("y_train columns:", y_train.columns.tolist())

    for target in ["avg_ram_usage_mb", "peak_ram_usage_mb"]:
        preds = train_ram_regressor(
            name= f"regressor_{target}",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            target=target
        )
        summary_df[f"pred_{target}"] = preds

    threshold_preds, threshold_probs = train_threshold_classifier(X_train, y_train, X_test, y_test)
    summary_df["pred_reached_threshold"] = threshold_preds
    summary_df["pred_reached_threshold_prob"] = threshold_probs

    for target in ["iterations_to_threshold", "seconds_to_threshold"]:
        preds, idx = train_performance_regressor(target, X_train, y_train, X_test, y_test, target)
        summary_df.loc[idx, f"pred_{target}"] = preds

    # Save summary.csv
    summary_path = PREDICTIONS_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nAll models saved to {MODELS_DIR} and predictions + summary saved to {PREDICTIONS_DIR}.")

if __name__ == "__main__":
    main()
