from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score
)

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 6)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent / "data"
FEATURES_DIR = BASE_DIR / "features"
PREDICTIONS_DIR = BASE_DIR / "predictions/xgboost"
MODELS_DIR = BASE_DIR / "models/xgboost"
PLOTS_DIR = SCRIPT_DIR / "plots_xgboost"
PLOTS_DIR.mkdir(exist_ok=True)


def plot_ram_scatter(y_true, y_pred, col_name):
    plt.figure()
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)

    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label="Perfect")

    m, b = np.polyfit(y_true, y_pred, 1)
    plt.plot(y_true, m * y_true + b, 'g-', label=f"Fit: y={m:.2f}x+{b:.1f}")

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{col_name} - Predicted vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{col_name}_scatter.png")
    plt.close()


def plot_ram_residuals(y_true, y_pred, col_name):
    residuals = y_pred - y_true

    plt.figure()
    sns.scatterplot(x=y_true, y=residuals, alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Actual")
    plt.ylabel("Residuals")
    plt.title(f"{col_name} - Residuals")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{col_name}_residuals.png")
    plt.close()

    plt.figure()
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title(f"{col_name} - Residual Distribution")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{col_name}_residual_hist.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, col_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{col_name} - Confusion Matrix")
    plt.savefig(PLOTS_DIR / f"{col_name}_confusion_matrix.png")
    plt.close()


def plot_roc_curve(y_true, y_probs, col_name):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{col_name} - ROC Curve")
    plt.legend()
    plt.savefig(PLOTS_DIR / f"{col_name}_roc.png")
    plt.close()


def plot_probability_histogram(y_probs, col_name):
    plt.figure()
    sns.histplot(y_probs, bins=20, kde=True)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title(f"{col_name} - Predicted Probabilities")
    plt.savefig(PLOTS_DIR / f"{col_name}_prob_hist.png")
    plt.close()


def plot_feature_importance(model_path, col_name, model_type=None):
    if model_type is None:
        model_type = "classifier" if "classifier" in model_path.name else "regressor"
    model = xgb.XGBRegressor() if model_type == "regressor" else xgb.XGBClassifier()
    model.load_model(model_path)

    importance = model.get_booster().get_score(importance_type='weight')
    importance = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values(by='importance', ascending=False)

    plt.figure()
    sns.barplot(x='importance', y='feature', data=importance)
    plt.title(f"{col_name} - Feature Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{col_name}_feature_importance.png")
    plt.close()

y_test = pd.read_csv(FEATURES_DIR / "y_test.csv")
summary = pd.read_csv(PREDICTIONS_DIR / "summary.csv")

for col in ["avg_ram_usage_mb", "peak_ram_usage_mb"]:
    y_true = y_test[col]
    y_pred = summary[f"pred_{col}"]
    plot_ram_scatter(y_true, y_pred, col)
    plot_ram_residuals(y_true, y_pred, col)
    print(f"{col} MAE:", mean_absolute_error(y_true, y_pred))
    print(f"{col} RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))

    model_file = MODELS_DIR / f"{col}_regressor.json"
    plot_feature_importance(model_file, col, model_type="regressor")


y_true = y_test["reached_threshold"]
y_pred = summary["pred_reached_threshold"]
y_probs = summary["pred_reached_threshold_prob"]

plot_confusion_matrix(y_true, y_pred, "Training_Success")
plot_roc_curve(y_true, y_probs, "Training_Success")
plot_probability_histogram(y_probs, "Training_Success")
print("Training Success Accuracy:", accuracy_score(y_true, y_pred))

plot_feature_importance(MODELS_DIR / "threshold_classifier.json", "Training_Success")

print(f"\nAll plots_rf saved to '{PLOTS_DIR.resolve()}' folder.")
