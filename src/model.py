"""
model.py - train and evaluate credit risk models

Uses LightGBM as the primary model (it's what most banks actually use
for tabular credit data) and XGBoost as a challenger for comparison.

Evaluation uses Gini and KS statistic, which are the metrics credit
risk teams actually report, not just accuracy.
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[INFO] xgboost not installed - will use LightGBM only")
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
import joblib
import warnings

warnings.filterwarnings("ignore")


def calculate_gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Gini = 2 * AUC - 1. Ranges from 0 (random) to 1 (perfect).
    
    Banks report Gini instead of AUC. A Gini of 0.35-0.45 is typical
    for a decent credit scorecard.
    """
    auc = roc_auc_score(y_true, y_prob)
    return 2 * auc - 1


def calculate_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """KS = max separation between default and non-default CDFs.
    
    Tells you at what probability threshold the model best distinguishes
    good borrowers from bad ones. KS above 40% is considered strong.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return max(tpr - fpr)


def train_lightgbm(
    X_train: pd.DataFrame, y_train: pd.Series, config: dict
) -> lgb.LGBMClassifier:
    """Train LightGBM. This is the primary model.
    
    scale_pos_weight=3 handles class imbalance by making the model
    pay 3x more attention to defaults during training.
    """
    params = config["model"]["lightgbm"]
    
    model = lgb.LGBMClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        scale_pos_weight=params["scale_pos_weight"],
        random_state=params["random_state"],
        verbosity=-1,
    )
    
    model.fit(X_train, y_train)
    print("[MODEL] LightGBM trained")
    return model


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, config: dict
):
    """Train XGBoost as a challenger model.
    
    In banking, you always compare your main model against at least one
    alternative (champion-challenger framework). This is required by
    model risk management to prove you picked the best approach.
    """
    if not HAS_XGBOOST:
        raise ImportError("xgboost is not installed")
    
    params = config["model"]["xgboost"]
    
    model = xgb.XGBClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        scale_pos_weight=params["scale_pos_weight"],
        random_state=params["random_state"],
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
    )
    
    model.fit(X_train, y_train)
    print("[MODEL] XGBoost trained")
    return model


def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model"
) -> dict:
    """Evaluate with banking-standard metrics.
    
    We care about probabilities (not just 0/1) because banks use
    the actual PD for risk-based pricing, capital allocation under
    Basel III, and IFRS 9 provisioning.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_prob)
    gini = calculate_gini(y_test, y_prob)
    ks = calculate_ks_statistic(y_test, y_prob)
    
    metrics = {
        "model_name": model_name,
        "auc_roc": round(auc, 4),
        "gini": round(gini, 4),
        "ks_statistic": round(ks, 4),
    }
    
    print(f"\n{'='*50}")
    print(f"  {model_name} Results")
    print(f"{'='*50}")
    print(f"  AUC-ROC:       {auc:.4f}")
    print(f"  Gini:          {gini:.4f}")
    print(f"  KS Statistic:  {ks:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    print(f"  FN={cm[1][0]:,}  TP={cm[1][1]:,}")
    
    return metrics


def compare_models(metrics_list: list) -> pd.DataFrame:
    """Side-by-side comparison of champion vs challenger."""
    comparison = pd.DataFrame(metrics_list).set_index("model_name")
    
    print(f"\n{'='*50}")
    print("  Champion-Challenger Comparison")
    print(f"{'='*50}")
    print(comparison.to_string())
    
    best = comparison["gini"].idxmax()
    print(f"\n  Best model: {best} (Gini = {comparison.loc[best, 'gini']:.4f})")
    
    return comparison


def save_model(model, feature_names: list, config: dict) -> None:
    """Save model + feature names. Both are needed for serving."""
    model_dir = config["paths"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, config["paths"]["model_file"])
    joblib.dump(feature_names, config["paths"]["feature_names_file"])
    
    print(f"[SAVE] Model -> {config['paths']['model_file']}")
    print(f"[SAVE] Features -> {config['paths']['feature_names_file']}")


def load_model(config: dict) -> tuple:
    """Load saved model and feature names."""
    model = joblib.load(config["paths"]["model_file"])
    feature_names = joblib.load(config["paths"]["feature_names_file"])
    print(f"[LOAD] Model loaded from {config['paths']['model_file']}")
    return model, feature_names


if __name__ == "__main__":
    from data_loader import load_config, get_train_test_split
    from feature_engineering import engineer_features
    
    config = load_config()
    X_train, X_test, y_train, y_test = get_train_test_split(config)
    X_train, X_test, woe_maps, feature_names = engineer_features(
        X_train, X_test, y_train, config
    )
    
    lgbm = train_lightgbm(X_train, y_train, config)
    xgb_model = train_xgboost(X_train, y_train, config)
    
    m1 = evaluate_model(lgbm, X_test, y_test, "LightGBM")
    m2 = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    compare_models([m1, m2])
