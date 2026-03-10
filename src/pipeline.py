"""
pipeline.py - end-to-end training orchestrator

run: python -m src.pipeline

this is the single entry point for the full ML pipeline.
in production, this would be scheduled via an orchestration tool
(like Airflow) to run on fresh data periodically.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_config, get_train_test_split
from src.feature_engineering import engineer_features
from src.model import (
    train_lightgbm, evaluate_model, compare_models, save_model, HAS_XGBOOST,
)
from src.explainability import explain_model
from src.fairness import run_fairness_audit
from src.scorecard import generate_scorecard_report


def run_pipeline(config_path: str = "config.yaml") -> dict:
    """Run the full pipeline end to end."""
    start_time = time.time()
    
    print("=" * 60)
    print("  CREDIT RISK SCORING PIPELINE")
    print("=" * 60)
    
    config = load_config(config_path)
    
    # step 1: load and clean
    print("\n[1/8] Loading and cleaning data...")
    X_train, X_test, y_train, y_test = get_train_test_split(config)
    X_test_raw = X_test.copy()
    
    # step 2: feature engineering
    print("\n[2/8] Engineering features...")
    X_train_fe, X_test_fe, woe_maps, feature_names = engineer_features(
        X_train, X_test, y_train, config
    )
    
    # step 3: train models
    print("\n[3/8] Training models...")
    lgbm_model = train_lightgbm(X_train_fe, y_train, config)
    
    all_metrics = []
    lgbm_metrics = evaluate_model(lgbm_model, X_test_fe, y_test, "LightGBM")
    all_metrics.append(lgbm_metrics)
    
    # train xgboost if available (champion-challenger comparison)
    xgb_metrics = None
    if HAS_XGBOOST:
        from src.model import train_xgboost
        xgb_model = train_xgboost(X_train_fe, y_train, config)
        xgb_metrics = evaluate_model(xgb_model, X_test_fe, y_test, "XGBoost")
        all_metrics.append(xgb_metrics)
    
    # step 4: compare and pick best
    print("\n[4/8] Comparing models...")
    if len(all_metrics) > 1:
        compare_models(all_metrics)
    
    best_model = lgbm_model
    best_name = "LightGBM"
    best_metrics = lgbm_metrics
    
    if xgb_metrics and xgb_metrics["gini"] > lgbm_metrics["gini"]:
        from src.model import train_xgboost
        best_model = xgb_model
        best_name = "XGBoost"
        best_metrics = xgb_metrics
    
    # step 5: save
    print(f"\n[5/8] Saving best model ({best_name})...")
    save_model(best_model, feature_names, config)
    
    # step 6: SHAP
    print("\n[6/8] Generating SHAP explanations...")
    explainer, importance_df = explain_model(best_model, X_train_fe, X_test_fe, config)
    
    # step 7: fairness (use X_test_raw for age groups, but predictions from engineered data)
    print("\n[7/8] Running fairness audit...")
    y_pred_fair = best_model.predict(X_test_fe)
    fairness_results = run_fairness_audit(
        y_pred_fair, X_test_raw, y_test, save_dir=config["paths"]["model_dir"]
    )
    
    # step 8: scorecard
    print("\n[8/8] Generating scorecard...")
    scorecard_report = generate_scorecard_report(best_model, X_test_fe, y_test, config)
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Best Model:      {best_name}")
    print(f"  AUC-ROC:         {best_metrics['auc_roc']}")
    print(f"  Gini:            {best_metrics['gini']}")
    print(f"  KS Statistic:    {best_metrics['ks_statistic']}")
    print(f"  Fairness:        {fairness_results.get('fairness_status', 'N/A')}")
    print(f"  Time:            {elapsed:.1f}s")
    print("=" * 60)
    
    return {
        "best_model_name": best_name,
        "lgbm_metrics": lgbm_metrics,
        "xgb_metrics": xgb_metrics,
        "fairness_results": fairness_results,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    results = run_pipeline()
