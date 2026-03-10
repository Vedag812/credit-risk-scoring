"""
explainability.py - SHAP model explanations

SHAP (from game theory) shows how each feature pushes a prediction
up or down relative to the baseline. banks need this for two reasons:
regulators require explainability, and customers have a right to know
why they were denied credit.
"""

import os
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore")


def create_shap_explainer(model, X_train: pd.DataFrame, config: dict):
    """Create a TreeExplainer and save it.
    
    TreeExplainer is exact for tree models (not an approximation).
    Way faster than the model-agnostic KernelExplainer.
    """
    explainer = shap.TreeExplainer(model)
    
    explainer_path = config["paths"]["shap_explainer_file"]
    os.makedirs(os.path.dirname(explainer_path), exist_ok=True)
    joblib.dump(explainer, explainer_path)
    print(f"[SHAP] Explainer saved to {explainer_path}")
    
    return explainer


def global_feature_importance(
    explainer, X_test: pd.DataFrame, save_dir: str = "models"
) -> pd.DataFrame:
    """Which features matter most across all applicants?
    
    This gives you the overall picture - e.g., "loan_percent_income and
    interest rate are the two biggest drivers of default prediction."
    """
    shap_values = explainer.shap_values(X_test)
    
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values
    
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    
    # summary plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_test, show=False, max_display=15)
    plt.title("SHAP Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_global_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_test, plot_type="bar", show=False, max_display=15)
    plt.title("Top 15 Features by Mean |SHAP Value|", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_bar_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"[SHAP] Plots saved to {save_dir}")
    return importance_df


def local_explanation(
    explainer, X_single: pd.DataFrame, save_dir: str = "models", idx: int = 0
) -> dict:
    """Explain a single applicant's score.
    
    This is what a loan officer would see: "this person got flagged
    because their loan-to-income ratio is high and they have a
    previous default on file."
    """
    shap_values = explainer.shap_values(X_single)
    
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values
    
    if len(shap_vals.shape) == 1:
        shap_vals = shap_vals.reshape(1, -1)
    
    contributions = {}
    for i, col in enumerate(X_single.columns):
        contributions[col] = round(float(shap_vals[0][i]), 4)
    
    sorted_contributions = dict(
        sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    )
    
    # waterfall plot
    try:
        plt.figure(figsize=(10, 6))
        explanation = shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value if not isinstance(
                explainer.expected_value, list
            ) else explainer.expected_value[1],
            data=X_single.values[0],
            feature_names=list(X_single.columns),
        )
        shap.waterfall_plot(explanation, show=False, max_display=10)
        plt.title(f"Applicant #{idx} Breakdown", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"shap_waterfall_{idx}.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[SHAP] Waterfall plot skipped: {e}")
    
    return sorted_contributions


def explain_model(model, X_train: pd.DataFrame, X_test: pd.DataFrame, config: dict):
    """Run the full explainability pipeline."""
    print("\n" + "="*50)
    print("  SHAP Analysis")
    print("="*50)
    
    explainer = create_shap_explainer(model, X_train, config)
    
    # use a sample for speed (SHAP on 30K+ rows takes a while)
    X_sample = X_test.sample(min(500, len(X_test)), random_state=42)
    importance_df = global_feature_importance(
        explainer, X_sample, save_dir=config["paths"]["model_dir"]
    )
    
    print("\nTop 10 features:")
    print(importance_df.head(10).to_string(index=False))
    
    # single applicant example
    print("\nSample applicant explanation:")
    sample = X_test.iloc[[0]]
    contribs = local_explanation(explainer, sample, save_dir=config["paths"]["model_dir"])
    
    for feat, val in list(contribs.items())[:5]:
        direction = "increases risk" if val > 0 else "decreases risk"
        print(f"  {feat}: {val:+.4f} ({direction})")
    
    return explainer, importance_df
