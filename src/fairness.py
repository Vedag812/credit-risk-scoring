"""
fairness.py - check if the model discriminates by age

any credit model at a bank has to comply with ECOA (Equal Credit
Opportunity Act) and the EU AI Act. if our model is denying older
applicants at a higher rate just because of their age, that's a
legal and ethical problem.

we check three things per age group:
  1. what % gets approved (selection rate)
  2. how well we catch actual defaults (true positive rate)
  3. how often we falsely flag someone (false positive rate)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")


def create_sensitive_groups(X_test: pd.DataFrame) -> pd.Series:
    """Bin applicants into age groups for fairness comparison.
    
    In a real bank you'd also check across gender, race (using HMDA data),
    and geography. Age is what we have in this dataset.
    """
    if "person_age" in X_test.columns:
        groups = pd.cut(
            X_test["person_age"],
            bins=[0, 30, 45, 100],
            labels=["Young (<30)", "Mid (30-45)", "Senior (45+)"],
        )
    else:
        groups = pd.Series("Unknown", index=X_test.index)
    
    return groups


def calculate_fairness_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive_groups: pd.Series
) -> dict:
    """Calculate selection rates and error rates per group."""
    results = {}
    unique_groups = sensitive_groups.dropna().unique()
    
    selection_rates = {}
    true_positive_rates = {}
    false_positive_rates = {}
    
    for group in unique_groups:
        mask = sensitive_groups == group
        group_pred = y_pred[mask]
        group_true = y_true[mask]
        
        if mask.sum() == 0:
            continue
        
        selection_rates[group] = (group_pred == 0).mean()
        
        defaults = group_true == 1
        if defaults.sum() > 0:
            true_positive_rates[group] = (group_pred[defaults] == 1).sum() / defaults.sum()
        else:
            true_positive_rates[group] = 0.0
        
        non_defaults = group_true == 0
        if non_defaults.sum() > 0:
            false_positive_rates[group] = (group_pred[non_defaults] == 1).sum() / non_defaults.sum()
        else:
            false_positive_rates[group] = 0.0
    
    if selection_rates:
        rates = list(selection_rates.values())
        dp_difference = max(rates) - min(rates)
    else:
        dp_difference = 0.0
    
    return {
        "selection_rates": selection_rates,
        "true_positive_rates": true_positive_rates,
        "false_positive_rates": false_positive_rates,
        "demographic_parity_difference": round(dp_difference, 4),
    }


def run_fairness_audit(
    y_pred: np.ndarray, X_test_raw: pd.DataFrame, y_test: pd.Series,
    save_dir: str = "models"
) -> dict:
    """Full fairness audit with visual report.
    
    Takes pre-computed predictions and raw test data (for age groups).
    The threshold is 0.10 for demographic parity difference. Above that,
    the model needs to be investigated for potential bias.
    """
    print("\n" + "=" * 50)
    print("  Fairness Audit")
    print("=" * 50)
    
    sensitive_groups = create_sensitive_groups(X_test_raw)
    
    if sensitive_groups.nunique() <= 1:
        print("[FAIR] Can't run audit - no sensitive groups (age was already encoded)")
        return {"status": "skipped", "reason": "no_sensitive_groups"}
    
    metrics = calculate_fairness_metrics(y_test.values, y_pred, sensitive_groups)
    
    print("\n  Approval rates per group:")
    for group, rate in metrics["selection_rates"].items():
        bar = "#" * int(rate * 40)
        print(f"    {group:>15}: {rate:.2%}  {bar}")
    
    print(f"\n  Demographic Parity Difference: {metrics['demographic_parity_difference']:.4f}")
    
    threshold = 0.10
    if metrics["demographic_parity_difference"] < threshold:
        print(f"  PASS - below {threshold} threshold")
        metrics["fairness_status"] = "PASS"
    else:
        print(f"  WARNING - above {threshold} threshold, needs investigation")
        metrics["fairness_status"] = "INVESTIGATE"
    
    print("\n  Default detection rate per group:")
    for group, rate in metrics["true_positive_rates"].items():
        print(f"    {group:>15}: {rate:.2%}")
    
    print("\n  False alarm rate per group:")
    for group, rate in metrics["false_positive_rates"].items():
        print(f"    {group:>15}: {rate:.2%}")
    
    _plot_fairness(metrics, save_dir)
    return metrics


def _plot_fairness(metrics: dict, save_dir: str) -> None:
    """Create a visual comparison across groups."""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        groups = list(metrics["selection_rates"].keys())
        
        # approval rates
        rates = [metrics["selection_rates"].get(g, 0) for g in groups]
        colors = ["#2ecc71" if r > 0.7 else "#e74c3c" for r in rates]
        axes[0].barh(groups, rates, color=colors, edgecolor="white")
        axes[0].set_title("Approval Rate by Group", fontsize=12)
        axes[0].set_xlim(0, 1)
        
        # true positive rates
        tpr = [metrics["true_positive_rates"].get(g, 0) for g in groups]
        axes[1].barh(groups, tpr, color="#3498db", edgecolor="white")
        axes[1].set_title("Default Detection Rate", fontsize=12)
        axes[1].set_xlim(0, 1)
        
        # false positive rates
        fpr = [metrics["false_positive_rates"].get(g, 0) for g in groups]
        axes[2].barh(groups, fpr, color="#e67e22", edgecolor="white")
        axes[2].set_title("False Alarm Rate", fontsize=12)
        axes[2].set_xlim(0, 1)
        
        plt.suptitle("Fairness Audit Results", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "fairness_audit.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n[FAIR] Fairness plot saved to {save_dir}")
    except Exception as e:
        print(f"[FAIR] Plot skipped: {e}")
