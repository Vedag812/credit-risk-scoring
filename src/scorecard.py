"""
scorecard.py - convert ML probabilities to a credit score

banks don't tell customers "your probability of default is 0.23".
they give a score like 720 (think FICO or CIBIL). this module
converts the LightGBM output into a points-based score using the
standard PDO (Points to Double the Odds) method.

the math:
  Score = Offset + Factor * ln(odds)
  odds  = (1 - PD) / PD
  Factor = PDO / ln(2)
  Offset = base_score - Factor * ln(base_odds)
"""

import numpy as np
import pandas as pd


def calculate_scorecard_params(config: dict) -> tuple:
    """Get the Factor and Offset from config."""
    base_score = config["scorecard"]["base_score"]
    base_odds = config["scorecard"]["base_odds"]
    pdo = config["scorecard"]["pdo"]
    
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return factor, offset


def probability_to_score(probability: float, config: dict) -> int:
    """Convert a default probability to a credit score.
    Higher score = safer borrower.
    """
    factor, offset = calculate_scorecard_params(config)
    probability = np.clip(probability, 0.001, 0.999)
    odds = (1 - probability) / probability
    score = offset + factor * np.log(odds)
    return int(round(score))


def batch_probability_to_score(probabilities: np.ndarray, config: dict) -> np.ndarray:
    """Score a whole array at once."""
    factor, offset = calculate_scorecard_params(config)
    probabilities = np.clip(probabilities, 0.001, 0.999)
    odds = (1 - probabilities) / probabilities
    scores = offset + factor * np.log(odds)
    return np.round(scores).astype(int)


def get_risk_category(score: int) -> str:
    """Map score to a risk band (similar to FICO tiers)."""
    if score >= 750:
        return "Excellent"
    elif score >= 700:
        return "Good"
    elif score >= 650:
        return "Fair"
    elif score >= 600:
        return "Poor"
    else:
        return "Very Poor"


def generate_scorecard_report(
    model, X_test: pd.DataFrame, y_test: pd.Series, config: dict
) -> pd.DataFrame:
    """Full scorecard analysis showing score distribution and default rates per band."""
    print("\n" + "=" * 50)
    print("  Scorecard Report")
    print("=" * 50)
    
    probabilities = model.predict_proba(X_test)[:, 1]
    scores = batch_probability_to_score(probabilities, config)
    
    report_df = pd.DataFrame({
        "probability_of_default": probabilities,
        "credit_score": scores,
        "risk_category": [get_risk_category(s) for s in scores],
        "actual_default": y_test.values,
    })
    
    score_bands = pd.cut(
        scores,
        bins=[0, 550, 600, 650, 700, 750, 1000],
        labels=["<550", "550-599", "600-649", "650-699", "700-749", "750+"],
    )
    
    band_stats = report_df.groupby(score_bands, observed=True).agg(
        count=("credit_score", "size"),
        avg_pd=("probability_of_default", "mean"),
        actual_default_rate=("actual_default", "mean"),
    )
    band_stats["count_pct"] = band_stats["count"] / band_stats["count"].sum() * 100
    
    print(f"\n  {'Band':>10} | {'Count':>6} | {'% Total':>8} | {'Avg PD':>8} | {'Default Rate':>12}")
    print("  " + "-" * 60)
    for band, row in band_stats.iterrows():
        print(f"  {str(band):>10} | {row['count']:>6.0f} | {row['count_pct']:>7.1f}% | "
              f"{row['avg_pd']:>7.2%} | {row['actual_default_rate']:>11.2%}")
    
    print(f"\n  Mean Score: {scores.mean():.0f} | Median: {np.median(scores):.0f} | "
          f"Std: {scores.std():.0f}")
    
    for cat in ["Excellent", "Good", "Fair", "Poor", "Very Poor"]:
        count = (report_df["risk_category"] == cat).sum()
        if count > 0:
            pct = count / len(report_df) * 100
            dr = report_df.loc[report_df["risk_category"] == cat, "actual_default"].mean()
            print(f"    {cat:>12}: {count:>5} ({pct:>5.1f}%) | Default Rate: {dr:.2%}")
    
    return report_df


if __name__ == "__main__":
    from data_loader import load_config
    config = load_config()
    print("Score examples:")
    for pd_val in [0.01, 0.05, 0.10, 0.20, 0.50, 0.80]:
        score = probability_to_score(pd_val, config)
        print(f"  PD={pd_val:.2f} -> Score={score} ({get_risk_category(score)})")
