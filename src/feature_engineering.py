"""
feature_engineering.py - create banking-relevant features

this is arguably the most important step. raw columns alone don't capture
what banks actually care about. things like debt-to-income ratio, how long
someone has been employed relative to their age, and weight-of-evidence
encoding are standard in every credit risk team.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


def create_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add standard financial ratios that banks look at.
    
    - loan_to_income: how much are they borrowing relative to what they make?
    - income_per_emp_year: are they on a good career trajectory?
    - credit_hist_to_age: if someone is 40 but only has 2 years of credit history,
      that's a red flag
    - interest_burden: actual cost of the loan relative to their income
    """
    df = df.copy()
    
    df["loan_to_income"] = df["loan_amnt"] / (df["person_income"] + 1)
    df["income_per_emp_year"] = df["person_income"] / (df["person_emp_length"] + 1)
    df["credit_hist_to_age"] = df["cb_person_cred_hist_length"] / (df["person_age"] + 1)
    df["interest_burden"] = (df["loan_int_rate"] * df["loan_amnt"]) / (df["person_income"] + 1)
    
    print(f"[FEAT] Created 4 financial ratio features")
    return df


def create_risk_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Bin continuous variables into risk groups.
    
    Banks don't just use raw numbers because the relationship between
    something like age and default isn't linear. A 22-year-old and a
    70-year-old are both higher risk than a 40-year-old, but for
    completely different reasons.
    """
    df = df.copy()
    
    df["age_bucket"] = pd.cut(
        df["person_age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["very_young", "young", "mid_career", "senior", "retired"],
    )
    
    df["emp_stability"] = pd.cut(
        df["person_emp_length"],
        bins=[-1, 1, 3, 7, 15, 100],
        labels=["new_hire", "junior", "mid_level", "senior", "veteran"],
    )
    
    df["income_tier"] = pd.cut(
        df["person_income"],
        bins=[0, 30000, 60000, 100000, 200000, float("inf")],
        labels=["low", "lower_mid", "mid", "upper_mid", "high"],
    )
    
    print(f"[FEAT] Created 3 risk bucket features")
    return df


def encode_woe(
    df: pd.DataFrame, target_col: str, columns: list, is_training: bool = True,
    woe_maps: dict = None
) -> tuple:
    """Weight-of-Evidence encoding for categorical variables.
    
    WoE = ln(% of non-defaults / % of defaults) for each category.
    
    Positive WoE = safer category, negative = riskier.
    
    Banks use this over one-hot encoding because:
    1. It captures how predictive each category is
    2. One column per feature instead of N dummies
    3. Regulators can directly audit the WoE values
    
    The +0.5 smoothing prevents division by zero for rare categories.
    """
    df = df.copy()
    
    if woe_maps is None:
        woe_maps = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        woe_col_name = f"{col}_woe"
        
        if is_training:
            total_events = (df[target_col] == 1).sum()
            total_non_events = (df[target_col] == 0).sum()
            
            woe_map = {}
            for cat in df[col].unique():
                mask = df[col] == cat
                events = (df.loc[mask, target_col] == 1).sum()
                non_events = (df.loc[mask, target_col] == 0).sum()
                
                # smoothing to avoid log(0)
                dist_events = (events + 0.5) / (total_events + 1)
                dist_non_events = (non_events + 0.5) / (total_non_events + 1)
                
                woe_map[cat] = np.log(dist_non_events / dist_events)
            
            woe_maps[col] = woe_map
        
        df[woe_col_name] = df[col].map(woe_maps.get(col, {})).fillna(0)
    
    print(f"[FEAT] Applied WoE encoding to {len(columns)} columns")
    return df, woe_maps


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode remaining categorical columns."""
    df = df.copy()
    
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
        print(f"[FEAT] One-hot encoded {len(categorical_cols)} categorical columns")
    
    return df


def create_interaction_features(df: pd.DataFrame, interaction_pairs: list) -> pd.DataFrame:
    """Multiply pairs of features together to capture compound risk.
    
    For example, a high loan_percent_income AND short employment length
    together is way riskier than either alone. Tree models can find these
    on their own, but making them explicit helps the model learn faster
    and makes the feature list more auditable.
    """
    df = df.copy()
    
    for col1, col2 in interaction_pairs:
        if col1 in df.columns and col2 in df.columns:
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
    
    n_created = len([p for p in interaction_pairs if p[0] in df.columns and p[1] in df.columns])
    print(f"[FEAT] Created {n_created} interaction features")
    return df


def engineer_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> tuple:
    """Run the full feature engineering pipeline.
    
    Important: WoE is fitted on training data ONLY, then applied to test.
    This prevents data leakage (test data shouldn't influence training).
    """
    # financial ratios (no leakage risk since they don't use the target)
    X_train = create_financial_ratios(X_train)
    X_test = create_financial_ratios(X_test)
    
    # risk buckets
    X_train = create_risk_buckets(X_train)
    X_test = create_risk_buckets(X_test)
    
    # interactions
    interaction_pairs = config["feature_engineering"]["interaction_pairs"]
    X_train = create_interaction_features(X_train, interaction_pairs)
    X_test = create_interaction_features(X_test, interaction_pairs)
    
    # WoE encoding (fit on train, apply to both)
    woe_columns = config["feature_engineering"]["woe_columns"]
    
    X_train_with_target = X_train.copy()
    X_train_with_target["loan_status"] = y_train.values
    
    X_train_with_target, woe_maps = encode_woe(
        X_train_with_target, "loan_status", woe_columns, is_training=True
    )
    X_train = X_train_with_target.drop("loan_status", axis=1)
    
    X_test, _ = encode_woe(
        X_test, None, woe_columns, is_training=False, woe_maps=woe_maps
    )
    
    # one-hot encode whatever's left
    X_train = encode_categoricals(X_train)
    X_test = encode_categoricals(X_test)
    
    # make sure both sets have the same columns
    common_cols = sorted(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    feature_names = list(X_train.columns)
    print(f"\n[FEAT] Final feature count: {len(feature_names)}")
    
    return X_train, X_test, woe_maps, feature_names


if __name__ == "__main__":
    from data_loader import load_config, get_train_test_split
    
    config = load_config()
    X_train, X_test, y_train, y_test = get_train_test_split(config)
    X_train, X_test, woe_maps, feature_names = engineer_features(
        X_train, X_test, y_train, config
    )
    print(f"\nFeature names: {feature_names[:10]}...")
    print(f"X_train shape: {X_train.shape}")
