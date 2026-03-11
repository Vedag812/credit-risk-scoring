"""
data_loader.py - download, load and clean the credit risk dataset

handles the messy stuff: missing values, outliers, data types.
real banking data is never clean, so this module does the scrubbing.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import warnings

warnings.filterwarnings("ignore")


def load_config(config_path: str = None) -> dict:
    """Load config from YAML.
    
    We keep all hyperparams and paths in config.yaml so nothing
    is hardcoded. In a real bank, config would be version-controlled
    separately so the model risk team can audit any parameter changes.
    """
    # resolve project root (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if config_path is None:
        config_path = os.path.join(project_root, "config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # make all relative paths absolute (relative to project root)
    if "data" in config:
        raw = config["data"].get("raw_path", "")
        if raw and not os.path.isabs(raw):
            config["data"]["raw_path"] = os.path.join(project_root, raw)
    
    if "paths" in config:
        for key in config["paths"]:
            val = config["paths"][key]
            if isinstance(val, str) and not os.path.isabs(val):
                config["paths"][key] = os.path.join(project_root, val)
    
    return config


def download_dataset(config: dict) -> str:
    """Download the dataset from Kaggle, or use local copy if it exists."""
    raw_path = config["data"]["raw_path"]
    
    if os.path.exists(raw_path):
        print(f"[INFO] Dataset already exists at {raw_path}")
        return raw_path
    
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    
    # check for kaggle credentials
    user = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    
    if not user or not key:
        print("[INFO] Kaggle credentials not found in environment. Using fallback...")
        _generate_synthetic_dataset(raw_path)
        return raw_path

    try:
        import kaggle
        kaggle_dataset = config["data"]["kaggle_dataset"]
        print(f"[INFO] Downloading from Kaggle: {kaggle_dataset}")
        kaggle.api.dataset_download_files(
            kaggle_dataset,
            path=os.path.dirname(raw_path),
            unzip=True,
        )
        print(f"[INFO] Downloaded to {raw_path}")
    except Exception as e:
        print(f"[WARN] Kaggle download failed: {e}")
        print("[INFO] Generating synthetic dataset as fallback...")
        _generate_synthetic_dataset(raw_path)

    return raw_path


def _generate_synthetic_dataset(output_path: str, n_samples: int = 32581) -> None:
    """Generate synthetic data that matches the real Kaggle dataset schema.
    
    This is just a fallback for when Kaggle credentials aren't set up.
    The column names and distributions roughly match the actual dataset.
    """
    np.random.seed(42)
    n = n_samples
    
    data = {
        "person_age": np.random.randint(20, 70, n),
        "person_income": np.random.lognormal(mean=10.5, sigma=0.8, size=n).astype(int),
        "person_home_ownership": np.random.choice(
            ["RENT", "MORTGAGE", "OWN", "OTHER"], n, p=[0.42, 0.41, 0.13, 0.04]
        ),
        "person_emp_length": np.random.exponential(scale=5, size=n).astype(int),
        "loan_intent": np.random.choice(
            ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"],
            n, p=[0.17, 0.15, 0.15, 0.20, 0.18, 0.15]
        ),
        "loan_grade": np.random.choice(
            ["A", "B", "C", "D", "E", "F", "G"], n, p=[0.30, 0.28, 0.20, 0.12, 0.06, 0.03, 0.01]
        ),
        "loan_amnt": np.random.lognormal(mean=9.0, sigma=0.7, size=n).astype(int),
        "loan_int_rate": np.round(np.random.uniform(5.5, 24.0, n), 2),
        "loan_percent_income": np.round(np.random.uniform(0.0, 0.8, n), 2),
        "cb_person_default_on_file": np.random.choice(["Y", "N"], n, p=[0.18, 0.82]),
        "cb_person_cred_hist_length": np.random.randint(2, 30, n),
    }
    
    # target: default probability goes up with higher rates, higher loan burden,
    # past defaults, and worse loan grades
    default_prob = (
        0.05
        + 0.02 * (data["loan_int_rate"] - 10) / 10
        + 0.15 * data["loan_percent_income"]
        + 0.10 * (data["cb_person_default_on_file"] == "Y").astype(float)
        + 0.05 * np.isin(data["loan_grade"], ["D", "E", "F", "G"]).astype(float)
    )
    default_prob = np.clip(default_prob, 0.02, 0.85)
    data["loan_status"] = np.random.binomial(1, default_prob)
    
    # sprinkle in some nulls to make it realistic
    df = pd.DataFrame(data)
    df.loc[np.random.random(n) < 0.05, "person_emp_length"] = np.nan
    df.loc[np.random.random(n) < 0.03, "loan_int_rate"] = np.nan
    
    df.to_csv(output_path, index=False)
    print(f"[INFO] Synthetic dataset saved to {output_path} ({n} rows)")


def load_and_clean(config: dict) -> pd.DataFrame:
    """Load the CSV and clean it up.
    
    Three things happen here:
    1. Missing values get imputed (median for numbers, mode for text)
    2. Outliers get capped using IQR (so a 144-year-old doesn't break the model)
    3. Data types get fixed
    
    I chose capping over dropping because in credit risk you want to keep
    every applicant in the data. Dropping rows would bias the sample.
    """
    raw_path = download_dataset(config)
    df = pd.read_csv(raw_path)
    print(f"[INFO] Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[CLEAN] {col}: filled nulls with median={median_val:.2f}")
    
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"[CLEAN] {col}: filled nulls with mode={mode_val}")
    
    # cap outliers with IQR
    # IQR is better than z-score here because income distributions are skewed
    for col in ["person_age", "person_income", "person_emp_length"]:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower, upper=upper)
            print(f"[CLEAN] {col}: capped to [{lower:.0f}, {upper:.0f}]")
    
    df["loan_status"] = df["loan_status"].astype(int)
    
    print(f"[INFO] After cleaning: {df.shape[0]} rows, default rate={df['loan_status'].mean():.2%}")
    return df


def get_train_test_split(config: dict) -> tuple:
    """Load data and do a stratified train/test split.
    
    Stratified splitting keeps the default rate the same in both sets.
    Without this, you might get unlucky and have way more defaults in
    test than train, which would make evaluation unreliable.
    """
    df = load_and_clean(config)
    
    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )
    
    print(f"[SPLIT] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"[SPLIT] Train default rate: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    config = load_config()
    X_train, X_test, y_train, y_test = get_train_test_split(config)
    print("\nSample features:")
    print(X_train.head())
