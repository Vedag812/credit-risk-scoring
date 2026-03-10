"""
create_notebook.py - generates a Kaggle-compatible Jupyter notebook
for credit risk EDA and modeling

Run: python notebooks/create_notebook.py
Output: notebooks/credit_risk_eda.ipynb
"""

import json
import os


def make_markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def generate_notebook():
    cells = []

    # --- Title ---
    cells.append(make_markdown_cell("""# Credit Risk Scoring - EDA & Modeling

Predicting loan defaults using LightGBM with SHAP explainability and a credit scorecard.

**What this notebook covers:**
- Data exploration and cleaning
- Feature engineering (financial ratios, WoE encoding)
- Model training (LightGBM vs XGBoost)
- SHAP explanations (global + local)
- Credit scorecard conversion
- Fairness analysis across age groups

Dataset: [Credit Risk Dataset on Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)"""))

    # --- Imports ---
    cells.append(make_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import lightgbm as lgb
import xgboost as xgb
import shap
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_columns', 20)

print("Libraries loaded")"""))

    # --- Load data ---
    cells.append(make_markdown_cell("""## 1. Load and Explore the Data"""))

    cells.append(make_code_cell("""# load the dataset
df = pd.read_csv('/kaggle/input/credit-risk-dataset/credit_risk_dataset.csv')
print(f"Dataset shape: {df.shape}")
df.head()"""))

    cells.append(make_code_cell("""# basic stats
df.describe()"""))

    cells.append(make_code_cell("""# check for missing values and data types
print("Missing values:")
print(df.isnull().sum())
print(f"\\nDefault rate: {df['loan_status'].mean():.2%}")"""))

    # --- EDA ---
    cells.append(make_markdown_cell("""## 2. Exploratory Data Analysis"""))

    cells.append(make_code_cell("""fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# age distribution
axes[0,0].hist(df['person_age'], bins=50, color='steelblue', edgecolor='white')
axes[0,0].set_title('Age Distribution')
axes[0,0].set_xlabel('Age')

# income distribution (log scale since it's skewed)
axes[0,1].hist(df['person_income'], bins=50, color='coral', edgecolor='white')
axes[0,1].set_title('Income Distribution')
axes[0,1].set_xlabel('Income')

# loan amount
axes[0,2].hist(df['loan_amnt'], bins=50, color='mediumseagreen', edgecolor='white')
axes[0,2].set_title('Loan Amount Distribution')
axes[0,2].set_xlabel('Amount')

# default rate by loan grade
grade_default = df.groupby('loan_grade')['loan_status'].mean().sort_index()
axes[1,0].bar(grade_default.index, grade_default.values, color='steelblue')
axes[1,0].set_title('Default Rate by Loan Grade')
axes[1,0].set_ylabel('Default Rate')

# default rate by home ownership
home_default = df.groupby('person_home_ownership')['loan_status'].mean()
axes[1,1].bar(home_default.index, home_default.values, color='coral')
axes[1,1].set_title('Default Rate by Home Ownership')
axes[1,1].set_ylabel('Default Rate')

# default rate by loan intent
intent_default = df.groupby('loan_intent')['loan_status'].mean().sort_values()
axes[1,2].barh(intent_default.index, intent_default.values, color='mediumseagreen')
axes[1,2].set_title('Default Rate by Loan Intent')

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=150)
plt.show()"""))

    cells.append(make_code_cell("""# interest rate vs default
fig, ax = plt.subplots(figsize=(10, 5))
df.groupby(pd.cut(df['loan_int_rate'], bins=10))['loan_status'].mean().plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('Default Rate by Interest Rate Band')
ax.set_ylabel('Default Rate')
ax.set_xlabel('Interest Rate Band')
plt.tight_layout()
plt.show()"""))

    cells.append(make_code_cell("""# correlation heatmap (numeric columns only)
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='RdBu_r', center=0, fmt='.2f')
plt.title('Feature Correlations')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()"""))

    # --- Cleaning ---
    cells.append(make_markdown_cell("""## 3. Data Cleaning

Handle missing values and cap outliers. I use IQR capping instead of dropping rows
because in credit risk you want to keep every applicant in the analysis."""))

    cells.append(make_code_cell("""# fill missing values
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())
        print(f"Filled {col} nulls with median")

# cap outliers using IQR
for col in ['person_age', 'person_income', 'person_emp_length']:
    if col in df.columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        before = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        print(f"Capped {before} outliers in {col}")

print(f"\\nClean dataset: {df.shape[0]} rows")"""))

    # --- Feature Engineering ---
    cells.append(make_markdown_cell("""## 4. Feature Engineering

Creating features that banks actually use: debt ratios, employment stability buckets,
and Weight-of-Evidence encoding (the industry standard for credit scorecards)."""))

    cells.append(make_code_cell("""# financial ratios
df['loan_to_income'] = df['loan_amnt'] / (df['person_income'] + 1)
df['income_per_emp_year'] = df['person_income'] / (df['person_emp_length'] + 1)
df['credit_hist_to_age'] = df['cb_person_cred_hist_length'] / (df['person_age'] + 1)
df['interest_burden'] = (df['loan_int_rate'] * df['loan_amnt']) / (df['person_income'] + 1)

# interaction feature
df['loan_pct_x_emp'] = df['loan_percent_income'] * df['person_emp_length']

print("Created 5 new features")
df[['loan_to_income', 'income_per_emp_year', 'credit_hist_to_age', 'interest_burden']].describe()"""))

    cells.append(make_code_cell("""# Weight-of-Evidence encoding
# WoE = ln(% non-defaults / % defaults) per category
# positive = safer, negative = riskier

def calculate_woe(df, col, target='loan_status'):
    total_good = (df[target] == 0).sum()
    total_bad = (df[target] == 1).sum()
    
    woe_map = {}
    for cat in df[col].unique():
        mask = df[col] == cat
        good = (df.loc[mask, target] == 0).sum()
        bad = (df.loc[mask, target] == 1).sum()
        
        dist_good = (good + 0.5) / (total_good + 1)
        dist_bad = (bad + 0.5) / (total_bad + 1)
        woe_map[cat] = np.log(dist_good / dist_bad)
    
    return woe_map

# apply WoE to categorical columns
for col in ['person_home_ownership', 'loan_intent', 'loan_grade']:
    woe_map = calculate_woe(df, col)
    df[f'{col}_woe'] = df[col].map(woe_map)
    print(f"\\nWoE for {col}:")
    for k, v in sorted(woe_map.items(), key=lambda x: x[1]):
        marker = "safe" if v > 0 else "risky"
        print(f"  {k:>20}: {v:+.3f} ({marker})")"""))

    # --- Train/test split ---
    cells.append(make_markdown_cell("""## 5. Model Training

Training LightGBM (primary) and XGBoost (challenger) with stratified splitting."""))

    cells.append(make_code_cell("""# encode remaining categoricals and split
df_model = pd.get_dummies(df, drop_first=True, dtype=int)
df_model = df_model.replace([np.inf, -np.inf], np.nan).fillna(0)

X = df_model.drop('loan_status', axis=1)
y = df_model['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
print(f"Train default rate: {y_train.mean():.2%}")
print(f"Test default rate:  {y_test.mean():.2%}")"""))

    cells.append(make_code_cell("""# train LightGBM
lgbm = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    num_leaves=31, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=3, random_state=42, verbosity=-1
)
lgbm.fit(X_train, y_train)

# train XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=3, random_state=42, eval_metric='logloss',
    use_label_encoder=False, verbosity=0
)
xgb_model.fit(X_train, y_train)

print("Both models trained")"""))

    # --- Evaluation ---
    cells.append(make_markdown_cell("""## 6. Model Evaluation

Using banking-standard metrics: Gini (= 2*AUC - 1) and KS statistic."""))

    cells.append(make_code_cell("""def evaluate(model, X_test, y_test, name):
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    gini = 2 * auc - 1
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ks = max(tpr - fpr)
    
    print(f"{name}:")
    print(f"  AUC-ROC:      {auc:.4f}")
    print(f"  Gini:         {gini:.4f}")
    print(f"  KS Statistic: {ks:.4f}")
    print()
    
    return auc, gini, ks, y_prob

auc_lgb, gini_lgb, ks_lgb, prob_lgb = evaluate(lgbm, X_test, y_test, "LightGBM")
auc_xgb, gini_xgb, ks_xgb, prob_xgb = evaluate(xgb_model, X_test, y_test, "XGBoost")

# pick the winner
if gini_lgb >= gini_xgb:
    print(f"Winner: LightGBM (Gini {gini_lgb:.4f} vs {gini_xgb:.4f})")
    best_model, best_prob = lgbm, prob_lgb
else:
    print(f"Winner: XGBoost (Gini {gini_xgb:.4f} vs {gini_lgb:.4f})")
    best_model, best_prob = xgb_model, prob_xgb"""))

    cells.append(make_code_cell("""# ROC curves side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (name, prob, auc_val) in zip(axes, [
    ("LightGBM", prob_lgb, auc_lgb),
    ("XGBoost", prob_xgb, auc_xgb)
]):
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_val:.4f}')
    ax.plot([0,1], [0,1], 'k--', alpha=0.5)
    ax.set_title(f'{name} ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()

plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150)
plt.show()"""))

    cells.append(make_code_cell("""# classification report for best model
y_pred = (best_prob >= 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt=',', cmap='Blues',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()"""))

    # --- SHAP ---
    cells.append(make_markdown_cell("""## 7. SHAP Explainability

Banks need to explain model decisions (required by regulators like the OCC).
SHAP tells us which features drove each prediction and by how much."""))

    cells.append(make_code_cell("""# global feature importance
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test.sample(500, random_state=42))

if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

plt.figure(figsize=(10, 8))
shap.summary_plot(sv, X_test.sample(500, random_state=42), show=False, max_display=15)
plt.title('SHAP Feature Importance')
plt.tight_layout()
plt.savefig('shap_importance.png', dpi=150, bbox_inches='tight')
plt.show()"""))

    cells.append(make_code_cell("""# single applicant explanation (what a loan officer would see)
single = X_test.iloc[[0]]
sv_single = explainer.shap_values(single)
if isinstance(sv_single, list):
    sv_s = sv_single[1]
else:
    sv_s = sv_single

if len(sv_s.shape) == 1:
    sv_s = sv_s.reshape(1, -1)

base_val = explainer.expected_value
if isinstance(base_val, list):
    base_val = base_val[1]

exp = shap.Explanation(values=sv_s[0], base_values=base_val,
                       data=single.values[0], feature_names=list(single.columns))
plt.figure(figsize=(10, 6))
shap.waterfall_plot(exp, show=False, max_display=10)
plt.title('Why this applicant got their score')
plt.tight_layout()
plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.show()"""))

    # --- Scorecard ---
    cells.append(make_markdown_cell("""## 8. Credit Scorecard

Converting probabilities to a FICO-like credit score using the standard
Points-to-Double-Odds (PDO) method. Higher score = safer borrower."""))

    cells.append(make_code_cell("""# scorecard conversion
base_score, base_odds, pdo = 600, 50, 20
factor = pdo / np.log(2)
offset = base_score - factor * np.log(base_odds)

probs = best_model.predict_proba(X_test)[:, 1]
probs_clipped = np.clip(probs, 0.001, 0.999)
odds = (1 - probs_clipped) / probs_clipped
scores = np.round(offset + factor * np.log(odds)).astype(int)

# score distribution
plt.figure(figsize=(10, 5))
plt.hist(scores[y_test == 0], bins=50, alpha=0.6, label='No Default', color='steelblue')
plt.hist(scores[y_test == 1], bins=50, alpha=0.6, label='Default', color='coral')
plt.xlabel('Credit Score')
plt.ylabel('Count')
plt.title('Score Distribution by Default Status')
plt.legend()
plt.tight_layout()
plt.savefig('score_distribution.png', dpi=150)
plt.show()

# score band analysis
def risk_category(s):
    if s >= 750: return 'Excellent'
    elif s >= 700: return 'Good'
    elif s >= 650: return 'Fair'
    elif s >= 600: return 'Poor'
    else: return 'Very Poor'

results = pd.DataFrame({'score': scores, 'default': y_test.values, 'prob': probs})
results['category'] = results['score'].apply(risk_category)

band_summary = results.groupby('category').agg(
    count=('score', 'size'),
    avg_score=('score', 'mean'),
    default_rate=('default', 'mean')
).reindex(['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor'])

print("Score Band Analysis:")
print(band_summary.to_string())"""))

    # --- Fairness ---
    cells.append(make_markdown_cell("""## 9. Fairness Check

Checking if the model treats different age groups fairly.
This matters for ECOA (Equal Credit Opportunity Act) compliance."""))

    cells.append(make_code_cell("""# create age groups on original data
if 'person_age' in df.columns:
    test_ages = df.loc[X_test.index, 'person_age'] if len(df) == len(X) + len(y) else X_test.get('person_age', pd.Series())
    
    if len(test_ages) == 0:
        # reconstruct from index
        print("Age column not directly available after encoding")
        print("In production, track protected attributes separately")
    else:
        age_groups = pd.cut(test_ages, bins=[0,30,45,100], labels=['Young','Mid','Senior'])
        y_pred_fair = (probs >= 0.5).astype(int)
        
        print("Approval rates by age group:")
        for group in ['Young', 'Mid', 'Senior']:
            mask = age_groups == group
            if mask.sum() > 0:
                approval_rate = (y_pred_fair[mask] == 0).mean()
                print(f"  {group:>8}: {approval_rate:.2%}")
        
        rates = [(y_pred_fair[age_groups == g] == 0).mean() for g in ['Young','Mid','Senior'] if (age_groups == g).sum() > 0]
        dp_diff = max(rates) - min(rates)
        print(f"\\nDemographic Parity Difference: {dp_diff:.4f}")
        print(f"Status: {'PASS' if dp_diff < 0.10 else 'INVESTIGATE'} (threshold: 0.10)")
else:
    print("Age not available for fairness check")"""))

    # --- Conclusion ---
    cells.append(make_markdown_cell("""## Summary

| Component | What I did |
|-----------|-----------|
| Data cleaning | Imputed missing values, capped outliers with IQR |
| Features | Financial ratios, WoE encoding, interaction features |
| Models | LightGBM (champion) vs XGBoost (challenger) |
| Evaluation | Gini, KS statistic, AUC-ROC |
| Explainability | SHAP global + local explanations |
| Scorecard | Probability to FICO-like score using PDO method |
| Fairness | Demographic parity check across age groups |

**Full project with FastAPI, Streamlit dashboard, Docker, and CI:**
[GitHub repo](https://github.com/YOUR_USERNAME/credit-risk-scoring)"""))

    # --- Build notebook ---
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "cells": cells
    }

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "credit_risk_eda.ipynb")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"Notebook saved to {output_path}")


if __name__ == "__main__":
    generate_notebook()
