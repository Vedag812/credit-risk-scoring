"""
Credit Risk Scoring System
===========================
End-to-end ML pipeline for predicting loan defaults.

Modules:
    data_loader          — Download and clean the credit risk dataset
    feature_engineering  — Create banking-relevant features
    model                — Train and evaluate LightGBM / XGBoost
    explainability       — SHAP-based model explanations
    fairness             — Bias audit with Fairlearn
    scorecard            — Convert ML probabilities to scorecard points
    pipeline             — One-command training orchestrator
"""
