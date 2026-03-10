"""
streamlit_app.py — Interactive Credit Risk Scoring Dashboard
==============================================================

Run:
    streamlit run app/streamlit_app.py

INTERVIEW TALKING POINT:
    "I built a Streamlit dashboard so that a loan officer can input
    an applicant's details and instantly see the risk score, SHAP
    explanation, and approve/deny recommendation. This simulates
    what a real bank's loan origination UI would look like."
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from src.data_loader import load_config
from src.feature_engineering import (
    create_financial_ratios,
    create_risk_buckets,
    create_interaction_features,
    encode_categoricals,
)
from src.scorecard import probability_to_score, get_risk_category

# ---- Page config ----
st.set_page_config(
    page_title="Credit Risk Scorer",
    page_icon="💳",
    layout="wide",
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .score-value {
        font-size: 4rem;
        font-weight: 800;
        margin: 0;
    }
    .approve {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .deny {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_config():
    """Load model artifacts (cached so it only runs once)."""
    config = load_config()
    try:
        mdl = joblib.load(config["paths"]["model_file"])
        feat_names = joblib.load(config["paths"]["feature_names_file"])

        try:
            explainer = joblib.load(config["paths"]["shap_explainer_file"])
        except Exception:
            explainer = None

        return mdl, feat_names, explainer, config, True
    except Exception as e:
        st.error(f"Model not found. Run `python -m src.pipeline` first.\n\nError: {e}")
        return None, None, None, config, False


model, feature_names, shap_explainer, config, model_loaded = load_model_and_config()

# ---- Header ----
st.markdown('<p class="main-header">💳 Credit Risk Scoring System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Production-grade ML model with SHAP explainability</p>', unsafe_allow_html=True)

if not model_loaded:
    st.stop()

# ---- Sidebar: Applicant Input ----
st.sidebar.header("📋 Applicant Details")

person_age = st.sidebar.slider("Age", 18, 80, 28)
person_income = st.sidebar.number_input("Annual Income ($)", 10000, 500000, 65000, step=5000)
person_home_ownership = st.sidebar.selectbox(
    "Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"]
)
person_emp_length = st.sidebar.slider("Employment Length (years)", 0, 40, 4)
loan_intent = st.sidebar.selectbox(
    "Loan Purpose",
    ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"],
)
loan_grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.sidebar.number_input("Loan Amount ($)", 500, 100000, 12000, step=500)
loan_int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 25.0, 11.5, step=0.5)
loan_percent_income = st.sidebar.slider("Loan % of Income", 0.0, 0.8, 0.18, step=0.01)
cb_default = st.sidebar.selectbox("Previous Default on File?", ["N", "Y"])
cb_cred_hist = st.sidebar.slider("Credit History Length (years)", 1, 30, 6)

# ---- Score button ----
if st.sidebar.button("🔍 Score Applicant", type="primary", use_container_width=True):
    # Build input
    input_data = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_default,
        "cb_person_cred_hist_length": cb_cred_hist,
    }])

    # Feature engineering
    input_data = create_financial_ratios(input_data)
    input_data = create_risk_buckets(input_data)
    interaction_pairs = config["feature_engineering"]["interaction_pairs"]
    input_data = create_interaction_features(input_data, interaction_pairs)
    input_data = encode_categoricals(input_data)

    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[feature_names]
    input_data = input_data.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Predict
    probability = float(model.predict_proba(input_data)[:, 1][0])
    credit_score = probability_to_score(probability, config)
    risk_category = get_risk_category(credit_score)
    decision = "APPROVE" if credit_score >= 650 else "DENY"

    # ---- Display results ----
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="score-card">
            <p style="margin:0; font-size:1rem;">CREDIT SCORE</p>
            <p class="score-value">{credit_score}</p>
            <p style="margin:0; font-size:1.2rem;">{risk_category}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="score-card">
            <p style="margin:0; font-size:1rem;">DEFAULT PROBABILITY</p>
            <p class="score-value">{probability:.1%}</p>
            <p style="margin:0; font-size:1.2rem;">Probability of Default</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        css_class = "approve" if decision == "APPROVE" else "deny"
        emoji = "✅" if decision == "APPROVE" else "❌"
        st.markdown(f"""
        <div class="score-card">
            <p style="margin:0; font-size:1rem;">DECISION</p>
            <p class="score-value" style="font-size:3rem;">{emoji}</p>
            <p style="margin:0; font-size:1.5rem; font-weight:700;">{decision}</p>
        </div>
        """, unsafe_allow_html=True)

    # ---- SHAP Explanation ----
    st.markdown("---")
    st.subheader("🔍 Why This Score? (SHAP Explanation)")

    if shap_explainer is not None:
        try:
            shap_values = shap_explainer.shap_values(input_data)
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0]

            contributions = dict(zip(feature_names, sv))
            sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

            factors_df = pd.DataFrame(sorted_contrib, columns=["Feature", "SHAP Value"])
            factors_df["Direction"] = factors_df["SHAP Value"].apply(
                lambda x: "↑ Increases Risk" if x > 0 else "↓ Decreases Risk"
            )
            factors_df["Impact"] = factors_df["SHAP Value"].abs()

            # Bar chart
            import plotly.express as px
            fig = px.bar(
                factors_df,
                x="SHAP Value", y="Feature",
                orientation="h",
                color="Direction",
                color_discrete_map={
                    "↑ Increases Risk": "#ef4444",
                    "↓ Decreases Risk": "#22c55e"
                },
                title="Top 10 Factors Influencing This Decision",
            )
            fig.update_layout(
                yaxis=dict(autorange="reversed"),
                height=400,
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")
    else:
        st.info("SHAP explainer not loaded. Run the full pipeline to enable explanations.")

    # ---- Applicant Summary ----
    st.subheader("📋 Applicant Summary")
    summary_data = {
        "Age": person_age,
        "Income": f"${person_income:,.0f}",
        "Home": person_home_ownership,
        "Employment": f"{person_emp_length} years",
        "Loan Purpose": loan_intent,
        "Loan Grade": loan_grade,
        "Loan Amount": f"${loan_amnt:,.0f}",
        "Interest Rate": f"{loan_int_rate}%",
        "Loan/Income": f"{loan_percent_income:.0%}",
        "Prior Default": cb_default,
        "Credit History": f"{cb_cred_hist} years",
    }
    col_a, col_b = st.columns(2)
    items = list(summary_data.items())
    for i, (k, v) in enumerate(items):
        target_col = col_a if i % 2 == 0 else col_b
        target_col.metric(k, v)

else:
    # Show instructions when page first loads
    st.info("👈 Fill in the applicant details in the sidebar and click **Score Applicant**")
    
    # Show sample SHAP plot if it exists
    shap_plot_path = os.path.join(config["paths"]["model_dir"], "shap_global_importance.png")
    if os.path.exists(shap_plot_path):
        st.subheader("📊 Global Feature Importance (from training)")
        st.image(shap_plot_path)
    
    fairness_plot_path = os.path.join(config["paths"]["model_dir"], "fairness_audit.png")
    if os.path.exists(fairness_plot_path):
        st.subheader("⚖️ Fairness Audit Results")
        st.image(fairness_plot_path)
