"""
schemas.py — Pydantic models for API input/output validation
=============================================================

WHY PYDANTIC:
    Pydantic automatically validates that the data sent to our API
    is correct (right types, within valid ranges). This prevents
    the model from crashing on bad input.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict


class LoanApplication(BaseModel):
    """Input schema — represents a single loan application.
    
    Each field has validation constraints to ensure data quality.
    """
    person_age: int = Field(..., ge=18, le=100, description="Applicant age (18-100)")
    person_income: float = Field(..., gt=0, description="Annual income in USD")
    person_home_ownership: str = Field(
        ..., description="RENT, MORTGAGE, OWN, or OTHER"
    )
    person_emp_length: float = Field(..., ge=0, description="Employment length in years")
    loan_intent: str = Field(
        ..., description="EDUCATION, MEDICAL, VENTURE, PERSONAL, DEBTCONSOLIDATION, or HOMEIMPROVEMENT"
    )
    loan_grade: str = Field(..., description="Loan grade: A through G")
    loan_amnt: float = Field(..., gt=0, description="Loan amount in USD")
    loan_int_rate: float = Field(..., ge=0, le=40, description="Interest rate (%)")
    loan_percent_income: float = Field(..., ge=0, le=1, description="Loan as fraction of income")
    cb_person_default_on_file: str = Field(..., description="Historical default: Y or N")
    cb_person_cred_hist_length: int = Field(..., ge=0, description="Credit history length in years")

    class Config:
        json_schema_extra = {
            "example": {
                "person_age": 28,
                "person_income": 65000,
                "person_home_ownership": "RENT",
                "person_emp_length": 4,
                "loan_intent": "EDUCATION",
                "loan_grade": "B",
                "loan_amnt": 12000,
                "loan_int_rate": 11.5,
                "loan_percent_income": 0.18,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 6,
            }
        }


class RiskPrediction(BaseModel):
    """Output schema — the risk assessment result."""
    probability_of_default: float = Field(..., description="PD (0 to 1)")
    credit_score: int = Field(..., description="Scorecard points (like FICO)")
    risk_category: str = Field(..., description="Excellent/Good/Fair/Poor/Very Poor")
    decision: str = Field(..., description="APPROVE or DENY")
    top_risk_factors: Dict[str, float] = Field(
        ..., description="Top 5 SHAP feature contributions"
    )
