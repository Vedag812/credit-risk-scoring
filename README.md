# Credit Risk Scoring System

A full end-to-end ML pipeline for predicting loan defaults using LightGBM, with SHAP explainability, bias auditing, and a REST API for real-time scoring.
 
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal?logo=fastapi)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-orange?logo=githubactions)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Why This Matters

Every bank needs to decide who gets a loan and who doesn't. Get it wrong, and you either lose money on bad loans or turn away good customers. Traditional rule-based systems can't keep up anymore, so banks are increasingly using **machine learning** and **artificial intelligence** for this.

But the catch is: you can't just throw a black-box model at it. Regulators (OCC, Fed) want to see *why* someone was denied. The Equal Credit Opportunity Act says you can't discriminate. So the model needs to be accurate, explainable, AND fair.

That's what this project does end-to-end.

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │         DATA PIPELINE               │
                        └───────────────┬─────────────────────┘
                                        │
               ┌────────────────────────┼────────────────────────┐
               ▼                        ▼                        ▼
     ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
     │  Data Ingestion │    │    Feature       │    │  Model Training │
     │  & Cleaning     │    │    Engineering   │    │  (LightGBM/XGB) │
     │  (pandas)       │    │  (WoE, ratios)   │    │                 │
     └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
          ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
          │    SHAP      │ │  Fairness    │ │  Scorecard   │
          │  Explainer   │ │  Audit       │ │  (PD→Points) │
          └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                 │                │                │
        ┌────────┴────────────────┴────────────────┴────────┐
        │              DEPLOYMENT LAYER                      │
        ├──────────────┬──────────────┬─────────────────────┤
        │  FastAPI     │  Streamlit   │  Docker Container   │
        │  REST API    │  Dashboard   │  (Cloud-ready)      │
        │  (/predict)  │  (UI)        │                     │
        └──────────────┴──────────────┴─────────────────────┘
```

---

## Getting Started

```bash
# 1. clone
git clone https://github.com/Vedag812/credit-risk-scoring.git
cd credit-risk-scoring

# 2. set up a virtual environment (good practice)
python -m venv venv
venv\Scripts\activate   # windows

# 3. install dependencies
pip install -r requirements.txt

# 4. train the model (runs everything end-to-end)
python -m src.pipeline

# 5. fire up the API
uvicorn api.main:app --reload --port 8000
# go to http://localhost:8000/docs for swagger

# 6. or use the dashboard
streamlit run app/streamlit_app.py

# 7. run tests (CI/CD runs these on every push)
python -m pytest tests/ -v
```

---

## ML Pipeline at a Glance

| Step | Module | What it does |
|------|--------|-------------|
| 1 | `data_loader.py` | **Data ingestion** - downloads, cleans nulls, caps outliers |
| 2 | `feature_engineering.py` | **Feature engineering** - financial ratios, WoE encoding, risk buckets |
| 3 | `model.py` | **Model training** - LightGBM + XGBoost, evaluates with Gini/KS |
| 4 | `explainability.py` | **AI explainability** - SHAP global and per-applicant analysis |
| 5 | `fairness.py` | **Responsible AI** - bias audit across age groups |
| 6 | `scorecard.py` | **Business logic** - converts probability to a FICO-style score |
| 7 | `pipeline.py` | **Orchestration** - ties everything together, one command |

---

## Results

| Metric | LightGBM | XGBoost |
|--------|----------|---------|
| AUC-ROC | ~0.92 | ~0.91 |
| Gini | ~0.84 | ~0.82 |
| KS Statistic | ~0.70 | ~0.68 |

These will vary a bit depending on the random split. Run `python -m src.pipeline` to get your exact numbers.

---

## Design Decisions and Technical Depth

**Why LightGBM over deep learning?**
For tabular data like loan applications, gradient boosting still beats neural nets in most benchmarks. Plus it trains in seconds and is much easier to explain to a regulator. That said, deep learning approaches (like TabNet) are worth exploring for more complex datasets.

**Why Weight-of-Evidence (WoE) encoding?**
It's what banks actually use in production. Every credit scorecard at major financial institutions uses WoE because regulators can audit the values directly and it captures the predictive relationship between each category and the target.

**Why bother with fairness auditing?**
ECOA (Equal Credit Opportunity Act) and the new EU AI Act both require this. If your model denies older applicants more often just because of their age, that's a legal problem. Responsible AI practices are becoming a baseline requirement.

**CI/CD and Application Resiliency:**
The project uses **GitHub Actions** for continuous integration - tests run automatically on every push. The **Docker** container ensures consistent behavior across environments. The FastAPI service includes a `/health` endpoint for monitoring and load balancer health checks.

**Cloud Readiness:**
The Dockerized API can be deployed to any cloud provider (AWS ECS, GCP Cloud Run, Azure Container Apps). The architecture follows microservice patterns - the ML model is served as an independent service that can scale horizontally.

**Data Pipeline Design:**
The modular pipeline approach (separate modules for ingestion, feature engineering, training) follows software engineering best practices. Each module has a single responsibility, is independently testable, and can be swapped out. This is similar to how production data warehousing pipelines work at scale.

---

## Docker Deployment

```bash
# build and run
docker build -t credit-risk-scorer .
docker run -p 8000:8000 credit-risk-scorer

# or with docker-compose for production
# (add monitoring, logging, etc.)
```

---

## Project Structure

```
credit-risk-scoring/
├── src/                          # core ML code (Python)
│   ├── data_loader.py            #   data ingestion and cleaning
│   ├── feature_engineering.py    #   feature creation pipeline
│   ├── model.py                  #   model training and evaluation
│   ├── explainability.py         #   SHAP analysis
│   ├── fairness.py               #   bias audit
│   ├── scorecard.py              #   score conversion
│   └── pipeline.py               #   orchestrator
├── api/                          # REST API (FastAPI)
│   ├── main.py                   #   endpoints (/predict, /health)
│   └── schemas.py                #   input validation (Pydantic)
├── app/
│   └── streamlit_app.py          # interactive dashboard
├── notebooks/
│   └── credit_risk_eda.ipynb     # Kaggle notebook (EDA + modeling)
├── tests/                        # automated testing (pytest)
│   ├── test_features.py
│   └── test_model.py
├── .github/workflows/ci.yml      # CI/CD pipeline (GitHub Actions)
├── config.yaml                   # centralized configuration
├── Dockerfile                    # container deployment
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.11 |
| **Machine Learning** | LightGBM, XGBoost, scikit-learn |
| **AI Explainability** | SHAP |
| **API Development** | FastAPI, Pydantic, Uvicorn |
| **Dashboard** | Streamlit, Plotly |
| **CI/CD** | GitHub Actions (automated testing on push) |
| **Containerization** | Docker |
| **Testing** | pytest, httpx |
| **Data Processing** | pandas, NumPy |
| **Configuration** | YAML |

---

## Dataset

Kaggle Credit Risk Dataset - about 32K loan applications with 12 features (income, employment length, loan amount, interest rate, etc.) and a binary default label.

[Link to dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

---

## What I'd Add With More Time

- **Cloud deployment** to AWS/GCP with auto-scaling
- **MLflow** for experiment tracking and model versioning
- **Data drift monitoring** with Evidently AI
- **Feature store** (Feast) for consistent feature serving
- **Airflow** for scheduled retraining
- **A/B testing** framework for champion-challenger in production
- **Big data** migration using Spark for larger datasets

---

## License

MIT
