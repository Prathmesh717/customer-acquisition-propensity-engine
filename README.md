# Customer Acquisition Propensity & Explainability Engine

An end-to-end Machine Learning, Predictive Analytics, and Model Governance Platform for customer acquisition propensity prediction, model explainability, drift monitoring, and business decision support.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?logo=scikitlearn" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Explainable%20AI-SHAP-purple" alt="SHAP">
  <img src="https://img.shields.io/badge/Monitoring-PSI-green" alt="PSI">
  <img src="https://img.shields.io/badge/Testing-Pytest-blue?logo=pytest" alt="Pytest">
</p>

---

## 1. Overview

The **Customer Acquisition Propensity & Explainability Engine** is an end-to-end machine learning application developed using Python and Streamlit. It enables organizations to predict customer acquisition propensity, understand model decisions, evaluate predictive performance, monitor data drift, and maintain model governance.

The platform integrates the machine learning lifecycle into a centralized application, connecting data quality assessment, feature selection, model development, explainability, customer scoring, monitoring, and business intelligence.

It is designed to help business stakeholders and data science teams make informed, transparent, and data-driven customer acquisition decisions.

### Key Capabilities

- **Data Quality Management:** Dataset profiling, validation, missing-value analysis, duplicate detection, and outlier treatment.
- **Feature Analysis:** Weight of Evidence (WOE) and Information Value (IV) analysis.
- **Predictive Modelling:** Automated training and comparison of multiple classification algorithms.
- **Model Explainability:** Global and customer-level explanations using SHAP.
- **Customer Scoring:** Propensity prediction and customer segmentation.
- **Hyperparameter Optimization:** Model optimization using GridSearchCV.
- **Model Monitoring:** Feature-level and prediction-level drift monitoring using PSI.
- **Model Governance:** Model registry, version tracking, performance metrics, and approval status.
- **Business Intelligence:** Interactive dashboards and actionable customer acquisition insights.

---

## 2. Business Requirements Document (BRD)

The platform addresses the following business requirements:

| ID | Business Requirement | Business Value |
|---|---|---|
| BR-01 | Predict customer propensity to purchase using historical behavioral and demographic data. | Helps sales teams prioritize high-value acquisition opportunities. |
| BR-02 | Explain model predictions using SHAP values. | Improves transparency, interpretability, and stakeholder trust. |
| BR-03 | Implement data quality gates to prevent malformed datasets from reaching the model. | Improves data reliability and prevents downstream errors. |
| BR-04 | Monitor data drift using PSI and evaluate model performance. | Helps identify distribution changes and potential model degradation. |
| BR-05 | Maintain a model registry with version history, evaluation metrics, and approval status. | Supports auditability, traceability, and model lifecycle management. |
| BR-06 | Provide a centralized governance dashboard for business stakeholders. | Enables visibility into model health, drift, and performance. |

---

## 3. System Architecture

The platform follows a modular machine learning workflow, with each stage contributing to the overall model development and governance lifecycle.

```text
                 CUSTOMER DATASET
                        |
                        v
              DATA QUALITY DASHBOARD
                        |
                        v
                  WOE & IV ANALYSIS
                        |
                        v
                  MODEL TRAINING
                        |
                        v
              MODEL PERFORMANCE EVALUATION
                        |
                        v
                MODEL EXPLAINABILITY
                        |
                        v
                  CUSTOMER SCORING
                        |
                        v
             HYPERPARAMETER OPTIMIZATION
                        |
                        v
             MODEL MONITORING & DRIFT
                        |
                        v
                MODEL GOVERNANCE
                        |
                        v
                BUSINESS INSIGHTS
```

### End-to-End Workflow

1. Upload historical customer data.
2. Validate dataset structure and assess data quality.
3. Perform WOE and IV analysis for feature selection.
4. Train multiple classification algorithms.
5. Compare models using predictive performance metrics.
6. Interpret model predictions using SHAP.
7. Score new customer datasets.
8. Optimize model hyperparameters.
9. Monitor feature and prediction drift using PSI.
10. Track model versions and governance information.
11. Generate business insights to support customer acquisition strategies.

---

## 4. Core Features

### 4.1 Data Quality Dashboard

Provides an interactive environment for assessing, validating, and preparing customer datasets before model development.

**Data Quality Capabilities**

- Missing-value analysis
- Duplicate detection and removal
- Dataset profiling
- Statistical summaries
- Correlation heatmaps
- Feature distribution analysis
- Outlier detection
- Numerical and categorical feature analysis

**Outlier Treatment Techniques**

| Technique | Description |
|---|---|
| IQR Capping | Identifies and caps extreme values using the interquartile range. |
| Percentile Capping | Caps values using selected percentile thresholds. |
| Log Transformation | Reduces skewness in suitable numerical variables. |
| Outlier Removal | Removes observations identified as outliers. |

**Outputs**

- Dataset quality report
- Missing-value summary
- Duplicate analysis
- Feature distribution visualizations
- Correlation analysis
- Outlier treatment results

### 4.2 Weight of Evidence (WOE) & Information Value (IV)

Implements WOE and IV analysis to assess the predictive strength of individual features.

**Key Capabilities**

- Weight of Evidence calculations
- Information Value calculations
- Feature predictive power assessment
- Interactive WOE visualizations
- Downloadable IV reports

**Information Value Interpretation**

| IV Range | Predictive Power |
|---|---|
| Below 0.02 | Not Useful |
| 0.02–0.10 | Weak |
| 0.10–0.30 | Medium |
| 0.30–0.50 | Strong |
| Above 0.50 | Suspiciously Powerful |

**Outputs**

- Feature-level IV report
- WOE tables
- Feature ranking
- Interactive visualizations
- Downloadable reports

### 4.3 Machine Learning Model Training

The platform supports automated training and comparison of multiple classification algorithms for customer propensity prediction.

**Supported Algorithms**

| Algorithm | Application |
|---|---|
| Logistic Regression | Interpretable baseline classification |
| Random Forest | Ensemble-based classification |
| Gradient Boosting | Sequential boosting for predictive performance |
| XGBoost | Optimized gradient boosting |
| LightGBM | Efficient gradient boosting |
| CatBoost | Gradient boosting with categorical feature support |

**Model Evaluation Metrics**

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Gini Coefficient
- Kolmogorov-Smirnov (KS) Statistic

**Performance Visualizations**

- ROC curves
- Precision-Recall curves
- Confusion matrices
- Lift charts
- Decile analysis
- Model comparison dashboards

**Best Model Selection**

The platform selects the best-performing model based on ROC-AUC, providing a consistent basis for comparing candidate classification algorithms.

### 4.4 Model Explainability Using SHAP

Integrates SHAP (SHapley Additive exPlanations) to interpret model predictions and understand the factors influencing customer propensity scores.

**Global Explainability**

- SHAP summary plots
- SHAP feature importance
- Global feature contribution analysis

**Local Explainability**

- Customer-level SHAP waterfall plots
- Top positive prediction drivers
- Top negative prediction drivers
- Individual prediction interpretation

**Business Value**

SHAP explanations help stakeholders understand why a customer receives a particular propensity score and which features contribute most strongly to the prediction.

### 4.5 Customer Scoring & Segmentation

Allows users to upload new customer datasets and generate propensity predictions using a trained model.

**Scoring Capabilities**

- Input schema validation
- Missing-column detection
- Extra-column identification
- Automated preprocessing
- Customer-level propensity scoring
- Customer segmentation

**Propensity Segmentation**

| Score Range | Segment |
|---|---|
| 0.00–0.30 | Low |
| 0.30–0.60 | Medium |
| 0.60–1.00 | High |

**Outputs**

- Individual customer propensity scores
- Customer segment classifications
- Top 100 high-propensity customers
- Downloadable prediction files

### 4.6 Hyperparameter Optimization

Provides model optimization using GridSearchCV to identify suitable hyperparameter configurations.

**Supported Models**

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

**Optimization Outputs**

- Best hyperparameter configuration
- Best cross-validation ROC-AUC
- Tuned model export
- Model registry tracking

### 4.7 Model Monitoring & Drift Detection

Implements Population Stability Index (PSI) to identify changes in feature distributions between reference training data and incoming production data.

**Population Stability Index (PSI)**

PSI measures changes in the distribution of a variable between two datasets.

**PSI Interpretation**

| PSI Value | Interpretation |
|---|---|
| Below 0.10 | No Significant Drift |
| 0.10–0.25 | Moderate Drift |
| Above 0.25 | Significant Drift |

**Monitoring Capabilities**

- Feature-level PSI calculations
- Prediction-score PSI
- Reference versus current distribution comparison
- Drift summaries
- Drift status indicators
- Governance dashboard integration

**Business Value**

Drift monitoring helps identify changes in customer characteristics and prediction distributions that may indicate a need for further investigation, model validation, or retraining.

### 4.8 Model Governance & Registry

Provides centralized model tracking to support model lifecycle management, traceability, and governance reviews.

**Model Registry Capabilities**

- Model version tracking
- Model training timestamps
- ROC-AUC, Gini, and KS metrics
- Model approval status
- Model metadata management
- Model history tracking

**Governance Controls**

- Dataset validation before model processing
- Model performance documentation
- Drift monitoring
- Prediction explainability
- Automated validation tests
- Model registry records

### 4.9 Business Insights & Decision Support

Provides executive-level performance indicators and customer acquisition insights through interactive dashboards.

**Key Performance Indicators**

- Best ROC-AUC
- Best Gini Coefficient
- Best KS Statistic
- Customer propensity distribution
- High-propensity customer segment
- Feature-level drift status

**Example Business Insight**

| Metric | Illustrative Result |
|---|---|
| Model ROC-AUC | 0.82 |
| Gini Coefficient | 0.64 |
| KS Statistic | 0.48 |
| Drift Status | Stable |
| High-Propensity Segment | Customers with high engagement activity |

**Illustrative Top Propensity Drivers**

- `cart_count`
- `session_duration`
- `pages_viewed`

**Example Business Recommendation**

Prioritize high-propensity customers for personalized acquisition campaigns and targeted offers to improve conversion opportunities.

*Note: The metrics and customer insights above are illustrative examples, not verified experimental results.*

---

## 5. Functional Testing & User Acceptance Testing (UAT)

The platform includes a set of validation scenarios designed to verify data quality controls, model outputs, drift calculations, and governance operations.

### UAT Test Scenarios

| Test ID | Test Scenario | Expected Result |
|---|---|---|
| TC-01 | Upload a dataset containing null `customer_id` values. | Data quality gate rejects the batch and records the validation error. |
| TC-02 | Upload a dataset containing duplicate rows. | Duplicate records are detected and handled before model training. |
| TC-03 | Calculate PSI for identical distributions. | PSI is approximately 0.0, indicating no distribution shift. |
| TC-04 | Calculate PSI for significantly shifted distributions. | PSI exceeds 0.25 and significant drift is flagged. |
| TC-05 | Validate model prediction probabilities. | All predicted probabilities fall between 0 and 1. |
| TC-06 | Verify the model registry after training. | Registry records model version, ROC-AUC, and training timestamp. |

### Test Documentation

The project includes the following testing documentation:

- `docs/UAT_Plan.md`
- `tests/test_data_quality.py`
- `tests/test_drift.py`
- `tests/test_model_output.py`

**Run Automated Tests**

```bash
pytest
```

*The test scenarios describe the intended validation behavior. Actual test execution and passing results should be verified in the project environment.*

---

## 6. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Application Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost, LightGBM, CatBoost |
| Explainable AI | SHAP |
| Statistical Analysis | WOE, IV, PSI |
| Data Visualization | Matplotlib, Seaborn, Altair |
| Model Persistence | Joblib |
| Database | SQLite |
| Hyperparameter Optimization | GridSearchCV |
| Testing | Pytest |

---

## 7. Project Structure

```text

customer-acquisition-propensity-engine/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── pages/
│   ├── 1_Data_Quality.py
│   ├── 2_WOE_IV_Analysis.py
│   ├── 3_Model_Training.py
│   ├── 4_Model_Explainability.py
│   ├── 5_Customer_Scoring.py
│   ├── 6_Model_Monitoring.py
│   └── 7_Business_Insights.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── validation.py
│   │   └── preprocessing.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── woe_iv.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   ├── tuning.py
│   │   └── scoring.py
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── shap_analysis.py
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── drift_detection.py
│   │
│   └── governance/
│       ├── __init__.py
│       └── model_registry.py
│
├── models/
│   └── .gitkeep
│
├── registry/
│   └── model_registry.db
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── reference/
│       └── .gitkeep
│
├── reports/
│   ├── model_evaluation/
│   │   └── .gitkeep
│   ├── explainability/
│   │   └── .gitkeep
│   └── monitoring/
│       └── .gitkeep
│
├── docs/
│   ├── BRD.md
│   ├── UAT_Plan.md
│   └── Governance_Checklist.md
│
├── tests/
│   ├── __init__.py
│   ├── test_data_quality.py
│   ├── test_woe_iv.py
│   ├── test_model_training.py
│   ├── test_model_output.py
│   ├── test_drift.py
│   └── test_model_registry.py
│
└── .streamlit/
    └── config.toml
```

---

## 8. Installation & Setup

### Prerequisites

- Python 3.10 or later
- pip
- Git

### Step 1: Clone the Repository

```bash
git clone https://Prathmesh717//customer-acquisition-propensity-engine.git

cd customer-acquisition-propensity-engine
```

Replace `yourusername` with your GitHub username.

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate the Environment

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### Step 5: Launch the Application

```bash
streamlit run app.py
```

The Streamlit application will be available at:

```text
http://localhost:8501
```

---

## 9. Dependencies

The primary project dependencies include:

```text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
shap
joblib
altair
pytest
```

**Note:** SQLite is included in Python's standard library through `sqlite3`; it does not normally need to be installed separately using pip.

---

## 10. Sample Business Workflow

The following workflow illustrates how a business user or data scientist can interact with the platform.

| Step | Activity | Outcome |
|---|---|---|
| 01 | Upload historical customer data. | Customer dataset becomes available for analysis. |
| 02 | Perform data quality assessment. | Dataset validation and quality insights. |
| 03 | Run WOE & IV analysis. | Feature predictive power assessment. |
| 04 | Train classification models. | Multiple candidate propensity models. |
| 05 | Compare model performance. | Selection of the best-performing model. |
| 06 | Review SHAP explanations. | Understanding of global and customer-level predictions. |
| 07 | Score new customer datasets. | Customer propensity scores and segments. |
| 08 | Perform hyperparameter tuning. | Optimized model configuration. |
| 09 | Monitor drift using PSI. | Identification of changing feature distributions. |
| 10 | Review model governance. | Model version and performance tracking. |
| 11 | Generate business insights. | Data-driven customer acquisition recommendations. |

---

## 11. Business Applications

The platform's capabilities can be adapted to several industries and customer acquisition use cases.

### Banking & Financial Services

- Term deposit subscription prediction
- Cross-sell propensity modelling
- Loan offer acceptance prediction
- Customer campaign prioritization

### Insurance

- Insurance policy purchase propensity
- Customer acquisition prioritization
- Product recommendation analysis
- Customer engagement analytics

### FinTech

- Lead prioritization
- Customer conversion prediction
- Financial product adoption analysis
- Customer segmentation

### Marketing & Retail

- Campaign targeting
- Purchase propensity prediction
- Customer segmentation
- Personalized marketing strategies

---

## 12. Future Enhancements

The platform can be extended with additional enterprise machine learning and deployment capabilities.

| Enhancement | Objective |
|---|---|
| MLflow Integration | Experiment tracking and model lifecycle management. |
| Model Versioning API | Programmatic access to registered models. |
| Real-Time Scoring API | Low-latency customer propensity predictions. |
| Automated Retraining Pipeline | Triggered or scheduled model retraining. |
| Feature Store Integration | Centralized feature management and reuse. |
| Docker Deployment | Containerized application deployment. |
| AWS / Azure Deployment | Cloud-based infrastructure and scalability. |
| Champion-Challenger Framework | Comparison of production and candidate models. |

---

## 13. Author

**Prathmesh Aghor**

Computer Engineering | Machine Learning | Predictive Analytics | Explainable AI

---

## License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Customer Acquisition Propensity & Explainability Engine</b><br>
  Turning customer data into explainable predictions and actionable business decisions.
</p>