# Business Requirements Document (BRD)

## Customer Acquisition Propensity & Explainability Engine

**Document Type:** Business Requirements Document  
**Project Name:** Customer Acquisition Propensity & Explainability Engine  
**Version:** 1.0  
**Status:** Project Specification  
**Author:** Prathmesh Aghor  
**Technology:** Python, Streamlit, Machine Learning, Explainable AI  
**Last Updated:** November 2026

---

# 1. Executive Summary

The Customer Acquisition Propensity & Explainability Engine is an end-to-end machine learning and analytics platform designed to predict customer acquisition propensity, explain model decisions, monitor data distribution changes, and support data-driven business decisions.

The platform integrates multiple stages of the machine learning lifecycle into a centralized application, including data quality assessment, feature analysis, predictive modelling, model evaluation, explainability, customer scoring, drift monitoring, and model governance.

The primary objective is to enable business stakeholders, data scientists, and analytics teams to identify customers with a higher likelihood of purchasing a product or subscribing to a service.

The platform also emphasizes model transparency, reliability, and governance through SHAP-based explainability, Population Stability Index (PSI) monitoring, validation controls, and model registry capabilities.

# 2. Business Background

Organizations across banking, insurance, financial services, fintech, and marketing collect large volumes of customer demographic, transactional, and behavioral data.

However, identifying customers who are most likely to purchase a product or respond to an acquisition campaign remains a significant challenge.

Traditional customer targeting approaches often rely on static segmentation, manual analysis, and historical campaign performance.

These approaches may result in:

- Inefficient customer acquisition campaigns.
- Poor prioritization of potential customers.
- Limited understanding of customer purchasing behavior.
- Difficulty interpreting machine learning predictions.
- Inadequate monitoring of changing customer data distributions.
- Limited traceability of model versions and performance.

The proposed platform addresses these challenges by combining predictive modelling, explainable AI, data quality validation, monitoring, and governance in one application.

# 3. Business Objectives

The primary business objectives are:

1. Predict customer acquisition propensity using historical customer data.
2. Identify high-propensity customers for targeted acquisition campaigns.
3. Improve model transparency through explainable machine learning.
4. Ensure data quality before model training and scoring.
5. Monitor changes in incoming customer data using PSI.
6. Maintain model version history and performance records.
7. Provide centralized dashboards for business and analytics stakeholders.
8. Support informed decisions through measurable model performance and customer insights.

# 4. Project Scope

## 4.1 In Scope

The project includes the following capabilities:

| Module | Scope |
|---|---|
| Data Quality Management | Data profiling, validation, missing values, duplicates, and outlier analysis. |
| WOE & IV Analysis | Feature assessment using Weight of Evidence and Information Value. |
| Model Training | Training multiple machine learning classification algorithms. |
| Model Evaluation | Comparing models using classification and statistical performance metrics. |
| Model Explainability | Global and local prediction interpretation using SHAP. |
| Customer Scoring | Generating propensity scores for new customer datasets. |
| Customer Segmentation | Categorizing customers into Low, Medium, and High propensity groups. |
| Hyperparameter Tuning | Optimizing model parameters using GridSearchCV. |
| Model Monitoring | Detecting feature and prediction distribution changes using PSI. |
| Model Governance | Tracking model versions, metrics, timestamps, and approval status. |
| Business Insights | Presenting model performance and customer acquisition insights. |

## 4.2 Out of Scope

The following capabilities are considered future enhancements unless separately implemented:

- Live integration with external banking or insurance systems.
- Real-time streaming data ingestion.
- Automated production model deployment.
- Fully automated model retraining.
- Cloud infrastructure provisioning.
- Enterprise identity and access management.
- Automated regulatory reporting.
- Full-featured champion-challenger deployment.
- Production-grade model serving APIs.

# 5. Stakeholders

| Stakeholder | Responsibility |
|---|---|
| Business Stakeholders | Define acquisition objectives and evaluate business outcomes. |
| Data Scientists | Develop, train, evaluate, and interpret machine learning models. |
| Data Analysts | Analyze customer data and generate business insights. |
| Risk and Governance Teams | Review model performance, validation, explainability, and monitoring information. |
| Application Users | Upload datasets, run analysis, score customers, and review dashboards. |
| Project Developer | Design, implement, test, and maintain the platform. |

# 6. Business Requirements

| Requirement ID | Business Requirement | Business Value | Priority |
|---|---|---|---|
| BR-01 | Predict customer acquisition propensity using historical behavioral and demographic data. | Improve customer targeting and acquisition prioritization. | High |
| BR-02 | Provide explainability for model predictions using SHAP. | Improve transparency and stakeholder confidence. | High |
| BR-03 | Implement data quality gates before model training and inference. | Prevent invalid data from affecting model outputs. | High |
| BR-04 | Monitor feature and prediction drift using PSI. | Identify distribution changes and potential model degradation. | High |
| BR-05 | Maintain a model registry with version history and evaluation metrics. | Improve traceability and model lifecycle management. | High |
| BR-06 | Provide a centralized governance and decision-support dashboard. | Improve visibility into model health and performance. | High |
| BR-07 | Compare multiple machine learning algorithms. | Support informed model selection. | High |
| BR-08 | Generate customer-level propensity scores and segments. | Enable targeted acquisition strategies. | High |
| BR-09 | Optimize model performance using hyperparameter tuning. | Improve predictive performance. | Medium |
| BR-10 | Generate downloadable reports and business insights. | Support analysis and stakeholder reporting. | Medium |

# 7. Functional Requirements

## FR-01: Data Quality Management

**Description:**  
The platform shall provide data quality assessment and validation capabilities for uploaded customer datasets.

**Functional Requirements:**

- Upload customer datasets.
- Identify missing values.
- Detect duplicate records.
- Generate statistical summaries.
- Analyze numerical and categorical variables.
- Generate correlation heatmaps.
- Visualize feature distributions.
- Detect outliers.
- Apply supported outlier treatment techniques.
- Validate dataset structure and required columns.

**Outlier Treatment Methods:**

- IQR Capping.
- Percentile Capping.
- Log Transformation.
- Outlier Removal.

**Expected Outputs:**

- Data quality summary.
- Missing-value report.
- Duplicate analysis.
- Feature profiling.
- Outlier analysis.
- Validation status.

**Acceptance Criteria:**

- The application successfully reads supported datasets.
- Missing values are identified correctly.
- Duplicate records are detected.
- Invalid dataset structures are reported.
- Data quality results are displayed to the user.

## FR-02: WOE & IV Feature Analysis

**Description:**  
The platform shall calculate Weight of Evidence and Information Value to evaluate the predictive strength of customer features.

**Functional Requirements:**

- Calculate WOE for eligible features.
- Calculate IV for eligible features.
- Generate feature-level IV reports.
- Rank features according to predictive strength.
- Display WOE visualizations.
- Allow report downloads.

**Information Value Interpretation:**

| IV Range | Interpretation |
|---|---|
| Below 0.02 | Not Useful |
| 0.02–0.10 | Weak |
| 0.10–0.30 | Medium |
| 0.30–0.50 | Strong |
| Above 0.50 | Suspiciously Powerful |

**Acceptance Criteria:**

- WOE and IV calculations produce valid results for supported inputs.
- Feature rankings are generated.
- Invalid or unsuitable features are handled appropriately.
- The IV report is available for review and download.

## FR-03: Machine Learning Model Training

**Description:**  
The platform shall train multiple classification algorithms to predict customer acquisition propensity.

**Supported Algorithms:**

1. Logistic Regression.
2. Random Forest.
3. Gradient Boosting.
4. XGBoost.
5. LightGBM.
6. CatBoost.

**Functional Requirements:**

- Prepare training and testing datasets.
- Perform required preprocessing.
- Handle numerical and categorical variables.
- Train supported classification algorithms.
- Generate prediction probabilities.
- Compare candidate models.
- Select the best-performing model based on ROC-AUC.
- Save the selected model for subsequent scoring.

**Evaluation Metrics:**

- Accuracy.
- Precision.
- Recall.
- F1 Score.
- ROC-AUC.
- Gini Coefficient.
- Kolmogorov-Smirnov (KS) Statistic.

**Visualizations:**

- ROC curves.
- Precision-Recall curves.
- Confusion matrices.
- Lift charts.
- Decile analysis.

**Acceptance Criteria:**

- Supported models can be trained on valid datasets.
- Evaluation metrics are calculated.
- Model comparisons are displayed.
- The best model is selected using the defined ROC-AUC criterion.
- The selected model can be persisted and loaded.

## FR-04: Model Explainability

**Description:**  
The platform shall provide global and local model explanations using SHAP.

**Global Explainability:**

- SHAP summary plots.
- SHAP feature importance.
- Global feature contribution analysis.

**Local Explainability:**

- Customer-level SHAP waterfall plots.
- Positive prediction drivers.
- Negative prediction drivers.
- Individual customer prediction explanations.

**Acceptance Criteria:**

- SHAP explanations are generated for supported trained models.
- Global feature contributions are visualized.
- Individual customer predictions can be interpreted.
- Explanation outputs are consistent with the selected model and prediction.

## FR-05: Customer Propensity Scoring

**Description:**  
The platform shall score new customer records using the selected trained model.

**Functional Requirements:**

- Accept new customer datasets.
- Validate input schema.
- Detect missing required columns.
- Identify unexpected additional columns.
- Apply the trained preprocessing pipeline.
- Generate propensity probabilities.
- Assign customer segments.
- Display high-propensity customer records.
- Allow prediction results to be downloaded.

**Customer Segmentation:**

| Propensity Score | Segment |
|---|---|
| 0.00–0.30 | Low |
| 0.30–0.60 | Medium |
| 0.60–1.00 | High |

**Acceptance Criteria:**

- Valid input datasets are scored successfully.
- Predicted probabilities fall between 0 and 1.
- Customer segments follow the configured thresholds.
- Invalid input schemas generate meaningful validation messages.
- Results can be exported.

## FR-06: Hyperparameter Optimization

**Description:**  
The platform shall support model optimization through GridSearchCV.

**Supported Models:**

- Logistic Regression.
- Random Forest.
- Gradient Boosting.
- XGBoost.
- LightGBM.
- CatBoost.

**Functional Requirements:**

- Configure model-specific parameter grids.
- Execute cross-validation.
- Evaluate candidate configurations.
- Identify the best parameter combination.
- Display cross-validation ROC-AUC.
- Export tuned models.
- Track tuned model information in the registry.

**Acceptance Criteria:**

- Supported parameter searches execute successfully.
- Best parameters are displayed.
- Cross-validation results are recorded.
- Tuned models can be persisted.

## FR-07: Model Monitoring and Drift Detection

**Description:**  
The platform shall compare reference training data with incoming datasets to identify distribution changes using Population Stability Index.

**PSI Calculation:**

PSI is calculated using the following formula:

PSI = Σ (Actual% − Expected%) × ln(Actual% / Expected%)

Where:

- Expected% represents the reference distribution.
- Actual% represents the current dataset distribution.

**PSI Interpretation:**

| PSI Value | Drift Status |
|---|---|
| Below 0.10 | No Significant Drift |
| 0.10–0.25 | Moderate Drift |
| Above 0.25 | Significant Drift |

**Functional Requirements:**

- Compare reference and current feature distributions.
- Calculate feature-level PSI.
- Calculate prediction-score PSI.
- Display drift summaries.
- Identify features with significant distribution changes.
- Present drift information in the governance dashboard.

**Acceptance Criteria:**

- Identical distributions produce PSI approximately equal to zero.
- Shifted distributions produce corresponding PSI changes.
- Drift thresholds are applied consistently.
- Feature-level results are displayed.
- Invalid comparisons are handled appropriately.

## FR-08: Model Governance and Registry

**Description:**  
The platform shall maintain model information to support traceability and governance reviews.

**Registry Information:**

- Model identifier.
- Model version.
- Algorithm name.
- Training timestamp.
- ROC-AUC.
- Gini coefficient.
- KS statistic.
- Approval status.
- Model file reference.

**Functional Requirements:**

- Register trained models.
- Maintain model version history.
- Store evaluation metrics.
- Track approval status.
- Retrieve registered model information.
- Support model comparison and review.

**Acceptance Criteria:**

- Successful model training generates a registry record.
- Registry entries contain the required metadata.
- Model versions can be distinguished.
- Historical model records can be reviewed.

## FR-09: Business Insights and Decision Support

**Description:**  
The platform shall provide business stakeholders with centralized dashboards for reviewing model performance and customer acquisition insights.

**Key Performance Indicators:**

- Best ROC-AUC.
- Best Gini coefficient.
- Best KS statistic.
- Customer propensity distribution.
- High-propensity customer count.
- Feature-level drift status.

**Functional Requirements:**

- Display model performance metrics.
- Visualize customer propensity segments.
- Identify important prediction drivers.
- Summarize drift monitoring results.
- Present actionable customer acquisition recommendations.

**Acceptance Criteria:**

- Business KPIs are displayed correctly.
- Dashboard metrics reflect the selected model and available dataset.
- Customer segment summaries are generated.
- Insights are understandable to non-technical stakeholders.

# 8. Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-01 | Usability | The application should provide an intuitive Streamlit interface. |
| NFR-02 | Performance | Data processing and model training should complete within reasonable time for supported datasets and hardware. |
| NFR-03 | Reliability | Invalid inputs should be handled without unexpected application failures. |
| NFR-04 | Maintainability | The application should use modular Python components. |
| NFR-05 | Scalability | The architecture should allow future integration of APIs and cloud infrastructure. |
| NFR-06 | Security | Uploaded datasets and model artifacts should be handled securely. |
| NFR-07 | Explainability | Supported model predictions should have interpretable explanations. |
| NFR-08 | Auditability | Model versions and evaluation metadata should be traceable. |
| NFR-09 | Compatibility | The application should support the documented Python environment and dependencies. |
| NFR-10 | Reproducibility | Model experiments should use documented preprocessing and evaluation procedures. |

# 9. Data Requirements

## 9.1 Input Data

The platform expects structured customer data containing relevant demographic, behavioral, transactional, or engagement information.

Potential input features include:

| Feature | Description |
|---|---|
| customer_id | Unique customer identifier. |
| age | Customer age. |
| income | Customer income or income category. |
| cart_count | Number of cart additions. |
| session_duration | Duration of customer sessions. |
| pages_viewed | Number of pages viewed. |
| previous_purchases | Historical purchase activity. |
| campaign_response | Historical campaign response. |
| target | Historical purchase or subscription outcome. |

The exact dataset schema depends on the selected business use case.

## 9.2 Target Variable

The target variable represents whether a customer purchased or subscribed to a product or service.

Example:

| Target Value | Meaning |
|---|---|
| 1 / Yes | Customer purchased or subscribed. |
| 0 / No | Customer did not purchase or subscribe. |

The application must support a clearly defined positive class when calculating propensity probabilities and evaluation metrics.

## 9.3 Data Validation

The platform should validate:

- Required columns.
- Target variable availability during training.
- Unique customer identifiers where required.
- Missing values.
- Duplicate records.
- Numerical and categorical data types.
- Invalid or unexpected values.
- Compatibility between training and scoring schemas.

# 10. Key Performance Indicators (KPIs)

The following indicators will be used to evaluate platform functionality and model performance.

| KPI | Description |
|---|---|
| ROC-AUC | Measures the model's ability to distinguish between positive and negative classes. |
| Gini | Measures discriminatory power, calculated as 2 × AUC − 1. |
| KS Statistic | Measures the maximum separation between cumulative positive and negative distributions. |
| Precision | Measures the proportion of positive predictions that are correct. |
| Recall | Measures the proportion of actual positive cases identified. |
| F1 Score | Harmonic mean of precision and recall. |
| PSI | Measures distribution changes between reference and current datasets. |
| High-Propensity Rate | Proportion of customers classified into the high-propensity segment. |
| Data Quality Pass Rate | Proportion of evaluated datasets passing defined validation rules. |

Model acceptance thresholds should be configured according to the business use case and validation requirements rather than assumed to be universal.

# 11. Functional Testing and User Acceptance Testing

The platform should be validated using functional and UAT scenarios.

| Test ID | Test Scenario | Expected Result |
|---|---|---|
| TC-01 | Upload a dataset with null customer_id values. | The configured validation rule rejects the invalid batch and records the error. |
| TC-02 | Upload a dataset containing duplicate records. | Duplicate records are detected and handled according to the configured rule. |
| TC-03 | Calculate PSI for identical distributions. | PSI is approximately 0.0. |
| TC-04 | Calculate PSI for significantly shifted distributions. | PSI exceeds 0.25 and significant drift is flagged. |
| TC-05 | Generate customer propensity scores. | All prediction probabilities are between 0 and 1. |
| TC-06 | Train and register a model. | The registry records model version, metrics, and timestamp. |
| TC-07 | Upload a scoring dataset with missing required columns. | Schema validation reports the missing columns. |
| TC-08 | Generate SHAP explanations. | Global or local explanation outputs are generated for supported models. |
| TC-09 | Run hyperparameter tuning. | Best parameters and cross-validation metrics are returned. |
| TC-10 | Review the business insights dashboard. | Dashboard KPIs reflect the available model and customer data. |

## 11.1 Automated Test Documentation

The project testing documentation is intended to include:

- `docs/UAT_Plan.md`
- `tests/test_data_quality.py`
- `tests/test_drift.py`
- `tests/test_model_output.py`

Automated tests can be executed using:

```bash
pytest
```

Passing results should be documented only after the tests have actually been executed.

# 12. Governance and Control Requirements

The platform incorporates governance-oriented capabilities to improve model transparency, traceability, and reliability.

## 12.1 Model Transparency

- SHAP-based global feature importance.
- Customer-level prediction explanations.
- Model performance reporting.

## 12.2 Data Quality Controls

- Schema validation.
- Missing-value checks.
- Duplicate detection.
- Input data validation.

## 12.3 Model Monitoring

- Feature-level PSI.
- Prediction-score PSI.
- Drift threshold evaluation.
- Monitoring summaries.

## 12.4 Model Lifecycle Management

- Model registry.
- Version history.
- Training metadata.
- Performance metrics.
- Approval status.

## 12.5 Governance Limitations

The presence of explainability, registry, and drift monitoring capabilities supports governance activities but does not independently establish regulatory compliance or complete enterprise model risk management.

Formal governance requires appropriate organizational policies, independent validation, documented approvals, access controls, and operational procedures.

# 13. Business Rules

| Rule ID | Business Rule |
|---|---|
| RULE-01 | A valid target variable must be available when training a supervised classification model. |
| RULE-02 | The positive target class must be explicitly identified for propensity scoring. |
| RULE-03 | Invalid datasets must not proceed through configured data quality gates. |
| RULE-04 | The best candidate model is selected using ROC-AUC. |
| RULE-05 | Customer propensity scores must be between 0 and 1. |
| RULE-06 | Customer segments must follow the configured propensity score thresholds. |
| RULE-07 | PSI values must be interpreted using the defined drift thresholds. |
| RULE-08 | Model registry entries must retain model identification and evaluation metadata. |
| RULE-09 | SHAP explanations must correspond to the model used to generate the prediction. |
| RULE-10 | Model performance and governance information must be presented using the available recorded data. |

# 14. Risks and Mitigation

| Risk ID | Risk | Potential Impact | Mitigation |
|---|---|---|---|
| R-01 | Poor-quality input data | Unreliable predictions. | Implement data quality validation and preprocessing. |
| R-02 | Class imbalance | Biased model performance. | Evaluate class distribution and use appropriate modelling techniques. |
| R-03 | Data leakage | Misleading evaluation results. | Separate training and testing data and validate preprocessing procedures. |
| R-04 | Model overfitting | Poor generalization to unseen customers. | Use cross-validation and appropriate model evaluation. |
| R-05 | Data drift | Changes in prediction reliability. | Monitor feature distributions and investigate significant PSI changes. |
| R-06 | Model explainability limitations | Incorrect interpretation of model behavior. | Validate SHAP outputs and communicate their limitations. |
| R-07 | Model registry inconsistencies | Poor model traceability. | Validate registry records and model metadata. |
| R-08 | Dependency compatibility issues | Application failures. | Maintain documented dependency versions and test the environment. |
| R-09 | Inappropriate customer targeting | Poor business outcomes. | Review model performance, segment thresholds, and business policies before campaign use. |

# 15. Assumptions and Dependencies

## Assumptions

- Historical customer data is available for model training.
- The target variable represents a clearly defined acquisition outcome.
- The dataset contains relevant customer features.
- Users can provide new customer datasets for scoring.
- Model evaluation is performed using appropriate training and testing procedures.
- PSI monitoring uses comparable reference and current distributions.

## Dependencies

- Python runtime.
- Streamlit application framework.
- Pandas and NumPy.
- Scikit-learn.
- XGBoost, LightGBM, and CatBoost.
- SHAP.
- Matplotlib, Seaborn, and Altair.
- Joblib.
- SQLite.
- Pytest.

# 16. Expected Business Outcomes

The expected outcomes of the platform include:

1. Improved prioritization of potential customers.
2. More informed customer acquisition decisions.
3. Greater transparency into machine learning predictions.
4. Better visibility into data quality issues.
5. Early identification of changing customer data distributions.
6. Improved model traceability and documentation.
7. Centralized access to model performance and business insights.
8. A reusable foundation for future enterprise analytics integration.

These are intended business outcomes; their actual impact must be established through empirical evaluation and business validation.

# 17. Future Enhancements

The following capabilities may be considered for future development:

| Enhancement | Description |
|---|---|
| MLflow Integration | Experiment tracking and model lifecycle management. |
| Model Versioning API | API-based model registration and retrieval. |
| Real-Time Scoring API | Real-time customer propensity predictions. |
| Automated Retraining | Scheduled or drift-triggered retraining workflows. |
| Feature Store | Centralized feature management. |
| Docker Deployment | Containerized application delivery. |
| AWS / Azure Deployment | Cloud infrastructure integration. |
| Champion-Challenger Framework | Comparison of candidate and incumbent models. |
| Role-Based Access Control | Controlled access to governance functionality. |
| Audit Logging | Detailed tracking of model and user activities. |

# 18. Project Deliverables

| Deliverable | Description |
|---|---|
| Streamlit Application | Centralized interface for analytics and model workflows. |
| Data Quality Module | Dataset profiling and validation. |
| WOE & IV Module | Feature predictive power analysis. |
| Model Training Module | Classification model development and comparison. |
| Explainability Module | SHAP-based prediction interpretation. |
| Customer Scoring Module | Customer propensity prediction and segmentation. |
| Monitoring Module | PSI-based distribution drift analysis. |
| Governance Module | Model registry and metadata tracking. |
| Business Insights Module | Stakeholder-oriented analytics dashboards. |
| Documentation | BRD, UAT plan, and governance checklist. |
| Automated Tests | Validation of data quality, drift, and model outputs. |

# 19. Approval and Sign-Off

This document defines the proposed business requirements and functional scope for the Customer Acquisition Propensity & Explainability Engine.

| Role | Name | Approval Status |
|---|---|---|
| Project Author | Prathmesh Aghor | Pending |
| Business Stakeholder | To Be Assigned | Pending |
| Technical Reviewer | To Be Assigned | Pending |
| Governance Reviewer | To Be Assigned | Pending |

---

**End of Document**

**Customer Acquisition Propensity & Explainability Engine**  
*Turning customer data into explainable predictions and actionable business decisions.*