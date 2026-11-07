# User Acceptance Testing (UAT) Plan

## Customer Acquisition Propensity & Explainability Engine

**Document Type:** User Acceptance Testing Plan  
**Project Name:** Customer Acquisition Propensity & Explainability Engine  
**Version:** 1.0  
**Status:** Test Specification  
**Author:** Prathmesh Aghor  
**Last Updated:** November 2026

---

# 1. Introduction

## 1.1 Purpose

This document defines the User Acceptance Testing (UAT) strategy for the Customer Acquisition Propensity & Explainability Engine.

The purpose of UAT is to verify that the platform meets its documented business and functional requirements, produces reliable outputs, and provides the expected capabilities for customer propensity prediction, model explainability, monitoring, and governance.

Testing focuses on the complete machine learning workflow, from customer dataset upload and validation to model training, scoring, monitoring, and business decision support.

## 1.2 Testing Objectives

The primary objectives are:

- Verify dataset upload and validation functionality.
- Validate missing-value and duplicate detection.
- Verify WOE and IV calculations.
- Validate machine learning model training.
- Verify model evaluation metrics.
- Validate SHAP-based explainability.
- Verify customer propensity scoring.
- Test customer segmentation logic.
- Validate hyperparameter optimization.
- Verify PSI drift calculations.
- Validate model registry operations.
- Verify business dashboard outputs.
- Identify functional defects and unexpected behavior.

## 1.3 Scope of Testing

The following modules are included in UAT:

| Module ID | Module | Testing Scope |
|---|---|---|
| MOD-01 | Data Quality Dashboard | Dataset validation, missing values, duplicates, and outliers. |
| MOD-02 | WOE & IV Analysis | WOE calculations, IV calculations, and feature ranking. |
| MOD-03 | Model Training | Classification algorithms, training, and model comparison. |
| MOD-04 | Model Explainability | Global and local SHAP explanations. |
| MOD-05 | Customer Scoring | Prediction probabilities and customer segmentation. |
| MOD-06 | Hyperparameter Tuning | GridSearchCV and optimized model output. |
| MOD-07 | Model Monitoring | PSI calculations and drift classification. |
| MOD-08 | Model Governance | Model registry and version tracking. |
| MOD-09 | Business Insights | KPIs and business intelligence dashboards. |

## 1.4 Out of Scope

Unless separately implemented, the following are excluded:

- Production cloud deployment testing.
- Real-time streaming validation.
- External enterprise system integration.
- Automated production retraining.
- Production API load testing.
- Formal regulatory certification.

---

# 2. Testing Approach

The platform will be tested using a combination of functional testing, integration testing, negative testing, data validation, and user acceptance testing.

| Testing Type | Description |
|---|---|
| Functional Testing | Verifies individual application features. |
| Integration Testing | Verifies interaction between modules. |
| Data Validation Testing | Checks input schema, missing values, and invalid records. |
| Model Testing | Validates model outputs and evaluation metrics. |
| Explainability Testing | Verifies SHAP explanation generation. |
| Monitoring Testing | Validates PSI calculations and drift thresholds. |
| Regression Testing | Checks that changes do not break existing functionality. |
| User Acceptance Testing | Confirms the application meets intended business requirements. |

---

# 3. Test Environment

## 3.1 Environment Requirements

| Component | Technology |
|---|---|
| Operating System | macOS / Windows / Linux |
| Programming Language | Python 3.10+ |
| Application Framework | Streamlit |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost, LightGBM, CatBoost |
| Explainability | SHAP |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Altair |
| Database | SQLite |
| Automated Testing | Pytest |

## 3.2 Test Execution Commands

Start the application:

```bash
streamlit run app.py
```

Run automated tests:

```bash
pytest
```

Run a specific test module:

```bash
pytest tests/test_data_quality.py
pytest tests/test_drift.py
pytest tests/test_model_output.py
```

Generate a detailed test report:

```bash
pytest -v
```

---

# 4. Test Data Requirements

Testing should use a representative customer dataset containing demographic, behavioral, and transactional information.

## 4.1 Sample Dataset Fields

| Field | Data Type | Description |
|---|---|---|
| customer_id | String / Integer | Unique customer identifier. |
| age | Numeric | Customer age. |
| income | Numeric | Customer income. |
| cart_count | Numeric | Number of cart additions. |
| session_duration | Numeric | Customer session duration. |
| pages_viewed | Numeric | Number of pages viewed. |
| previous_purchases | Numeric | Historical purchases. |
| campaign_response | Categorical | Historical campaign response. |
| target | Binary | Purchase or subscription outcome. |

The actual schema must match the dataset and use case configured in the application.

## 4.2 Test Dataset Scenarios

The following test datasets should be prepared:

1. Valid customer dataset.
2. Dataset containing missing values.
3. Dataset containing duplicate records.
4. Dataset with invalid column names.
5. Dataset with unexpected categorical values.
6. Dataset with missing required scoring columns.
7. Dataset with identical reference and current distributions.
8. Dataset with significantly shifted distributions.
9. Dataset containing imbalanced target classes.

---

# 5. Functional Test Cases

## 5.1 Data Quality Testing

| Test ID | Test Scenario | Test Input | Expected Result |
|---|---|---|---|
| UAT-DQ-01 | Upload a valid dataset. | Valid customer CSV. | Dataset loads successfully. |
| UAT-DQ-02 | Detect missing values. | Dataset with null values. | Missing values are identified and summarized. |
| UAT-DQ-03 | Detect duplicate records. | Dataset containing duplicates. | Duplicate records are identified. |
| UAT-DQ-04 | Generate statistical summary. | Valid numerical dataset. | Statistical summary is displayed. |
| UAT-DQ-05 | Generate correlation heatmap. | Dataset with numerical features. | Correlation visualization is generated. |
| UAT-DQ-06 | Detect outliers. | Dataset containing extreme values. | Outlier observations are identified. |
| UAT-DQ-07 | Apply IQR capping. | Dataset with numerical outliers. | Values are capped according to IQR thresholds. |
| UAT-DQ-08 | Upload an unsupported or malformed file. | Invalid input file. | A meaningful validation error is displayed. |

## 5.2 WOE & IV Testing

| Test ID | Test Scenario | Test Input | Expected Result |
|---|---|---|---|
| UAT-WOE-01 | Calculate WOE. | Valid binary target dataset. | WOE values are calculated. |
| UAT-WOE-02 | Calculate IV. | Valid feature and target. | Feature-level IV is generated. |
| UAT-WOE-03 | Rank features. | Multiple eligible features. | Features are ranked by IV. |
| UAT-WOE-04 | Handle missing values. | Dataset containing null feature values. | Missing values are handled according to configured rules. |
| UAT-WOE-05 | Generate WOE visualization. | Valid WOE results. | WOE chart is displayed. |
| UAT-WOE-06 | Export IV report. | Completed IV analysis. | Report is generated successfully. |

## 5.3 Model Training Testing

| Test ID | Test Scenario | Test Input | Expected Result |
|---|---|---|---|
| UAT-ML-01 | Train Logistic Regression. | Valid training dataset. | Model trains successfully. |
| UAT-ML-02 | Train Random Forest. | Valid training dataset. | Model trains successfully. |
| UAT-ML-03 | Train Gradient Boosting. | Valid training dataset. | Model trains successfully. |
| UAT-ML-04 | Train XGBoost. | Valid training dataset. | Model trains successfully. |
| UAT-ML-05 | Train LightGBM. | Valid training dataset. | Model trains successfully. |
| UAT-ML-06 | Train CatBoost. | Valid training dataset. | Model trains successfully. |
| UAT-ML-07 | Calculate ROC-AUC. | Actual labels and predicted probabilities. | ROC-AUC is calculated correctly. |
| UAT-ML-08 | Calculate Gini. | Valid ROC-AUC. | Gini equals 2 × AUC − 1. |
| UAT-ML-09 | Calculate KS statistic. | Actual labels and prediction scores. | KS statistic is calculated correctly. |
| UAT-ML-10 | Generate ROC curve. | Valid model predictions. | ROC curve is displayed. |
| UAT-ML-11 | Generate confusion matrix. | Actual and predicted labels. | Confusion matrix is displayed. |
| UAT-ML-12 | Select best model. | Multiple trained models. | Model with the highest ROC-AUC is selected. |

## 5.4 Model Explainability Testing

| Test ID | Test Scenario | Test Input | Expected Result |
|---|---|---|---|
| UAT-XAI-01 | Generate SHAP summary plot. | Trained supported model. | Summary plot is generated. |
| UAT-XAI-02 | Generate feature importance. | Model and explanation data. | Global feature importance is displayed. |
| UAT-XAI-03 | Generate local explanation. | Individual customer record. | Customer-level SHAP explanation is displayed. |
| UAT-XAI-04 | Identify positive drivers. | Customer-level SHAP values. | Positive contribution features are identified. |
| UAT-XAI-05 | Identify negative drivers. | Customer-level SHAP values. | Negative contribution features are identified. |

## 5.5 Customer Scoring Testing

| Test ID | Test Scenario | Test Input | Expected Result |
|---|---|---|---|
| UAT-CS-01 | Score valid customers. | Valid customer scoring dataset. | Propensity scores are generated. |
| UAT-CS-02 | Validate probability range. | Model prediction output. | All probabilities fall between 0 and 1. |
| UAT-CS-03 | Assign Low segment. | Score below 0.30. | Customer is classified as Low. |
| UAT-CS-04 | Assign Medium segment. | Score from 0.30 to below 0.60. | Customer is classified as Medium. |
| UAT-CS-05 | Assign High segment. | Score from 0.60 to 1.00. | Customer is classified as High. |
| UAT-CS-06 | Handle missing columns. | Dataset missing required features. | Schema validation reports missing columns. |
| UAT-CS-07 | Export scoring results. | Successfully scored dataset. | Prediction file is generated. |
| UAT-CS-08 | Identify high-propensity customers. | Scored customer dataset. | Highest-scoring customers are displayed. |

## 5.6 Hyperparameter Tuning Testing

| Test ID | Test Scenario | Test Input | Expected Result |
|---|---|---|---|
| UAT-HT-01 | Execute GridSearchCV. | Valid model and parameter grid. | Hyperparameter search executes. |
| UAT-HT-02 | Identify best parameters. | Completed search results. | Best parameter configuration is displayed. |
| UAT-HT-03 | Evaluate cross-validation AUC. | Completed cross-validation. | Best cross-validation AUC is displayed. |
| UAT-HT-04 | Export tuned model. | Successfully optimized model. | Tuned model is saved. |

## 5.7 Model Monitoring Testing

| Test ID | Test Scenario | Test Input | Expected Result |
|---|---|---|---|
| UAT-MON-01 | Compare identical distributions. | Same reference and current data. | PSI is approximately 0.0. |
| UAT-MON-02 | Detect moderate drift. | Moderately shifted distributions. | PSI is between 0.10 and 0.25. |
| UAT-MON-03 | Detect significant drift. | Significantly shifted distributions. | PSI exceeds 0.25. |
| UAT-MON-04 | Calculate feature-level PSI. | Reference and current features. | PSI is calculated for eligible features. |
| UAT-MON-05 | Calculate score PSI. | Reference and current prediction scores. | Score distribution drift is reported. |
| UAT-MON-06 | Display drift dashboard. | Completed monitoring results. | Drift status and summary are displayed. |

## 5.8 Model Governance Testing

| Test ID | Test Scenario | Test Input | Expected Result |
|---|---|---|---|
| UAT-GOV-01 | Register a trained model. | Trained model metadata. | Registry record is created. |
| UAT-GOV-02 | Verify model version. | New model registration. | Model version is recorded. |
| UAT-GOV-03 | Verify performance metrics. | Model evaluation results. | AUC, Gini, and KS are stored. |
| UAT-GOV-04 | Verify timestamp. | Model registration event. | Training or registration timestamp is recorded. |
| UAT-GOV-05 | Review model history. | Multiple registry records. | Historical entries can be retrieved. |
| UAT-GOV-06 | Update approval status. | Registered model. | Approval status is updated according to configured workflow. |

## 5.9 Business Insights Testing

| Test ID | Test Scenario | Test Input | Expected Result |
|---|---|---|---|
| UAT-BI-01 | Display best AUC. | Model comparison results. | Best AUC is displayed. |
| UAT-BI-02 | Display best Gini. | Model evaluation results. | Gini is displayed correctly. |
| UAT-BI-03 | Display best KS. | Model evaluation results. | KS statistic is displayed correctly. |
| UAT-BI-04 | Display customer segments. | Scored customer dataset. | Segment distribution is displayed. |
| UAT-BI-05 | Display monitoring status. | PSI results. | Drift status is displayed. |
| UAT-BI-06 | Generate business recommendations. | Available model and segment insights. | Recommendations reflect the available analysis. |

---

# 6. Integration Testing

Integration testing verifies the interaction between application modules.

| Test ID | Integration Scenario | Expected Result |
|---|---|---|
| INT-01 | Data Quality → WOE & IV | Validated dataset is available for feature analysis. |
| INT-02 | WOE & IV → Model Training | Eligible selected features can be used for training. |
| INT-03 | Model Training → Explainability | Trained model is available for SHAP analysis. |
| INT-04 | Model Training → Customer Scoring | Saved model can score new customer records. |
| INT-05 | Customer Scoring → Business Insights | Propensity scores are reflected in business summaries. |
| INT-06 | Model Training → Model Registry | Model metadata and performance metrics are registered. |
| INT-07 | Reference Data → Monitoring | Reference distributions can be compared with current data. |
| INT-08 | Monitoring → Governance Dashboard | Drift summaries are displayed in governance views. |

---

# 7. Negative Testing

Negative testing verifies that the application handles invalid inputs and exceptional conditions correctly.

| Test ID | Scenario | Expected Result |
|---|---|---|
| NEG-01 | Upload an empty dataset. | Application displays a validation error. |
| NEG-02 | Upload an unsupported file format. | File is rejected with a meaningful message. |
| NEG-03 | Train without a target column. | Training is prevented. |
| NEG-04 | Use a dataset containing only one target class. | Model training is blocked or an appropriate error is reported. |
| NEG-05 | Score data with missing required features. | Missing columns are reported. |
| NEG-06 | Generate SHAP without a trained model. | Application displays an informative message. |
| NEG-07 | Run PSI with incompatible reference and current features. | Invalid comparisons are reported. |
| NEG-08 | Load a missing or corrupted model artifact. | Application handles the loading error gracefully. |

---

# 8. Entry and Exit Criteria

## 8.1 Entry Criteria

UAT can begin when:

- The application environment is configured.
- Required dependencies are installed.
- The application starts successfully.
- Test datasets are available.
- Core modules are accessible.
- Test cases have been documented.

## 8.2 Exit Criteria

UAT can be considered complete when:

- All planned test cases have been executed.
- Critical defects have been resolved.
- Expected module outputs have been validated.
- Integration workflows have been checked.
- Test results have been documented.
- Outstanding defects have been recorded and reviewed.

---

# 9. Defect Management

Defects should be categorized according to severity and business impact.

| Severity | Description | Example |
|---|---|---|
| Critical | Application or essential workflow is unusable. | Model training crashes for valid input. |
| High | Important functionality produces incorrect output. | Incorrect propensity probabilities. |
| Medium | A feature works partially or displays incorrect information. | Incorrect dashboard metric label. |
| Low | Minor usability or presentation issue. | Formatting issue in a report. |

Each defect should include:

- Defect ID.
- Description.
- Steps to reproduce.
- Expected result.
- Actual result.
- Severity.
- Status.
- Resolution details.

---

# 10. UAT Execution Summary

Use the following table to record actual test execution results.

| Test Category | Total Test Cases | Passed | Failed | Blocked | Not Run |
|---|---:|---:|---:|---:|---:|
| Data Quality | 8 | 0 | 0 | 0 | 8 |
| WOE & IV | 6 | 0 | 0 | 0 | 6 |
| Model Training | 12 | 0 | 0 | 0 | 12 |
| Explainability | 5 | 0 | 0 | 0 | 5 |
| Customer Scoring | 8 | 0 | 0 | 0 | 8 |
| Hyperparameter Tuning | 4 | 0 | 0 | 0 | 4 |
| Monitoring | 6 | 0 | 0 | 0 | 6 |
| Governance | 6 | 0 | 0 | 0 | 6 |
| Business Insights | 6 | 0 | 0 | 0 | 6 |
| **Total** | **61** | **0** | **0** | **0** | **61** |

*Initial execution template. Update the counts after running the tests.*

---

# 11. UAT Sign-Off

| Role | Name | Status |
|---|---|---|
| Project Author | Prathmesh Aghor | Pending |
| Business Reviewer | To Be Assigned | Pending |
| Technical Reviewer | To Be Assigned | Pending |
| Testing Reviewer | To Be Assigned | Pending |

UAT approval should be recorded only after the relevant acceptance criteria have been reviewed and actual test results documented.

---

# 12. Conclusion

This UAT plan establishes a structured approach to validating the Customer Acquisition Propensity & Explainability Engine.

It covers individual modules, end-to-end workflows, model outputs, explainability, data drift monitoring, governance functionality, and business decision-support capabilities.

Successful execution of these tests will provide evidence that the implemented platform behaves according to its documented requirements.

---

**End of Document**

**Customer Acquisition Propensity & Explainability Engine**  
*Turning customer data into explainable predictions and actionable business decisions.*