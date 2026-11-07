# Model Governance Checklist

## Customer Acquisition Propensity & Explainability Engine

**Document Type:** Model Governance and Validation Checklist  
**Project Name:** Customer Acquisition Propensity & Explainability Engine  
**Version:** 1.0  
**Status:** Governance Control Specification  
**Author:** Prathmesh Aghor  
**Last Updated:** November 2026

---

# 1. Purpose

This document defines the governance controls and validation checkpoints for the Customer Acquisition Propensity & Explainability Engine.

The purpose is to establish a structured approach to managing machine learning models throughout their lifecycle, including:

- Data quality and validation.
- Feature assessment.
- Model development.
- Model performance evaluation.
- Explainability and transparency.
- Customer propensity scoring.
- Data drift monitoring.
- Model registry and version tracking.
- Model review and approval.
- Ongoing monitoring and documentation.

The checklist is intended to support model traceability, transparency, reliability, and responsible use.

It is a project-level governance framework and should not be interpreted as certification of regulatory compliance.

# 2. Governance Objectives

The governance framework aims to:

1. Ensure input data is validated before model usage.
2. Maintain traceability of training datasets and model versions.
3. Document model development and evaluation procedures.
4. Provide explanations for model predictions.
5. Monitor changes in data distributions.
6. Maintain records of model performance.
7. Support model review and approval workflows.
8. Improve reproducibility and audit readiness.
9. Identify potential model risks and limitations.
10. Provide stakeholders with visibility into model health.

# 3. Model Lifecycle Governance

The platform follows a structured model lifecycle.

```text
        DATA COLLECTION
              |
              v
        DATA VALIDATION
              |
              v
       FEATURE ASSESSMENT
              |
              v
         MODEL TRAINING
              |
              v
      MODEL PERFORMANCE REVIEW
              |
              v
       MODEL EXPLAINABILITY
              |
              v
        MODEL REGISTRATION
              |
              v
        CUSTOMER SCORING
              |
              v
       MODEL MONITORING
              |
              v
       PERIODIC MODEL REVIEW
              |
              v
     RETRAINING / RETIREMENT
```

# 4. Governance Roles and Responsibilities

| Role | Responsibilities |
|---|---|
| Model Developer | Develops, trains, and documents machine learning models. |
| Data Analyst | Reviews input data, feature distributions, and business insights. |
| Model Validator | Reviews model methodology, evaluation, and validation evidence. |
| Business Stakeholder | Evaluates business relevance and acquisition outcomes. |
| Governance Reviewer | Reviews model documentation, traceability, and control evidence. |
| Application Administrator | Manages application configuration, model artifacts, and registry records. |

For an individual portfolio project, these roles may be performed by the project author. They represent intended responsibilities rather than a claim that a separate governance team exists.

# 5. Data Governance Checklist

## 5.1 Data Collection and Documentation

| ID | Governance Check | Expected Evidence | Status |
|---|---|---|---|
| DG-01 | Data source is identified. | Dataset source documentation. | Not Verified |
| DG-02 | Dataset purpose is documented. | Data description. | Not Verified |
| DG-03 | Dataset fields are documented. | Data dictionary. | Not Verified |
| DG-04 | Target variable is clearly defined. | Target definition. | Not Verified |
| DG-05 | Data usage permissions are reviewed. | Source terms or permission records. | Not Verified |
| DG-06 | Sensitive information is identified. | Data inventory and handling rules. | Not Verified |

## 5.2 Data Quality Controls

| ID | Governance Check | Expected Evidence | Status |
|---|---|---|---|
| DQ-01 | Missing-value analysis is performed. | Missing-value report. | Not Verified |
| DQ-02 | Duplicate records are identified. | Duplicate analysis. | Not Verified |
| DQ-03 | Required columns are validated. | Schema validation output. | Not Verified |
| DQ-04 | Data types are checked. | Data profiling report. | Not Verified |
| DQ-05 | Outliers are assessed. | Outlier analysis. | Not Verified |
| DQ-06 | Data quality thresholds are documented. | Validation configuration. | Not Verified |
| DQ-07 | Invalid datasets are prevented from proceeding. | Data quality gate test results. | Not Verified |
| DQ-08 | Data transformations are documented. | Preprocessing documentation. | Not Verified |

# 6. Feature Governance

Feature governance ensures that model inputs are relevant, documented, and appropriately processed.

| ID | Governance Check | Expected Evidence | Status |
|---|---|---|---|
| FG-01 | Model features are documented. | Feature dictionary. | Not Verified |
| FG-02 | Feature selection methodology is recorded. | WOE and IV report. | Not Verified |
| FG-03 | High-IV features are investigated. | Feature analysis. | Not Verified |
| FG-04 | Potential data leakage is assessed. | Leakage assessment. | Not Verified |
| FG-05 | Feature transformations are recorded. | Preprocessing configuration. | Not Verified |
| FG-06 | Training and scoring schemas are consistent. | Schema comparison results. | Not Verified |
| FG-07 | Sensitive or restricted features are reviewed. | Feature governance review. | Not Verified |

# 7. Model Development Governance

## 7.1 Model Documentation

| ID | Governance Check | Expected Evidence | Status |
|---|---|---|---|
| MD-01 | Model objective is documented. | Model description. | Not Verified |
| MD-02 | Target variable is documented. | Target definition. | Not Verified |
| MD-03 | Training dataset is identified. | Dataset reference. | Not Verified |
| MD-04 | Training and testing methodology is documented. | Model training configuration. | Not Verified |
| MD-05 | Algorithm selection is documented. | Model comparison report. | Not Verified |
| MD-06 | Preprocessing pipeline is recorded. | Pipeline configuration. | Not Verified |
| MD-07 | Random state and reproducibility settings are documented. | Training configuration. | Not Verified |
| MD-08 | Hyperparameter configuration is recorded. | Model parameters. | Not Verified |

## 7.2 Supported Algorithms

The platform supports the following candidate algorithms:

- Logistic Regression.
- Random Forest.
- Gradient Boosting.
- XGBoost.
- LightGBM.
- CatBoost.

The selected model and its configuration should be recorded in the model registry.

# 8. Model Performance Governance

Model performance must be evaluated using appropriate metrics and documented validation procedures.

| ID | Governance Check | Expected Evidence | Status |
|---|---|---|---|
| MP-01 | ROC-AUC is calculated. | Model evaluation report. | Not Verified |
| MP-02 | Gini coefficient is calculated. | Gini metric. | Not Verified |
| MP-03 | KS statistic is calculated. | KS metric. | Not Verified |
| MP-04 | Precision is evaluated. | Classification report. | Not Verified |
| MP-05 | Recall is evaluated. | Classification report. | Not Verified |
| MP-06 | F1 Score is evaluated. | Classification report. | Not Verified |
| MP-07 | Confusion matrix is reviewed. | Confusion matrix visualization. | Not Verified |
| MP-08 | Model generalization is assessed. | Cross-validation or holdout evaluation. | Not Verified |
| MP-09 | Class imbalance is assessed. | Target distribution report. | Not Verified |
| MP-10 | Model selection rationale is documented. | Model comparison results. | Not Verified |

## 8.1 Performance Metric Definitions

| Metric | Purpose |
|---|---|
| ROC-AUC | Evaluates the model's ability to distinguish between classes. |
| Gini | Measures discriminatory power; Gini = 2 × AUC − 1. |
| KS Statistic | Measures maximum separation between cumulative class distributions. |
| Precision | Evaluates correctness of positive predictions. |
| Recall | Evaluates identification of actual positive cases. |
| F1 Score | Balances precision and recall. |

Performance acceptance thresholds should be defined based on the use case, baseline model, validation data, and business objectives.

# 9. Model Explainability Governance

The platform uses SHAP to provide global and local model explanations.

## 9.1 Global Explainability

| ID | Governance Check | Expected Evidence | Status |
|---|---|---|---|
| EX-01 | Global feature importance is generated. | SHAP feature importance plot. | Not Verified |
| EX-02 | SHAP summary visualization is available. | SHAP summary plot. | Not Verified |
| EX-03 | Important prediction drivers are documented. | Feature contribution report. | Not Verified |
| EX-04 | Explanation outputs are associated with the correct model version. | Model and explanation metadata. | Not Verified |

## 9.2 Local Explainability

| ID | Governance Check | Expected Evidence | Status |
|---|---|---|---|
| EX-05 | Customer-level explanations are available. | SHAP waterfall plot. | Not Verified |
| EX-06 | Positive contribution features are identified. | Local explanation. | Not Verified |
| EX-07 | Negative contribution features are identified. | Local explanation. | Not Verified |
| EX-08 | Explanations are reviewed for consistency. | Validation record. | Not Verified |

## 9.3 Explainability Limitations

- SHAP explains model behavior; it does not establish causation.
- Feature contribution does not prove that changing a feature will change customer behavior.
- Explanations depend on the selected model, input data, and SHAP configuration.
- Explanation outputs should be interpreted alongside business context.

# 10. Customer Scoring Governance

| ID | Governance Check | Expected Evidence | Status |
|---|---|---|---|
| CS-01 | Scoring dataset schema is validated. | Schema validation report. | Not Verified |
| CS-02 | Correct registered model is used. | Model version reference. | Not Verified |
| CS-03 | Prediction probabilities are within 0–1. | Prediction validation results. | Not Verified |
| CS-04 | Customer segments follow configured thresholds. | Segment validation. | Not Verified |
| CS-05 | Scoring results are reproducible. | Model and preprocessing references. | Not Verified |
| CS-06 | Scoring output is documented. | Prediction export. | Not Verified |
| CS-07 | Customer identifiers are handled appropriately. | Data handling controls. | Not Verified |

## 10.1 Customer Propensity Segments

| Score Range | Segment |
|---|---|
| 0.00–0.30 | Low |
| 0.30–0.60 | Medium |
| 0.60–1.00 | High |

These thresholds are configurable business rules and should be validated against the intended acquisition strategy.

# 11. Model Monitoring and Drift Governance

The platform uses Population Stability Index to compare reference and current data distributions.

## 11.1 PSI Monitoring Checklist

| ID | Governance Check | Expected Evidence | Status |
|---|---|---|---|
| MON-01 | Reference dataset is identified. | Reference data metadata. | Not Verified |
| MON-02 | Current monitoring dataset is identified. | Current data metadata. | Not Verified |
| MON-03 | Feature-level PSI is calculated. | Feature PSI report. | Not Verified |
| MON-04 | Prediction-score PSI is calculated. | Score PSI report. | Not Verified |
| MON-05 | PSI thresholds are documented. | Monitoring configuration. | Not Verified |
| MON-06 | Significant drift is flagged. | Drift summary. | Not Verified |
| MON-07 | Monitoring results are retained. | Monitoring history. | Not Verified |
| MON-08 | Drift findings are reviewed. | Review record. | Not Verified |

## 11.2 PSI Thresholds

| PSI Value | Interpretation | Recommended Review |
|---|---|---|
| Below 0.10 | No Significant Drift | Continue routine monitoring. |
| 0.10–0.25 | Moderate Drift | Investigate changing distributions. |
| Above 0.25 | Significant Drift | Perform detailed review and assess model suitability. |

## 11.3 Monitoring Limitations

PSI identifies changes in distributions. It does not independently establish that model predictive performance has deteriorated.

Where actual outcome labels become available, model performance should also be evaluated using appropriate metrics.

# 12. Model Registry Governance

The model registry maintains information about model versions and their evaluation results.

## 12.1 Required Registry Metadata

| Field | Description |
|---|---|
| Model ID | Unique model identifier. |
| Model Version | Version of the trained model. |
| Algorithm | Model algorithm name. |
| Training Timestamp | Date and time of model training. |
| ROC-AUC | Model ROC-AUC. |
| Gini | Model Gini coefficient. |
| KS | Model KS statistic. |
| Approval Status | Model review or approval state. |
| Model Artifact Path | Reference to the saved model file. |

## 12.2 Registry Checklist

| ID | Governance Check | Expected Evidence | Status |
|---|---|---|---|
| REG-01 | Model has a unique identifier. | Registry entry. | Not Verified |
| REG-02 | Model version is recorded. | Version metadata. | Not Verified |
| REG-03 | Algorithm name is recorded. | Registry metadata. | Not Verified |
| REG-04 | Training timestamp is recorded. | Timestamp. | Not Verified |
| REG-05 | Evaluation metrics are recorded. | AUC, Gini, and KS. | Not Verified |
| REG-06 | Approval status is recorded. | Approval field. | Not Verified |
| REG-07 | Model artifact can be retrieved. | Artifact path and load test. | Not Verified |
| REG-08 | Historical models can be reviewed. | Registry history. | Not Verified |

# 13. Model Approval and Release Controls

A model should be reviewed before being designated for business use.

## 13.1 Suggested Model Lifecycle Statuses

| Status | Meaning |
|---|---|
| Development | Model is being developed. |
| Validation Pending | Model is awaiting validation. |
| Approved | Model has completed the required review. |
| Rejected | Model did not meet the review requirements. |
| Retired | Model is no longer intended for use. |

## 13.2 Release Checklist

| ID | Release Control | Expected Evidence | Status |
|---|---|---|---|
| REL-01 | Model documentation is complete. | Model documentation. | Not Verified |
| REL-02 | Performance metrics are reviewed. | Evaluation report. | Not Verified |
| REL-03 | Explainability outputs are reviewed. | SHAP report. | Not Verified |
| REL-04 | Data quality checks are completed. | Data validation report. | Not Verified |
| REL-05 | Model artifact is saved. | Model file. | Not Verified |
| REL-06 | Registry entry is created. | Registry record. | Not Verified |
| REL-07 | Approval status is updated. | Approval record. | Not Verified |
| REL-08 | Rollback or previous-version reference is available where required. | Model history. | Not Verified |

# 14. Security and Data Privacy Governance

| ID | Governance Check | Expected Evidence | Status |
|---|---|---|---|
| SEC-01 | Customer data sources are documented. | Data source record. | Not Verified |
| SEC-02 | Sensitive information is identified. | Data inventory. | Not Verified |
| SEC-03 | Personal information is minimized where possible. | Dataset review. | Not Verified |
| SEC-04 | Uploaded data is handled securely. | Application configuration. | Not Verified |
| SEC-05 | Model artifacts are stored appropriately. | Storage controls. | Not Verified |
| SEC-06 | Secrets and credentials are not hardcoded. | Code review. | Not Verified |
| SEC-07 | Data retention requirements are documented. | Retention policy. | Not Verified |
| SEC-08 | Access controls are reviewed where implemented. | Access configuration. | Not Verified |

# 15. Model Risk Management

The following risks should be reviewed during model validation and periodic governance reviews.

| Risk ID | Model Risk | Potential Impact | Mitigation |
|---|---|---|---|
| MR-01 | Overfitting | Poor performance on unseen customers. | Cross-validation and holdout evaluation. |
| MR-02 | Data Leakage | Overestimated model performance. | Leakage assessment and proper data splitting. |
| MR-03 | Class Imbalance | Poor identification of acquisition prospects. | Evaluate class distribution and suitable metrics. |
| MR-04 | Data Drift | Changing customer distributions. | PSI monitoring and investigation. |
| MR-05 | Feature Bias | Unequal or inappropriate model behavior. | Feature review and subgroup evaluation where appropriate. |
| MR-06 | Explainability Misinterpretation | Incorrect business conclusions. | Review SHAP outputs with domain context. |
| MR-07 | Incorrect Model Version | Inconsistent predictions. | Model registry and version validation. |
| MR-08 | Poor Data Quality | Unreliable predictions. | Data quality gates. |
| MR-09 | Unvalidated Thresholds | Ineffective customer segmentation. | Business validation of segment boundaries. |
| MR-10 | Model Performance Degradation | Reduced predictive usefulness. | Outcome-based performance review when labels become available. |

# 16. Governance Review Schedule

A suggested review cadence is:

| Review Activity | Suggested Frequency |
|---|---|
| Data Quality Review | Every dataset upload or scoring batch. |
| Model Performance Review | After training and during periodic validation. |
| Drift Monitoring | Whenever a new monitoring dataset becomes available. |
| Registry Review | Whenever a model is registered or updated. |
| Explainability Review | During model validation and significant model changes. |
| Governance Checklist Review | During each formal model review. |
| Model Revalidation | According to business risk and model usage requirements. |

The actual cadence should be configured according to the operating environment and availability of monitoring data.

# 17. Governance Evidence and Documentation

The following artifacts should be retained where applicable:

| Artifact | Purpose |
|---|---|
| BRD.md | Documents business requirements. |
| UAT_Plan.md | Documents testing scenarios and acceptance criteria. |
| Governance_Checklist.md | Documents governance controls. |
| Data Quality Report | Records dataset validation results. |
| WOE-IV Report | Records feature assessment. |
| Model Evaluation Report | Records model performance. |
| SHAP Reports | Records model explanation outputs. |
| PSI Monitoring Report | Records distribution drift analysis. |
| Model Registry | Tracks model versions and metadata. |
| Automated Test Results | Provides validation evidence. |

# 18. Governance Status Summary

Use this section to record the current governance status.

| Governance Area | Status | Evidence Reference |
|---|---|---|
| Data Governance | Not Verified | Data Quality Report |
| Feature Governance | Not Verified | WOE-IV Report |
| Model Development | Not Verified | Training Configuration |
| Model Performance | Not Verified | Model Evaluation Report |
| Explainability | Not Verified | SHAP Report |
| Customer Scoring | Not Verified | Prediction Report |
| Drift Monitoring | Not Verified | PSI Report |
| Model Registry | Not Verified | Registry Database |
| Security and Privacy | Not Verified | Security Review |
| Model Approval | Not Verified | Approval Record |

**Status Definitions**

- **Not Verified:** Evidence has not yet been reviewed.
- **In Progress:** The control is being implemented or reviewed.
- **Implemented:** The control exists in the application.
- **Verified:** The control has been tested and supporting evidence reviewed.
- **Not Applicable:** The control does not apply to the current project scope.

# 19. Governance Sign-Off

| Role | Name | Approval Status |
|---|---|---|
| Model Developer | Prathmesh Aghor | Pending |
| Technical Reviewer | To Be Assigned | Pending |
| Model Validator | To Be Assigned | Pending |
| Business Reviewer | To Be Assigned | Pending |
| Governance Reviewer | To Be Assigned | Pending |

Approval should be recorded only after the relevant controls and evidence have been reviewed.

# 20. Conclusion

This checklist establishes a structured governance approach for the Customer Acquisition Propensity & Explainability Engine.

It provides a framework for reviewing data quality, model development, predictive performance, explainability, customer scoring, drift monitoring, model registration, and operational controls.

The objective is to make the platform more transparent, traceable, maintainable, and suitable for structured model review.

---

**End of Document**

**Customer Acquisition Propensity & Explainability Engine**  
*Turning customer data into explainable predictions and actionable business decisions.*