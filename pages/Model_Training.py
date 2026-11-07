import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# ============================================================
# DASHBOARD
# ============================================================

st.title("Customer Acquisition Propensity & Explainability Engine")

if "df" not in st.session_state:
    st.warning("Please upload a dataset first in the Data Quality Page.")
    st.stop()

df = st.session_state["df"].copy()

st.dataframe(df.head(), use_container_width=True)

target = st.selectbox("Select Target Variable", df.columns)

st.session_state["target"] = target


# ============================================================
# MODEL TRAINING
# ============================================================

if st.button("Train Models"):

    # Remove missing target values
    df = df.dropna(subset=[target]).copy()

    y_original = df[target]

    if y_original.nunique() != 2:
        st.error("Target must contain exactly two classes.")
        st.stop()

    # --------------------------------------------------------
    # TARGET ENCODING
    # --------------------------------------------------------

    observed_classes = list(pd.unique(y_original))

    if set(observed_classes) == {"no", "yes"}:
        negative_label = "no"
        positive_label = "yes"
    else:
        negative_label = observed_classes[0]
        positive_label = observed_classes[1]

    label_to_int = {
        negative_label: 0,
        positive_label: 1
    }

    y = y_original.map(label_to_int).astype("int64")

    X = df.drop(columns=[target])

    # --------------------------------------------------------
    # FEATURE IDENTIFICATION
    # --------------------------------------------------------

    cat_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    num_cols = X.select_dtypes(
        exclude=["object", "category", "bool"]
    ).columns.tolist()

    # --------------------------------------------------------
    # PREPROCESSING PIPELINE
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())
                    ]
                ),
                num_cols
            ),

            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent")
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore")
                        )
                    ]
                ),
                cat_cols
            )

        ],
        remainder="drop"
    )

    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # MACHINE LEARNING MODELS
    # --------------------------------------------------------

    models = {

        "Logistic Regression":
            LogisticRegression(max_iter=5000),

        "Random Forest":
            RandomForestClassifier(random_state=42),

        "Gradient Boosting":
            GradientBoostingClassifier(random_state=42),

        "XGBoost":
            XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_jobs=2
            ),

        "LightGBM":
            LGBMClassifier(
                random_state=42,
                verbosity=-1
            ),

        "CatBoost":
            CatBoostClassifier(
                verbose=0,
                random_state=42,
                thread_count=2
            )
    }

    results = []
    trained_models = {}

    fig_roc, ax_roc = plt.subplots(figsize=(10, 7))
    fig_pr, ax_pr = plt.subplots(figsize=(10, 7))

    # ========================================================
    # TRAIN AND EVALUATE MODELS
    # ========================================================

    for name, model in models.items():

        st.write(f"Training: {name}")

        pipe = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", model)
            ]
        )

        pipe.fit(X_train, y_train)

        trained_models[name] = pipe

        # Predictions
        pred = pipe.predict(X_test)

        # Positive class probability
        positive_index = list(pipe.classes_).index(1)

        prob = pipe.predict_proba(X_test)[:, positive_index]

        # ----------------------------------------------------
        # EVALUATION METRICS
        # ----------------------------------------------------

        auc = roc_auc_score(y_test, prob)

        gini = 2 * auc - 1

        acc = accuracy_score(y_test, pred)

        prec = precision_score(
            y_test,
            pred,
            pos_label=1,
            zero_division=0
        )

        rec = recall_score(
            y_test,
            pred,
            pos_label=1,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            pred,
            pos_label=1,
            zero_division=0
        )

        # ----------------------------------------------------
        # KS STATISTIC
        # ----------------------------------------------------

        fpr, tpr, _ = roc_curve(
            y_test,
            prob,
            pos_label=1
        )

        ks_value = float((tpr - fpr).max())

        results.append([
            name,
            acc,
            prec,
            rec,
            f1,
            auc,
            gini,
            ks_value
        ])

        # ----------------------------------------------------
        # ROC CURVE
        # ----------------------------------------------------

        ax_roc.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc:.3f})"
        )

        # ----------------------------------------------------
        # PRECISION RECALL CURVE
        # ----------------------------------------------------

        precision, recall, _ = precision_recall_curve(
            y_test,
            prob,
            pos_label=1
        )

        ax_pr.plot(
            recall,
            precision,
            label=f"{name} (F1={f1:.3f})"
        )

    # ========================================================
    # FINALIZE ROC CURVE
    # ========================================================

    ax_roc.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray"
    )

    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve Comparison")
    ax_roc.legend()

    # ========================================================
    # FINALIZE PRECISION RECALL CURVE
    # ========================================================

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve Comparison")
    ax_pr.legend()

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "AUC",
            "GINI",
            "KS"
        ]
    )

    # ========================================================
    # BEST MODEL SELECTION
    # ========================================================

    best_model_name = results_df.sort_values(
        "AUC",
        ascending=False
    ).iloc[0]["Model"]

    best_model = trained_models[best_model_name]

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        best_model,
        "models/best_model.pkl"
    )

    best_ks = results_df.loc[
        results_df["Model"] == best_model_name,
        "KS"
    ].iloc[0]

    st.success(
        f"Best Model: {best_model_name} | "
        f"AUC = {results_df['AUC'].max():.4f} | "
        f"KS = {best_ks:.4f}"
    )

    # ========================================================
    # DASHBOARD TABS
    # ========================================================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Model Comparison",
        "ROC & PR Curves",
        "Confusion Matrix",
        "Lift & KS",
        "Feature Importance"
    ])

    # ========================================================
    # TAB 1: MODEL COMPARISON
    # ========================================================

    with tab1:

        st.subheader("Model Performance Comparison")

        st.dataframe(
            results_df.style.highlight_max(
                subset=["AUC", "KS"],
                color="lightgreen"
            ),
            use_container_width=True
        )

    # ========================================================
    # TAB 2: ROC AND PR CURVES
    # ========================================================

    with tab2:

        st.subheader("ROC Curve Comparison")

        st.pyplot(fig_roc)

        st.subheader("Precision-Recall Curve Comparison")

        st.pyplot(fig_pr)

    # ========================================================
    # TAB 3: CONFUSION MATRIX
    # ========================================================

    with tab3:

        st.subheader(
            f"Confusion Matrix - {best_model_name}"
        )

        best_pred = best_model.predict(X_test)

        cm = confusion_matrix(
            y_test,
            best_pred,
            labels=[0, 1]
        )

        fig_cm, ax_cm = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax_cm,
            xticklabels=[
                str(negative_label),
                str(positive_label)
            ],
            yticklabels=[
                str(negative_label),
                str(positive_label)
            ]
        )

        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")

        st.pyplot(fig_cm)

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            best_pred,
            labels=[0, 1],
            target_names=[
                str(negative_label),
                str(positive_label)
            ],
            output_dict=True,
            zero_division=0
        )

        st.dataframe(
            pd.DataFrame(report).transpose(),
            use_container_width=True
        )

    # ========================================================
    # TAB 4: LIFT AND KS ANALYSIS
    # ========================================================

    with tab4:

        st.subheader(
            f"Lift Chart - {best_model_name}"
        )

        positive_index = list(best_model.classes_).index(1)

        best_prob = best_model.predict_proba(
            X_test
        )[:, positive_index]

        risk_df = pd.DataFrame({
            "Probability": best_prob
        })

        # ----------------------------------------------------
        # PROPENSITY SEGMENTS
        # ----------------------------------------------------

        risk_df["Risk Segment"] = pd.cut(
            risk_df["Probability"],
            bins=[
                -0.000001,
                0.3,
                0.7,
                1.000001
            ],
            labels=[
                "Low Propensity",
                "Medium Propensity",
                "High Propensity"
            ],
            include_lowest=True
        )

        st.subheader("Propensity Segment Distribution")

        st.dataframe(
            risk_df["Risk Segment"]
            .value_counts()
            .rename_axis("Segment")
            .reset_index(name="Customers"),
            use_container_width=True
        )

        # ----------------------------------------------------
        # DECILE ANALYSIS
        # ----------------------------------------------------

        decile_df = pd.DataFrame({
            "Actual": y_test.to_numpy(),
            "Probability": best_prob
        })

        decile_df = decile_df.sort_values(
            "Probability",
            ascending=False
        ).reset_index(drop=True)

        decile_df["Decile"] = pd.qcut(
            decile_df.index,
            q=10,
            labels=False,
            duplicates="drop"
        ) + 1

        decile_summary = (
            decile_df.groupby(
                "Decile",
                observed=False
            )
            .agg(
                Customers=("Actual", "count"),
                Responders=("Actual", "sum"),
                Response_Rate=("Actual", "mean")
            )
            .reset_index()
        )

        st.subheader("Decile Analysis")

        st.dataframe(
            decile_summary,
            use_container_width=True
        )

        # ----------------------------------------------------
        # LIFT CHART
        # ----------------------------------------------------

        lift = decile_summary.set_index(
            "Decile"
        )["Response_Rate"]

        fig_lift, ax_lift = plt.subplots(
            figsize=(10, 6)
        )

        lift.plot(
            kind="bar",
            ax=ax_lift
        )

        ax_lift.axhline(
            y=float(y_test.mean()),
            linestyle="--",
            label="Overall Response Rate"
        )

        ax_lift.set_ylabel("Response Rate")

        ax_lift.set_xlabel(
            "Decile (1 = Highest Predicted Propensity)"
        )

        ax_lift.legend()

        st.pyplot(fig_lift)

        st.metric(
            "KS Statistic (Best Model)",
            round(float(best_ks), 4)
        )

    # ========================================================
    # TAB 5: FEATURE IMPORTANCE
    # ========================================================

    with tab5:

        st.subheader("Feature Importance")

        model_step = best_model.named_steps["model"]

        if hasattr(model_step, "feature_importances_"):

            feature_names = (
                best_model
                .named_steps["preprocessor"]
                .get_feature_names_out()
            )

            importances = model_step.feature_importances_

            if len(feature_names) == len(importances):

                feat_imp = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances
                }).sort_values(
                    "Importance",
                    ascending=False
                )

                st.dataframe(
                    feat_imp.head(20),
                    use_container_width=True
                )

            else:

                st.warning(
                    "Feature names and importance values have different lengths."
                )

        else:

            st.info(
                f"{best_model_name} does not support feature importance."
            )

        # ----------------------------------------------------
        # CROSS VALIDATION
        # ----------------------------------------------------

        st.subheader("Cross Validation Performance")

        minority_count = int(
            y.value_counts().min()
        )

        cv_folds = min(5, minority_count)

        if cv_folds >= 2:

            cv_auc = cross_val_score(
                best_model,
                X,
                y,
                cv=cv_folds,
                scoring="roc_auc",
                n_jobs=1
            ).mean()

            st.metric(
                f"Mean CV AUC ({cv_folds}-Fold)",
                f"{cv_auc:.4f}"
            )

        else:

            st.info(
                "Not enough examples in the minority class "
                "for stratified cross-validation."
            )

    # ========================================================
    # SAVE RESULTS TO SESSION STATE
    # ========================================================

    st.session_state["best_model"] = best_model

    st.session_state["trained_models"] = trained_models

    st.session_state["preprocessor"] = (
        best_model.named_steps["preprocessor"]
    )

    st.session_state["results_df"] = results_df

    st.session_state["best_model_name"] = best_model_name

    st.session_state["X"] = X

    st.session_state["num_cols"] = num_cols

    st.session_state["cat_cols"] = cat_cols

    st.session_state["df"] = df

    st.session_state["X_train"] = X_train

    st.session_state["y_train"] = y_train

    st.session_state["X_test"] = X_test

    st.session_state["y_test"] = y_test

    st.session_state["negative_label"] = negative_label

    st.session_state["positive_label"] = positive_label

    st.session_state["training_data"] = X_train.copy()

    st.success("Models Trained Successfully!")