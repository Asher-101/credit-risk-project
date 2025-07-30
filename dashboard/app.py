#import os, sys

## ensure top-level src/ and the vendored dashboard/credit_risk/ are on sys.path
#root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#src = os.path.join(root, "src")
#vend = os.path.join(os.path.dirname(__file__), "credit_risk")

#for p in (src, vend):
#    if os.path.isdir(p) and p not in sys.path:
#       sys.path.insert(0, p)

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from credit_risk.data_ingest    import clean_data, load_data
from credit_risk.features       import generate_features
from credit_risk.fairness_audit import compute_fairness_metrics
from fairlearn.reductions       import ExponentiatedGradient, EqualizedOdds
from sklearn.metrics            import roc_curve, auc, precision_recall_curve, average_precision_score, ConfusionMatrixDisplay

# CACHING LAYERS----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_cleaned(df: pd.DataFrame) -> pd.DataFrame:
    return clean_data(df)

@st.cache_data(show_spinner=False)
def get_features(df: pd.DataFrame, target: str):
    return generate_features(df, target_col=target, stratify=True)

@st.cache_resource(show_spinner=False)
def load_trained_model(path: str = "models/best_model.joblib"):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_sample_raw():
    # assumes data/raw/borrowers.csv checked into repo
    return load_data("data/raw/borrowers.csv")

# DASHBOARD---------------------------------------------------------------------------------

def main():
    st.title("Credit Risk Explorer")

    # ---Sidebar: input + sample-download ---
    st.sidebar.header("Data Input")
    uploaded = st.sidebar.file_uploader("Upload Borrowers CSV", type=["csv"])
    st.sidebar.markdown("---")
    st.sidebar.caption("Or download a template:")
    sample_df = load_sample_raw()
    st.sidebar.download_button(
        "Download sample CSV",
        data=sample_df.to_csv(index=False),
        file_name="borrowers_template.csv",
        mime="text/csv",
    )
    if uploaded is None:
        st.sidebar.info("Please upload a CSV file to proceed.")
        return
    
    # Load and clean data
    df = pd.read_csv(uploaded)
    df_clean = get_cleaned(df)
    
    # ---pick target---
    binary_cols = [
        c for c in df_clean.columns
        if df_clean[c].dropna().nunique() == 2
    ]

    target_col = st.sidebar.selectbox(
        "Select your binary target column",
        options=binary_cols
    )

    # Feature generation using their choice (cached)
    x_train, x_test, y_train, y_test = get_features(df_clean, target_col)

    #if they aren't already {0,1}, build a mapping
    classes = sorted(y_train.unique())
    if set(classes) != {0,1}:
        mapping = { classes[0]: 0, classes[1]:1 }
        y_train = y_train.map(mapping).astype(int)
        y_test  = y_test.map(mapping).astype(int)


    # Load model (cached)
    model = load_trained_model()
    preproc = model.named_steps["preproc"]
    req_feats = preproc.feature_names_in_

    # align columns
        # reindex x's so any missing columns get filled with zeros
    x_train = x_train.reindex(columns=req_feats, fill_value=0)
    x_test  = x_test. reindex(columns=req_feats, fill_value=0)

    # predictions (cached once)
    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]

    # --- Performance tab: show ROC/PR + confusion matrix at slider point ---
    perf, fair, cohort = st.tabs(["Performance", "Fairness", "Cohorts"])

    with perf:
        # detect whether we've got exactly two classes
        st.markdown("### Performance Metrics")
        st.caption("Slide the threshold to see how the ROC curve, Precision-Recall curve and confusion matrix update at that operating point.")
        if y_test.nunique() == 2:   
            thr = st.slider("Probability threshold", 0.0, 1.0, 0.5, 0.01)
            preds_thr = (probs >= thr).astype(int)

            # ---ROC Curve & AUC + annotated operating point ---
            st.markdown("### ROC Curve")
            st.caption("Plot of True Positive vs False Positive rate; red dot shows current threshold.")
            fpr, tpr, roc_th = roc_curve(y_test, probs)
            roc_auc     = auc(fpr, tpr)
            # find index where roc_th is closest to thr
            idx = np.nanargmin(np.abs(roc_th - thr))
            fig1, ax1 = plt.subplots()
            ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax1.scatter(fpr[idx], tpr[idx],
                        color="red", label=f"op @ thr={thr:.2f}")
            ax1.plot([0,1], [0,1], "--", color="grey", label="Chance")
            ax1.set_xlabel("False Positive Rate")
            ax1.set_ylabel("True Positive Rate")
            ax1.legend(loc="lower right")
            st.pyplot(fig1)

            # --- Precision–Recall Curve & AP + point---
            st.markdown("### Precision-Recall Curve")
            st.caption("Trade-off between precision and recall; red dot is at the chosen threshold.")
            precision, recall, pr_th = precision_recall_curve(y_test, probs)
            ap = average_precision_score(y_test, probs)
            # pr_th has length = len(precision)-1, so append a dummy threshold=1.0
            pr_th_full = np.append(pr_th, 1.0)
            pr_idx = np.nanargmin(np.abs(pr_th_full - thr))
            fig2, ax2 = plt.subplots()
            ax2.plot(recall, precision, label=f"AP = {ap:.2f}")
            ax2.scatter(recall[pr_idx], precision[pr_idx],
                        color="red", label=f"op @ thr={thr:.2f}")
            ax2.set_xlabel("Recall")
            ax2.set_ylabel("Precision")
            ax2.legend(loc="best")
            st.pyplot(fig2)


            # ---Confusion Matrix ---
                # heatmap of TP/FP/TN/FN helps diagnose specific error patterns
            st.markdown("### Confusion Matrix")
            st.caption("Counts of TP, FP, TN, FN at the selected threshold.")
            fig3, ax3 = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(
                y_test, preds_thr, normalize=None, cmap="Blues", ax=ax3
            )
            ax3.set_title(f"Confusion matrix @ thr={thr:.2f}")
            st.pyplot(fig3)
        else:
            st.warning("Your target column has more than two classes, so binary-only metrics (ROC, PR) are disabled.")


    with fair:
        # only predictions, baseline slices & mitigation plots
        st.markdown("## Fairness Audit")
        st.caption("Examine approval rates, error rates and Equalized Odds before and after mitigation.")
        st.header("Model Predictions (Test Set)")
        results = pd.DataFrame({
            "True": y_test,
            "Predicted": preds,
            "Probability": probs
        },index=y_test.index)
        st.dataframe(results)

        # ---Baseline fairness---
        age_groups = pd.cut(
            df_clean["age"],
            bins=[18, 25, 35, 45, 55, 65, 100],
            labels=['18-24','25-34','35-44','45-54','55-64','65+'],
            include_lowest=True
        )
        sens_default = pd.Series(
            age_groups.loc[y_test.index].astype(str),
            index=y_test.index,
            name="age_group"
        )
        fm_base = compute_fairness_metrics(y_test, preds, sens_default)

        # show the default baseline
        st.header("Baseline Fairness Metrics by Age Group")
        st.table(fm_base)

        # allows user pick *another* slice
        slice_col = st.selectbox(
            "Slice fairness by...",
            options=[ None ] + df_clean.select_dtypes(["category","object","int","float"]).columns.tolist()
        )
        if slice_col:
            sens_new = df_clean[slice_col].loc[y_test.index]
            fm_base = compute_fairness_metrics(y_test, preds, sens_new)
            st.subheader(f"Fairness Metrics by {slice_col}")
            st.table(fm_base)

        # Equalized-Odds mitigation (route weights into the 'clf' step)
        mitigator = ExponentiatedGradient(
            model,
            constraints=EqualizedOdds(),
            eps=0.01,
            max_iter=50,
            sample_weight_name="clf__sample_weight"
        )
        sens_train = pd.Series(age_groups.loc[y_train.index].astype(str),
                               index=y_train.index,
                               name="age_group")
        mitigator.fit(x_train, y_train, sensitive_features=sens_train)

        y_pred_opt = mitigator.predict(x_test)
        fm_opt     = compute_fairness_metrics(y_test, y_pred_opt, sens_default)
        st.write("fm_base:", fm_base)
        st.write("fm_opt:", fm_opt)

        # Plot before vs after side-by-side
        st.subheader("Fairness: Before vs After Mitigation")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before**")
            fig1, ax1 = plt.subplots(figsize=(4,3))
            fm_base.plot.bar(ax=ax1)
            plt.xticks(rotation=45)
            st.pyplot(fig1)
        with col2:
            st.markdown("**After Equalized Odds**")
            fig2, ax2 = plt.subplots(figsize=(4,3))
            fm_opt.plot.bar(ax=ax2)
            plt.xticks(rotation=45)
            st.pyplot(fig2)

        # ---Probability Distribution Histogram---
            # show how predicted probabilities are distributed across all test samples
        st.markdown("### Predicted Probability Histogram")
        st.caption("Shows how model scores are distributed for each class.")
        st.header("Predicted Probability Distribution")
        fig_hist,ax_hist = plt.subplots()
        ax_hist.hist(
            [probs[y_test==0], probs[y_test==1]],
            bins=10, stacked=True,
            label=["Class 0", "Class 1"]
        )
        ax_hist.set_xlabel("Predicted Probability")
        ax_hist.set_ylabel("Count")
        ax_hist.legend()
        st.pyplot(fig_hist)

        # ---Lift Chart/Cumulative Gains---
            # shows how much better you do over random by selecting the top N% of highest-score applicants
        st.markdown("### Lift Chart / Cumulative Gains")
        st.caption("Cumulative true positive rate when targeting the top N% of borrowers by score.")
        def lift_curve(y_true, y_score, nbins=10):
            df_1 = pd.DataFrame({"y": y_true, "score": y_score})
            df_1["decile"] = pd.qcut(df_1["score"], nbins, labels=False)
            gains = df_1.groupby("decile")["y"].mean().iloc[::-1].reset_index(drop=True)
            return gains.cumsum() / gains.cumsum().iloc[-1]


        lift = lift_curve(y_test, probs, nbins=10)
        st.header("Lift Chart")
        fig_lift, ax_lift = plt.subplots()
        pct = np.linspace(1/len(lift), 1.0, len(lift)) * 100
        ax_lift.plot(pct, lift, marker="o")
        ax_lift.set_xlabel("Top % of Population")
        ax_lift.set_ylabel("Cumulative True Positive Rate")
        st.pyplot(fig_lift)
    
    with cohort:
        # ---Summary Statistics & Cohort Analysis---
            # quick table of key demographics or feature summaties, allow user to pick another sensitive attribute
            # (e.g employment length bins) and recompute fairness
        st.markdown("## Cohort Anaysis")
        st.caption("Explore counts and distributions across different demographic or feature-based groups.")
        st.header("Cohort Analysis")
        cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()
        cohort_col = st.selectbox("Group by...", [""] + cat_cols)
        if cohort_col:
            cohort = df_clean[cohort_col]
            df_cohort = cohort.value_counts().rename_axis(cohort_col).reset_index(name="count")
            st.subheader(f"Counts by {cohort_col}")
            st.bar_chart(df_cohort.set_index(cohort_col)["count"])


    # --- Global feature importances + download buttons(always visible) ---
    st.header("Global feature importances")

    # extract preprocessor & classifier
    preproc = model.named_steps["preproc"]
    clf     = model.named_steps["clf"]

    # transform once so we know the output shape
    x_tst = preproc.transform(x_test)    # shape (n_test, n_feats_out)

    # get the exact feature names out of the transformer
    feat_names = preproc.get_feature_names_out()
    assert x_tst.shape[1] == len(feat_names)

    # pull the importances
    if hasattr(clf, "feature_importances_"):
        imps = clf.feature_importances_
    else:
        # logistic regression case: use absolute coefficients
        imps = np.abs(clf.coef_.ravel())

    # build a Series, sort and take top 10
    imp = pd.Series(imps, index=feat_names)
    imp = imp.sort_values(ascending=False).head(10)

    # render as a bar chart
    st.bar_chart(imp)

    # ---Download buttons---
        # let users export the cleaned data or the results table
    st.download_button(
        "Download results as CSV",
        data=results.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download cleaned data",
        data=df_clean.to_csv(index=False),
        file_name="cleaned_data.csv"
    )
    st.download_button(
        "Download trained model",
        data=open("models/best_model.joblib", "rb").read(),
        file_name="best_model.joblib",
        mime="application/octet-stream"
    )

if __name__ == "__main__":
    main()