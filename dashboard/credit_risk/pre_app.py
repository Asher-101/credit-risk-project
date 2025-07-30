import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from credit_risk.data_ingest    import clean_data
from credit_risk.features       import generate_features
from credit_risk.fairness_audit import compute_fairness_metrics
from fairlearn.reductions       import ExponentiatedGradient, EqualizedOdds
from sklearn.metrics            import roc_curve, auc, precision_recall_curve, average_precision_score, ConfusionMatrixDisplay
from sklearn.calibration        import calibration_curve

# Streamlit Dashboard for Credit Risk Project

def main():
    st.title("Credit Risk Explorer")
    st.sidebar.header("Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload Borrowers CSV", type=["csv"])

    if uploaded_file is None:
        st.sidebar.info("Please upload a CSV file to proceed.")
        return
    
    # Load and clean data
    df = pd.read_csv(uploaded_file)
    df_clean = clean_data(df)
    
    # Feature generation
    x_train, x_test, y_train, y_test = generate_features(df_clean)

    # Load your trained pipeline (preproc + classifier)
    model = joblib.load("models/best_model.joblib")

    # Predictions table
    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]

    # ---ROC Curve & AUC---
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc     = auc(fpr, tpr)
    st.header("ROC Curve")
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0,1], [0,1], "--", color="grey", label="Chance")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    # --- Precision–Recall Curve & AP ---
    precision, recall, pr_thresh = precision_recall_curve(y_test, probs)
    ap_score = average_precision_score(y_test, probs)

    st.header("Precision-Recall Curve")
    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(recall, precision, label=f"AP = {ap_score:.2f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left")
    st.pyplot(fig_pr)

    # ---Calibration Curve---
        # visualizes whether predicted probabilities are well calibrated
    st.header("Calibration Curve")
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)

    fig_cal, ax_cal = plt.subplots()
    ax_cal.plot(prob_pred, prob_true, "o-")
    ax_cal.plot([0,1], [0,1], "--", color="grey", label="Perfectly Calibrated")
    ax_cal.set_xlabel("Mean Predicted Probability")
    ax_cal.set_ylabel("Fraction of Positives")
    ax_cal.legend()
    st.pyplot(fig_cal)

    # ---Confusion Matrix---
        # heatmap of TP/FP/TN/FN helps diagnose specific error patterns
    st.header("Confusion Matrix (@ default 0.5)")
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds, normalize=None, cmap="Blues", ax=ax_cm
    )
    st.pyplot(fig_cm)

    # ---Model Predictions---
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
    sens_test = pd.Series(
        age_groups.loc[y_test.index].astype(str),
        index=y_test.index,
        name="age_group"
    )
    fm_base = compute_fairness_metrics(y_test, preds, sens_test)

    st.header("Baseline Fairness Metrics by Age Group")
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
    fm_opt     = compute_fairness_metrics(y_test, y_pred_opt, sens_test)
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

    # ---Interactive Threshold---
        # lets user pick a probability cutoff and immediately see updated metrics & fairness
    st.header("Interactive Threshold")

    thr = st.slider("Probability threshold", 0.0, 1.0, 0.5, 0.01)
    preds_thr = (probs >= thr).astype(int)

    # show new accuracy & AUC under this threshold
    acc = (preds_thr == y_test).mean()
    st.metric("Accuracy", f"{acc:.3f}")

    # Recompute fairness under new preds
    fm_thr = compute_fairness_metrics(y_test, preds_thr, sens_test)
    st.subheader(f"Fairness at threshold = {thr:.2f}")
    st.table(fm_thr)

    # ---Probability Distribution Histogram---
        # show how predicted probabilities are distributed across all test samples
    st.header("Predicted Probability Distribution")
    fig,ax = plt.subplots()
    ax.hist([probs[y_test==0], probs[y_test==1]],
            bins=10, stacked=True, label=["No-default", "Default"])
    ax.set_xlabel("Predicted P(default=1)")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)

    # ---Lift Chart/Cumulative Gains---
        # shows how much better you do over random by selecting the top N% of highest-score applicants
    def lift_curve(y_true, y_score, nbins=10):
        df = pd.DataFrame({"y": y_true, "score": y_score})
        df["decile"] = pd.qcut(df["score"], nbins, labels=False)
        gains = df.groupby("decile")["y"].mean().iloc[::-1].reset_index(drop=True)
        return gains.cumsum() / gains.cumsum().iloc[-1]
    

    lift = lift_curve(y_test, probs, nbins=10)
    st.header("Lift Chart")
    fig_lift, ax_lift = plt.subplots()
    pct = np.arange(1, len(lift)+1) * (100/len(lift))
    ax_lift.plot(pct, lift, marker="o")
    ax_lift.set_xlabel("Top % of Population")
    ax_lift.set_ylabel("Cumulative True Positive Rate")
    st.pyplot(fig_lift)

    # ---Summary Statistics & Cohort Analysis---
        # quick table of key demographics or feature summaties, allow user to pick another sensitive attribute
        # (e.g employment length bins) and recompute fairness
    st.sidebar.header("Cohort Analysis")
    cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()
    cohort_col = st.sidebar.selectbox("Group by...", [""] + cat_cols)
    if cohort_col:
        cohort = df_clean[cohort_col]
        df_cohort = cohort.value_counts().rename_axis(cohort_col).reset_index(name="count")
        st.subheader(f"Counts by {cohort_col}")
        st.bar_chart(df_cohort.set_index(cohort_col)["count"])


    # --- Global feature importances from the model itself ---
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

if __name__ == "__main__":
    main()