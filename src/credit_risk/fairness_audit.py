import os
import argparse
import pandas as pd
import joblib

from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate, true_positive_rate
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
import matplotlib.pyplot as plt

from credit_risk.data_ingest import load_data, clean_data
from credit_risk.features import generate_features
#-------------------------------------------------------------
# src/fairness_audit.py
# Fairness metrics and post-processing for credit-risk-project
#-------------------------------------------------------------

def compute_fairness_metrics(y_true, y_pred, sensitive_features):
    """
    Returns a DataFrame of fairness metrics (Selection Rate, TPR, FPR) by sensitive group.
    """
    metrics = {
        'Selection Rate': selection_rate,
        'True Positive Rate': true_positive_rate, 
        'False Positive Rate': false_positive_rate
    }
    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    df = mf.by_group
    df.index.name = sensitive_features.name
    return df


def plot_fairness_by_group(metrics_df: pd.DataFrame, output_path: str):
    """
    Saves bar plots of each fairness metric by group to the specified path.
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, len(metrics_df.columns), figsize=(5*len(metrics_df.columns), 4), sharey=True)
    if len(metrics_df.columns) == 1:
        axes = [axes]
    
    for ax, col in zip(axes, metrics_df.columns):
        metrics_df[col].plot(kind="bar", ax=ax, title=col)
        ax.set_xlabel(metrics_df.index.name)
        if ax is axes[0]:
            ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Saved fairness plot -> {output_path}")


def apply_expgrad_mitigator(
        estimator,
        x_train, y_train,
        x_test,
        sensitive_train,    # pandas Series
        sensitive_test      # not used by the ExponentiatedGradient itself
):
    """
    Train an ExponentiatedGradient mitigator to enforce equalized odds,
    then return both the fitted mitigator and its test-time predictions"""
    mitigator = ExponentiatedGradient(
        estimator, 
        constraints=EqualizedOdds(),
        eps=0.01,       # how close to perfectly fair
        max_iter=50,    # small problem, so it will converge fast
        )
    
    # must pass plain 1D arrays for sensitive_features
    sf_train = pd.Series(sensitive_train.values, name="age_group")
    mitigator.fit(x_train, y_train, sensitive_features=sf_train)

    # call .predict on the test set
    y_pred_opt = mitigator.predict(x_test)
    return mitigator, y_pred_opt


def main():
    parser = argparse.ArgumentParser(prog="credit-risk-audit",
        description="Run a fairness audit on a trained credit-risk model")
    parser.add_argument("--data",
                        default="data/raw/borrowers.csv",
                        help="path to raw CSV file")
    parser.add_argument("--report-dir",
                        default="reports",
                        help="directory to save fairness plots")
    parser.add_argument("--constraint",
                        default="equalized_odds",
                        choices=["equalized_odds", "demographic_parity"])
    parser.add_argument("--no-plot", action="store_true",
                        help="skip saving/plotting figures (CI-friendly)")
    args = parser.parse_args()

    # Ingest and clean
    df_raw = load_data(args.data)
    df_clean = clean_data(df_raw)

    # Generate features
    x_train, x_test, y_train, y_test = generate_features(df_clean)

    # Train model
    from credit_risk.train_model import train_and_evaluate
    model_path = train_and_evaluate(x_train, x_test, y_train, y_test)
    model = joblib.load(model_path)

    # build group labels as pure-Python strings
    age_bins = [18,25,35,45,55,65,100]
    age_labels = ['18-24','25-34','35-44','45-54','55-64', '65+']
    sens = pd.cut(df_clean['age'], bins=age_bins, labels=age_labels, include_lowest=True)
    sens_train = sens.loc[y_train.index].astype(str)
    sens_test = sens.loc[y_test.index].astype(str)

    print("=== Baseline Fairness Metrics ===")
    y_pred = model.predict(x_test)
    fm = compute_fairness_metrics(y_test, y_pred, sensitive_features=sens_test)
    print(fm)
    if not args.no_plot:
        path = os.path.join(args.report_dir, f"fairness_baseline.png")
        plot_fairness_by_group(fm, path)

    # Post-process for specified constraint (only equalized_odds supported currently)
    if args.constraint != "equalized_odds":
        raise NotImplementedError(f"Constraint {args.constraint!r} not supported")
    print(f"\n=== Applying ExponentiatedGradient ({args.constraint}) ===")
    mitigator, y_pred_opt = apply_expgrad_mitigator(
        model, x_train, y_train, x_test,
        sensitive_train = pd.Series(sens_train.values, name="age_group"),
        sensitive_test = sens_test
    )

    fm_opt = compute_fairness_metrics(y_test, y_pred_opt, sensitive_features=sens_test)
    print(f"\n=== Fairness metrics (after {args.constraint}) ===")
    print(fm_opt)
    if not args.no_plot:
        path = os.path.join(args.report_dir, f"fairness_{args.constraint}.png")
        plot_fairness_by_group(fm_opt, path)


if __name__ == "__main__":
    main()