import os
import click
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

from credit_risk.data_ingest import load_data, clean_data
from credit_risk.features    import generate_features
from fairlearn.metrics       import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from fairlearn.reductions    import ExponentiatedGradient, EqualizedOdds

__version__ = "0.1.0"

def compute_fairness_metrics(y_true, y_pred, sensitive_features):
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

def plot_bar(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(1, len(df.columns), figsize=(5*len(df.columns), 4), sharey=True)
    if len(df.columns) == 1:
        axes=[axes]
    for ax, col in zip(axes, df.columns):
        df[col].plot(kind="bar", ax=ax, title=col)
        ax.set_xlabel(df.index.name)
        if ax is axes[0]:
            ax.set_ylabel(df.index.name)
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    click.echo(f"Saved plot to {path}")

@click.group()
@click.version_option(__version__, prog_name="credit-risk")
def cli():
    """credit-risk: train, audit fairness, and explain with SHAP."""
    pass

@cli.command()
@click.option("--data", default="data/raw/borrowers.csv", help="Path to raw CSV")
@click.option("--outdir", default="reports",            help="Where to dump metrics and plots")
def audit(data, outdir):
    """Run training + fairness audit (equalized odds via ExponentiatedGradient)."""
    df = load_data(data)
    df_clean = clean_data(df)
    x_train, x_test, y_train, y_test = generate_features(df_clean)
    
    #train and save model
    from credit_risk.train_model import train_and_evaluate
    model_path = train_and_evaluate(x_train, x_test, y_train, y_test)
    model = joblib.load(model_path)

    # baseline fairness
    sens = pd.cut(df_clean["age"],
                  bins=[18,25,35,45,55,65,100],
                  labels=['18-24','25-34','35-44','45-54','55-64','65+'],
                  include_lowest=True)
    sens_test = pd.Series(sens.loc[y_test.index].astype(str),
                          index=y_test.index,
                          name="age_group")
    
    y_pred = model.predict(x_test)
    fm = compute_fairness_metrics(y_test, y_pred, sens_test)
    click.echo("=== Baseline fairnes ===")
    click.echo(fm.to_string())
    plot_bar(fm, os.path.join(outdir, "fairness_baseline.png"))

    # equalized odds via ExponentiatedGradient
    mitigator = ExponentiatedGradient(model,
                                      constraints=EqualizedOdds(),
                                      eps=0.01,
                                      max_iter=50)
    sf_train = pd.Series(sens.loc[y_train.index].astype(str),
                         index=y_train.index,
                         name="age_group")
    mitigator.fit(x_train, y_train, sensitive_features=sf_train)
    y_pred_opt = mitigator.predict(x_test)

    fm2 = compute_fairness_metrics(y_test, y_pred_opt, sens_test)
    click.echo("=== After equalized odds ===")
    click.echo(fm2.to_string())
    plot_bar(fm2, os.path.join(outdir, "fairness_equalized_odds.png"))

@cli.command()
@click.option("--data",       default="data/raw/borrowers.csv", help="Path to raw CSV")
@click.option("--background", default=50,                       help="Background sample size")
@click.option("--outpath",    default="reports/shap_force.png", help="Where to save force plot")
def explain(data, background, outpath):
    """Compute a SHAP force‐plot for one test example."""

    # ingest, clean, features…
    df        = load_data(data)
    df_clean  = clean_data(df)
    x_train, x_test, y_train, y_test = generate_features(df_clean)

    # train (or load) your model…
    from credit_risk.train_model import train_and_evaluate
    model_path = train_and_evaluate(x_train, x_test, y_train, y_test)
    model      = joblib.load(model_path)

    # hard‐code “first” test row
    # idx already passed in
    # sample background & explainer…
    idx = 0
    # sample background & explainer…
    bg        = x_train.sample(background, replace=True, random_state=0).values.astype(float)
    explainer = shap.Explainer(model.predict_proba, bg)
    exp       = explainer(x_test.values.astype(float))

    # pull out the class‐1 slice for idx=0
    vals      = exp.values         # shape (n_samples, n_classes, n_features)
    base_vals = exp.base_values    # shape (n_classes,)
    sv1       = vals[idx, 1, :]
    ev1       = base_vals[1]

    # generate & save force‐plot
    shap.initjs()
    fig = shap.force_plot(ev1, sv1, x_test.iloc[idx], matplotlib=True)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    click.echo(f"Saved SHAP force plot (idx={idx}) to {outpath}")

if __name__ == "__main__":
    cli()
