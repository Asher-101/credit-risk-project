import sys, os
import joblib
import warnings
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from credit_risk.features import build_preprocessor

# add project tool (one level up) to Python's module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from credit_risk.data_ingest import load_data, clean_data

warnings.filterwarnings("ignore", category = UserWarning)


def train_and_evaluate(x_train, x_test, y_train, y_test, output_dir = "models"):
    # build shared preprocessor once
    preproc = build_preprocessor(x_train)

    # wrap each model in a pipeline
    pipe_lr = Pipeline([
        ("preproc", preproc),
        ("clf",     LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe_rf = Pipeline([
        ("preproc", preproc),
        ("clf",     RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # fit and evaluate LR
    pipe_lr.fit(x_train, y_train)
    lr_proba = pipe_lr.predict_proba(x_test)[:, 1]
    lr_pred  = pipe_lr.predict(x_test)

    print("=== Logistic Regression Metrics ===")
    print("ROC-AUC:", roc_auc_score(y_test, lr_proba))
    print(classification_report(y_test, lr_pred))

    # fit and evaluate RF
    pipe_rf.fit(x_train, y_train)
    rf_proba = pipe_rf.predict_proba(x_test)[:, 1]
    rf_pred  = pipe_rf.predict(x_test)


    print("\n=== Random Forest Metrics ===")
    print("ROC-AUC:", roc_auc_score(y_test, rf_proba))
    print(classification_report(y_test, rf_pred))

    # pick whichever pipeline gave higher AUC
    best_model = pipe_rf if roc_auc_score(y_test, rf_proba) > roc_auc_score(y_test, lr_proba) else pipe_lr
    Path(output_dir).mkdir(exist_ok=True)
    model_path = Path(output_dir) / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"\n Saved best model to {model_path}")

    return model_path


if __name__ == "__main__":
    from credit_risk.data_ingest import load_data, clean_data
    from credit_risk.features import generate_features

    # Run end-to-end from raw CSV
    df_raw = load_data("data/raw/borrowers.csv")
    df_clean = clean_data(df_raw)
    x_train, x_test, y_train, y_test = generate_features(
        df_clean, 
        stratify=False     #disable stratification
    )

    train_and_evaluate(x_train, x_test, y_train, y_test)