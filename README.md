# Credit-Risk Scoring & Fairness Analysis
A streamlit app for exploring credit risk modeling, including data cleaning, 
feature generation, fairness audits, and model performance visualization.

## Repository Structure
        ```
        credit-risk-project/
        ├── .github/                 # GitHub workflows, issue templates, etc.
        ├── .devcontainer/           # Devcontainer definition (if you use Codespaces)
        ├── .venv/                   # (Optionally) your local virtual-env directory
        ├── dashboard/
        │   ├── app.py               # Main Streamlit app (entrypoint)
        │   └── credit_risk/         # Vendored modules for deployment
        │       ├── __init__.py      # Package marker
        │       ├── data_ingest.py   # Data loading and cleaning
        │       ├── features.py      # Feature generation logic
        │       ├── fairness_audit.py# Fairness metrics functions
        │       └── train_model.py   # (if you vendored your training code too)
        ├── data/
        │   ├── raw/                 # Raw/sample input data
        │   └── processed/           # (Optional) intermediate outputs
        ├── models/                  # Pre-trained model artifacts
        │   └── best_model.joblib
        ├── notebooks/               # Exploration / demo notebooks (+ exported PDFs)
        ├── reports/                 # Generated reports, slides, etc.
        ├── src/                     # Local dev package (editable install)
        │   └── credit_risk/
        │       ├── __init__.py
        │       ├── data_ingest.py
        │       ├── features.py
        │       ├── fairness_audit.py
        │       └── train_model.py
        ├── tests/                   # Unit / integration tests
        ├── .gitignore
        ├── README.md                # ← you are here
        ├── requirements.txt         # Python deps
        ├── runtime.txt              # Python version for Streamlit Cloud
        └── setup.py                 # (Optional) if you want pip-installable package
        ```


---


## Setup & installation

1. **Clone the repository:**
 ```bash
   git clone https://github.com/your-org/credit-risk-project.git
   cd credit-risk-project
 ```
2. **Create & activate virtual environment(recommended):**
 ```bash
   python3 -m venv venv
   source venv/bin/activate
 ```
3. **Install dependencies:**
 ```bash
 pip install --upgrade pip
 pip install -r requirements.txt
 ```

4. *(Optional) Development package install:*
 ```bash
 pip install -e src/credit_risk
 ```
---


## Running locally

From the project root, launch the streamlit app:

```bash
streamlit run dashboard/app.py
```

Navigate to the URL printed in the console (usually http://localhost:8501).

## Deployment on streamlit cloud

1. **Python version:** This app uses Python 3.9 (pinned in runtime.txt)

2. **Dependencies:** Streamlit Cloud installs exactly what's in requirements.txt

3. **Launch Command:** Streamlit Cloud will automatically run dashboard/app.py as the main module.

Just push your changes to the main branch on Git Hub. Streamlit Cloud will:
- Detect runtime.txt -> use Python 3.9
- Install dependencies from requirements.txt
- Locate dashboard/app.py and start the app
---


## App features

- **Data Ingestion & Cleaning:** Upload your own borrower CSV or download a sample template.
- **Feature generation:** Automatic train/test split and preprocessing pipeline.
- **Model Performance:** ROC curve, Precision-Recall, confusion matrix, lift chart
- **Fairness Audit:** Slice-based fairness metrics and Equalized Odds mitigation.
- **Cohort Analysis:** Demographic breakdowns and bar charts
- **Download Results:** Export predictions, cleaned data, and the trained model.
---


## License
This project is licensed under MIT License. Feel free to use and adapt



