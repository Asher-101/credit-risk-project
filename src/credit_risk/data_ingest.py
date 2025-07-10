import pandas as pd
import os

# Load and clean borrower data for credit-risk project

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load loan data from a CSV into a pandas DataFrame.
    Raises FileNotFoundError if the path does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {file_path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the raw borrower data:
      1) Drop duplicates
      2) Fill missing values
      3) Clip/correct outliers
      4) Create derived features and bins
      5) One-hot encode
    """
    # 1) De-duplicate
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {before - len(df)} duplicate rows")

    # 2) Missing values -> median
    for col in ['employment_length', 'age', 'annual_income']:
        if col in df.columns:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            print(f"Filled missing '{col}' with median={med}")
    
    # 3) Outlier clipping
    if 'debt_to_income' in df.columns:
        df['debt_to_income'] = df['debt_to_income'].clip(0,1)
        print(f"Clipped 'debt_to_income' to [0,1]")
    
    # 4) Derived features
    if 'debt_to_income' in df.columns:
        df['dti_pct'] = df['debt_to_income'] * 100

    # 5) Binning & Encoding
    if 'employment_length' in df.columns:
        bins = [0,1,5,10,20,40]
        labels = ['<1', '1-5', '5-10', '10-20', '20+']
        df['emp_bin'] = pd.cut(df['employment_length'], bins=bins, labels=labels, include_lowest=True)
    if 'age' in df.columns:
        a_bins = [18,25,35,45,55,65,100]
        a_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        df['age_bin'] = pd.cut(df['age'], bins=a_bins, labels=a_labels, include_lowest=True)
    
    cat_cols = [c for c in ['emp_bin', 'age_bin'] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        print(f"One-hot encoded columns: {cat_cols}")
    
    print(f"Cleaned data has {len(df)} rows and {len(df.columns)} columns")
    return df


def save_data(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("Ingest & clean credit data")
    p.add_argument("--input", "-i", required=True, help="raw CSV path")
    p.add_argument("--output", "-o", default="data/processed/cleaned.csv", help="cleaned CSV path")
    args = p.parse_args()

    df0 = load_data(args.input)
    df1 = clean_data(df0)
    save_data(df1, args.output)