import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

#--------------------------------------------------------------
# src/features.py
# Feature engineering for credit-risk project
#--------------------------------------------------------------


def generate_features(
    df: pd.DataFrame,
    target_col: str = 'default',
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits cleaned DataFrame into train/test sets.
    Returns: x_train, x_test, y_train, y_test   (unscaled)
    """
    # Separate target
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col} not in DataFrame")
    y = df[target_col]
    x = df.drop(columns=[target_col])

    # Attempt stratified split if requested
    split_kwargs = {
        'test_size': test_size,
        'random_state': random_state
    }
    if stratify:
        split_kwargs['stratify'] = y
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            **split_kwargs
        )
    except ValueError as e:
        warnings.warn(
            f"Stratified split failed ({e}); falling back to unstratified split.",
            UserWarning
        )
        # Retry without Stratification
        split_kwargs.pop('stratify', None)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, 
            test_size = test_size,
            random_state=random_state
        )

    print(
        f"Feature generation complete: {len(x_train)} training samples, "
        f"{len(x_test)} test samples"
    )
    if stratify:
        print("Stratification: on")
    else:
        print("Stratification: off")

    return x_train, x_test, y_train, y_test


def build_preprocessor(df: pd.DataFrame):
    """
    Returns a ColumnTransformer that:
        - standardizes any numeric cols
        - one-hot-encodes all the rest
    # can import this in train_model.py and plug into Pipeline.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    numeric_cols     = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.columns.difference(numeric_cols).tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(),       numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
    return preprocessor
