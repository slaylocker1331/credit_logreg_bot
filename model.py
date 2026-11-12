"""Feature definitions and the sklearn pipeline shared by training and the bot."""
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

MODEL_PATH = 'credit_model.pkl'
TARGET = 'loan_status'  # 1 = high risk (default), so approval probability is P(class 0)

HOME_OWNERSHIP = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
LOAN_INTENT = ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
LOAN_GRADE = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
DEFAULT_ON_FILE = ['N', 'Y']

CATEGORICAL = {
    'person_home_ownership': HOME_OWNERSHIP,
    'loan_intent': LOAN_INTENT,
    'loan_grade': LOAN_GRADE,
    'cb_person_default_on_file': DEFAULT_ON_FILE,
}
NUMERIC = [
    'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
    'loan_percent_income', 'cb_person_cred_hist_length',
]
ENGINEERED = ['log_income', 'log_loan_amnt', 'loan_to_income']
RAW_FEATURES = NUMERIC + list(CATEGORICAL)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['log_income'] = np.log1p(df['person_income'])
    df['log_loan_amnt'] = np.log1p(df['loan_amnt'])
    # loan_percent_income in the dataset is rounded to 2 decimals; keep an exact ratio too
    df['loan_to_income'] = df['loan_amnt'] / df['person_income'].clip(lower=1)
    return df


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer([
        ('num', StandardScaler(), NUMERIC + ENGINEERED),
        ('cat', OneHotEncoder(categories=list(CATEGORICAL.values()), drop='if_binary'), list(CATEGORICAL)),
    ])
    return Pipeline([
        ('features', FunctionTransformer(add_features)),
        ('preprocess', preprocess),
        ('clf', LogisticRegression(max_iter=2000, random_state=42)),
    ])


def make_row(answers: dict) -> pd.DataFrame:
    row = dict(answers)
    row['loan_percent_income'] = round(row['loan_amnt'] / row['person_income'], 2)
    return pd.DataFrame([row])[RAW_FEATURES]


def predict_risk(model_data: dict, answers: dict) -> tuple[float, bool]:
    """Returns (probability of default, approved?)."""
    risk = float(model_data['pipeline'].predict_proba(make_row(answers))[0, 1])
    return risk, risk < model_data['threshold']


def save_model(model_data: dict, path: str = MODEL_PATH) -> None:
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)


def load_model(path: str = MODEL_PATH) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)
