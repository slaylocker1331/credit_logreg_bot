import pandas as pd
import pytest

from model import RAW_FEATURES, TARGET, build_pipeline
from train_model import best_threshold

SAFE = {
    'person_age': 35, 'person_income': 90_000, 'person_home_ownership': 'MORTGAGE',
    'person_emp_length': 8, 'loan_intent': 'EDUCATION', 'loan_grade': 'A', 'loan_amnt': 5_000,
    'loan_int_rate': 7.5, 'cb_person_default_on_file': 'N', 'cb_person_cred_hist_length': 10,
}
RISKY = SAFE | {
    'person_income': 20_000, 'person_home_ownership': 'RENT', 'person_emp_length': 0,
    'loan_grade': 'G', 'loan_amnt': 15_000, 'loan_int_rate': 21.0, 'cb_person_default_on_file': 'Y',
    'cb_person_cred_hist_length': 2,
}


@pytest.fixture(scope='session')
def train_df():
    return pd.read_csv('train.csv')


@pytest.fixture(scope='session')
def model_data(train_df):
    pipeline = build_pipeline().fit(train_df[RAW_FEATURES], train_df[TARGET])
    proba = pipeline.predict_proba(train_df[RAW_FEATURES])[:, 1]
    return {'pipeline': pipeline, 'threshold': best_threshold(train_df[TARGET], proba)}
