import pytest

from model import CATEGORICAL, TARGET, predict_risk
from tests.conftest import RISKY, SAFE


@pytest.mark.parametrize('column', list(CATEGORICAL))
def test_categories_match_data(train_df, column):
    assert set(train_df[column].unique()) == set(CATEGORICAL[column])


def test_target_one_means_high_risk(train_df):
    by_grade = train_df.groupby('loan_grade')[TARGET].mean()
    assert by_grade['G'] > by_grade['A']


def test_predict_risk_orders_profiles(model_data):
    safe_risk, safe_approved = predict_risk(model_data, SAFE)
    risky_risk, risky_approved = predict_risk(model_data, RISKY)
    assert safe_risk < risky_risk
    assert safe_approved and not risky_approved
