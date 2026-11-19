# Telegram Bot for Credit Approval Prediction 🏦

A Telegram bot (Russian-language interface) that uses logistic regression to estimate the default risk of a loan application and predict whether it will be approved.

## What is this?

The bot asks the client 10 questions (age, income, housing, credit history, loan purpose, etc.), runs the answers through a trained scikit-learn pipeline and returns the estimated **risk of default**. An application is considered approved when the risk is below the model's decision threshold.

> In the dataset `loan_status = 1` means **high risk / default** (its rate grows from 5% for grade A to 82% for grade G), so the approval decision is `P(loan_status = 1) < threshold`.

## Requirements 📋

- Python 3.9+
- Telegram bot token

## Installation 🚀

```bash
pip install -r requirements.txt
```

### 1. Train the model

```bash
python train_model.py
```

This script loads `train.csv`, trains the pipeline on an 80/20 split, picks the decision threshold that maximizes F1 on the validation set, refits on all data and saves everything to `credit_model.pkl`.

**Expected output:**
```
Validation Results (threshold = 0.34):
==================================================
accuracy   0.9135
precision  0.6966
recall     0.6958
f1         0.6962
roc_auc    0.9102
==================================================

Model saved to credit_model.pkl
```

### 2. Set the token

Get a token from **@BotFather** (`/newbot`), then:

```bash
cp .env.example .env
# edit .env: TELEGRAM_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
```

or `export TELEGRAM_TOKEN="..."`.

### 3. Run the bot

```bash
python bot.py
```

Open your bot in Telegram and press `/start`.

## Bot Parameters 🎯

| Parameter | Model feature | Input |
|-----------|---------------|-------|
| Age | `person_age` | number, 18–100 |
| Annual income, $ | `person_income` | number (`50000`, `50 000`, `50,000` are all accepted) |
| Housing | `person_home_ownership` | button: RENT / OWN / MORTGAGE / OTHER |
| Years at current job | `person_emp_length` | number, at most age − 14 |
| Credit history length, years | `cb_person_cred_hist_length` | number, at most age − 16 |
| Previous default | `cb_person_default_on_file` | button: Y / N |
| Loan purpose | `loan_intent` | button |
| Loan grade | `loan_grade` | button: A–G |
| Loan amount, $ | `loan_amnt` | number |
| Interest rate, % | `loan_int_rate` | number (`8.5` or `8,5`) |

`loan_percent_income` is computed from amount and income.

## File Structure 📁

```
├── model.py                  # Features, categories and the sklearn pipeline (shared code)
├── train_model.py            # Training, threshold selection, saving credit_model.pkl
├── bot.py                    # Telegram bot
├── credit_prediction.ipynb   # Analysis notebook + Kaggle submission
├── tests/                    # pytest tests for the model and the bot dialog
├── train.csv / test.csv      # Data (Kaggle Playground Series S4E10)
├── requirements.txt
└── requirements-dev.txt
```

## How it Works ⚙️

```
raw answers → add_features → OneHotEncoder + StandardScaler → LogisticRegression → P(default) → threshold
```

- **Feature engineering:** `log(income)`, `log(loan amount)`, exact loan-to-income ratio.
- **Encoding:** one-hot for housing, purpose and grade (the risk is not linear across grades: D is much riskier than C); binary for previous default.
- **One pipeline object** is saved, so the bot passes raw values (`'RENT'`, `'Y'`) and encoding can never drift between training and inference.
- **Threshold** is tuned on validation (default 0.5 gave recall of only ~46%).

## Model Results 📊

| | Before | Now |
|---|---|---|
| ROC-AUC | 0.883 | **0.910** |
| F1 | 0.551 | **0.696** |
| Recall (high-risk class) | 0.432 | **0.696** |
| Precision | 0.758 | 0.697 |

## Tests 🧪

```bash
pip install -r requirements-dev.txt
pytest
```

Tests check that the categories match the data, that predictions are sane for low/high-risk profiles, number parsing, and walk through the full bot dialog.

## Troubleshooting 🔧

- **`credit_model.pkl` not found** — run `python train_model.py`.
- **`credit_model.pkl` is from an old version / fails to load** — the bot refuses to start; retrain with `python train_model.py`.
- **`TELEGRAM_TOKEN` not set** — create `.env` from `.env.example` or export the variable.
- **Errors loading the model after upgrading scikit-learn** — retrain: pickled models are tied to the library version.

## Security ⚠️

Never commit `.env` (it is in `.gitignore`) and never publish your token.

## License 📄

MIT
