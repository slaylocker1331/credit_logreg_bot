# Telegram Bot for Credit Approval Prediction 🏦

A Telegram bot that uses logistic regression to predict the probability of loan approval.

## What is this?

The bot analyzes customer data (age, income, housing type, loan purpose, etc.) and based on a trained machine learning model, predicts whether a credit application will be approved or rejected.

## Requirements 📋

- Python 3.8+
- pip
- Telegram bot token

## Installation 🚀

### 1. Clone/download the repository

```bash
cd /path/to/project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a token for your bot

1. Open Telegram and find **@BotFather**
2. Send the `/newbot` command
3. Follow the instructions
4. Copy your token (looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

### 4. Train the model

First, train the model and save it:

```bash
python train_model.py
```

This script:
- Loads data from `train.csv`
- Trains logistic regression
- Saves the model to `credit_model.pkl`
- Outputs metrics on the validation set

**Expected output:**
```
Validation Results:
==================================================
Accuracy:  0.8995
Precision: 0.7576
Recall:    0.4323
F1-Score:  0.5505
ROC-AUC:   0.8833
==================================================

Model saved to credit_model.pkl
```

### 5. Run the bot

```bash
python bot.py
```

**Before the first run**, create a `.env` file with your token:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your token
# TELEGRAM_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
```

Or set the environment variable directly:

```bash
export TELEGRAM_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
python bot.py
```

## Usage 💬

1. Open Telegram and find your bot
2. Press `/start`
3. Follow the bot's instructions
4. Enter your information
5. Get your prediction! ✨

## Bot Parameters 🎯

The bot collects the following information:

| Parameter | Description | Example |
|-----------|-------------|---------|
| **Age** | Your age in years | 35 |
| **Income** | Annual income in dollars | 50000 |
| **Housing Type** | RENT, OWN, MORTGAGE, OTHER | RENT |
| **Employment Length** | Years at current job | 5 |
| **Loan Purpose** | PERSONAL, EDUCATION, MEDICAL, VENTURE, HOME_IMPROVEMENT, DEBT_CONSOLIDATION | EDUCATION |
| **Loan Grade** | A-G (A best, G worst) | B |
| **Loan Amount** | Requested amount in $ | 10000 |
| **Interest Rate** | Rate in % | 8.5 |

## File Structure 📁

```
playground-series-s4e10/
├── train.csv                    # Training data
├── test.csv                     # Test data
├── sample_submission.csv        # Example results
├── credit_prediction.ipynb      # Jupyter notebook with analysis
├── train_model.py              # Model training script
├── bot.py                      # Telegram bot
├── credit_model.pkl            # Saved model (created after train_model.py)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## How it Works ⚙️

### Model Training (`train_model.py`)

```
train.csv → Data Preprocessing → Normalization → Logistic Regression → credit_model.pkl
```

1. **Preprocessing:**
   - Fill missing values
   - Encode categorical variables

2. **Normalization:**
   - StandardScaler for feature scaling

3. **Training:**
   - Logistic regression with max_iter=1000
   - 80% training / 20% validation split

### Prediction in Bot (`bot.py`)

```
User Input → Data Preparation → Normalization → 
Logistic Regression → Probability → Result
```

## Model Results 📊

On the validation set:
- **Accuracy:** ~90%
- **ROC-AUC:** ~0.88
- **Precision:** ~76% 
- **Recall:** ~43%

## Troubleshooting 🔧

### Error: "credit_model.pkl not found"

**Solution:** Run `python train_model.py` first

### Error: "TOKEN not set"

**Solution:** Replace `TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"` with your actual token

### Bot is not responding

**Solution:** 
- Check your internet connection
- Verify the token is correct
- Restart the bot

### Import error for libraries

**Solution:** Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Additional Commands 📝

- `/start` - Start a new application
- `/help` - Bot help
- `/cancel` - Cancel current application

## Security ⚠️

- **Never commit your `.env` file** to Git (it's in `.gitignore`)
- **Never publish your token** online
- Use the `.env` file to store sensitive data:

```bash
# .env (create this file locally, don't commit)
TELEGRAM_TOKEN=your_token_here
```

The bot will automatically load variables from `.env` using `python-dotenv`:

```python
from dotenv import load_dotenv
import os

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
```

## License 📄

MIT

## Author ✍️

Created as a machine learning project for credit approval prediction.

## Support & Contact 💬

If you have questions or issues, check:
1. Are all dependencies installed?
2. Is the token set correctly?
3. Has the model been trained (`train_model.py`)?

---

**Happy using! 🚀**
