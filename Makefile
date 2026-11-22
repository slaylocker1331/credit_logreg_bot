# Conda environment to run in (override: make run CONDA_ENV=other, or make run PYTHON=python3)
CONDA_ENV ?= slayer
CONDA_BASE := $(shell conda info --base 2>/dev/null)
ifneq ($(wildcard $(CONDA_BASE)/envs/$(CONDA_ENV)/bin/python),)
PYTHON ?= $(CONDA_BASE)/envs/$(CONDA_ENV)/bin/python
else
PYTHON ?= python3
endif

DB ?= bot_data.db
DB_QUERY = SELECT a.id, a.created_at, u.first_name, u.last_name, u.link, u.phone, \
	a.person_age, a.person_income, a.person_home_ownership, a.person_emp_length, \
	a.cb_person_cred_hist_length, a.cb_person_default_on_file, a.loan_intent, a.loan_grade, \
	a.loan_amnt, a.loan_int_rate, round(a.risk, 3) AS risk, a.approved \
	FROM applications a JOIN users u USING (user_id) ORDER BY a.id DESC

.PHONY: help install install-dev train run test clean db export

help:
	@echo "make install      - install dependencies"
	@echo "make install-dev  - install dependencies + pytest"
	@echo "make train        - retrain the model (credit_model.pkl)"
	@echo "make run          - start the Telegram bot (trains the model first if needed)"
	@echo "make test         - run tests"
	@echo "make db           - show the last 20 applications from the database"
	@echo "make export       - export all applications with user info to users_export.csv"
	@echo "make clean        - remove caches and the trained model"
	@echo ""
	@echo "Python: $(PYTHON)"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev: install
	$(PYTHON) -m pip install -r requirements-dev.txt

# Rebuilt automatically when missing or when the training code/data changes
credit_model.pkl: train_model.py model.py train.csv
	$(PYTHON) train_model.py

train:
	$(PYTHON) train_model.py

run: credit_model.pkl
	@test -f .env || { echo "No .env file. Run: cp .env.example .env and set TELEGRAM_TOKEN"; exit 1; }
	$(PYTHON) bot.py

test:
	$(PYTHON) -m pytest

db:
	@test -f $(DB) || { echo "No $(DB) yet — it is created when the bot starts"; exit 1; }
	@sqlite3 -header -column $(DB) "$(DB_QUERY) LIMIT 20"

export:
	@test -f $(DB) || { echo "No $(DB) yet — it is created when the bot starts"; exit 1; }
	@sqlite3 -header -csv $(DB) "$(DB_QUERY)" > users_export.csv
	@echo "Saved users_export.csv"

clean:
	rm -rf __pycache__ tests/__pycache__ .pytest_cache credit_model.pkl
