"""Local SQLite database of bot users and their applications."""
from __future__ import annotations

import sqlite3
from datetime import datetime

DB_PATH = 'bot_data.db'

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id       INTEGER PRIMARY KEY,
    username      TEXT,
    first_name    TEXT,
    last_name     TEXT,
    link          TEXT,
    phone         TEXT,
    language_code TEXT,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS applications (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                     INTEGER NOT NULL REFERENCES users(user_id),
    created_at                  TEXT NOT NULL,
    person_age                  INTEGER,
    person_income               INTEGER,
    person_home_ownership       TEXT,
    person_emp_length           REAL,
    cb_person_cred_hist_length  INTEGER,
    cb_person_default_on_file   TEXT,
    loan_intent                 TEXT,
    loan_grade                  TEXT,
    loan_amnt                   INTEGER,
    loan_int_rate               REAL,
    loan_percent_income         REAL,
    risk                        REAL NOT NULL,
    approved                    INTEGER NOT NULL,
    threshold                   REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_applications_user ON applications(user_id);
"""

ANSWER_FIELDS = [
    'person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
    'cb_person_cred_hist_length', 'cb_person_default_on_file', 'loan_intent', 'loan_grade',
    'loan_amnt', 'loan_int_rate',
]


def now() -> str:
    return datetime.now().isoformat(timespec='seconds')


def user_link(user_id: int, username: str | None) -> str:
    return f'https://t.me/{username}' if username else f'tg://user?id={user_id}'


class Database:
    def __init__(self, path: str = DB_PATH):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def upsert_user(self, user) -> None:
        """Saves or refreshes a telegram.User (name and username can change)."""
        ts = now()
        self.conn.execute(
            """INSERT INTO users (user_id, username, first_name, last_name, link, language_code, first_seen, last_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   username = excluded.username, first_name = excluded.first_name,
                   last_name = excluded.last_name, link = excluded.link,
                   language_code = excluded.language_code, last_seen = excluded.last_seen""",
            (user.id, user.username, user.first_name, user.last_name, user_link(user.id, user.username),
             user.language_code, ts, ts),
        )
        self.conn.commit()

    def save_phone(self, user, phone: str) -> None:
        self.upsert_user(user)
        self.conn.execute('UPDATE users SET phone = ? WHERE user_id = ?', (phone, user.id))
        self.conn.commit()

    def save_application(self, user, answers: dict, risk: float, approved: bool, threshold: float) -> int:
        self.upsert_user(user)
        values = [answers.get(f) for f in ANSWER_FIELDS]
        ratio = round(answers['loan_amnt'] / answers['person_income'], 4)
        cur = self.conn.execute(
            f"""INSERT INTO applications (user_id, created_at, {', '.join(ANSWER_FIELDS)},
                                          loan_percent_income, risk, approved, threshold)
                VALUES ({', '.join('?' * (len(ANSWER_FIELDS) + 6))})""",
            [user.id, now(), *values, ratio, risk, int(approved), threshold],
        )
        self.conn.commit()
        return cur.lastrowid

    def get_user(self, user_id: int) -> sqlite3.Row | None:
        return self.conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()

    def get_applications(self, user_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            'SELECT * FROM applications WHERE user_id = ? ORDER BY id', (user_id,)
        ).fetchall()
