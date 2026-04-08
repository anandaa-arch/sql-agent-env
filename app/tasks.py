"""
Three tasks (easy → medium → hard) each with:
  - Schema + seed SQL
  - Task description
  - A deterministic grader function
"""
from typing import Any, Dict, List, Optional, Tuple
import sqlite3


# ──────────────────────────────────────────────────────────────────────────────
# Helper: normalise a result-set for comparison
# ──────────────────────────────────────────────────────────────────────────────

def _normalise(rows: List[tuple]) -> List[tuple]:
    """Sort and lower-case strings so comparison is order-insensitive."""
    def _cell(v: Any):
        if isinstance(v, str):
            return v.strip().lower()
        if isinstance(v, float):
            return round(v, 2)
        return v
    return sorted([tuple(_cell(c) for c in r) for r in rows])


def _run_query(conn: sqlite3.Connection, sql: str) -> Tuple[Optional[List[tuple]], Optional[str]]:
    try:
        cur = conn.execute(sql)
        return cur.fetchall(), None
    except Exception as e:
        return None, str(e)


# ──────────────────────────────────────────────────────────────────────────────
# TASK 1 – EASY
# Domain: e-commerce customers & orders
# Objective: Return full names + emails of customers who have placed ≥1 order
# ──────────────────────────────────────────────────────────────────────────────

TASK1_SCHEMA_SQL = """
CREATE TABLE customers (
    id       INTEGER PRIMARY KEY,
    name     TEXT NOT NULL,
    email    TEXT NOT NULL,
    city     TEXT,
    joined   TEXT    -- ISO date string
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    amount      REAL,
    status      TEXT,   -- 'completed', 'pending', 'cancelled'
    order_date  TEXT
);

INSERT INTO customers VALUES
 (1, 'Alice Sharma',   'alice@example.com',   'Mumbai',    '2022-01-15'),
 (2, 'Bob Mehta',      'bob@example.com',     'Delhi',     '2022-03-10'),
 (3, 'Carla D Cruz',   'carla@example.com',   'Bangalore', '2023-06-01'),
 (4, 'David Singh',    'david@example.com',   'Pune',      '2023-08-20'),
 (5, 'Eva Pillai',     'eva@example.com',     'Chennai',   '2024-01-05');

INSERT INTO orders VALUES
 (1,  1, 1200.00, 'completed', '2023-02-01'),
 (2,  1,  350.50, 'completed', '2023-05-10'),
 (3,  2,  899.99, 'pending',   '2023-07-22'),
 (4,  3,  150.00, 'cancelled', '2023-09-11'),
 (5,  3,  450.00, 'completed', '2024-01-15'),
 (6,  3,  275.00, 'completed', '2024-03-01');
-- David (4) and Eva (5) have NO orders
"""

TASK1_DESCRIPTION = """
## Task: Customers With Orders

You have two tables:
- **customers** (id, name, email, city, joined)
- **orders** (id, customer_id, amount, status, order_date)

**Goal:** Return the `name` and `email` of every customer who has placed **at least one order** (any status counts).

**Expected output columns:** `name`, `email`  
**Hint:** Some customers have never ordered — exclude them.
"""

TASK1_EXPECTED_SQL = """
SELECT DISTINCT c.name, c.email
FROM customers c
JOIN orders o ON c.id = o.customer_id
ORDER BY c.name;
"""

TASK1_SCHEMA_INFO = {
    "customers": [
        "id INTEGER PRIMARY KEY",
        "name TEXT – full name",
        "email TEXT",
        "city TEXT",
        "joined TEXT – ISO date e.g. '2022-01-15'"
    ],
    "orders": [
        "id INTEGER PRIMARY KEY",
        "customer_id INTEGER – FK → customers.id",
        "amount REAL – order value in USD",
        "status TEXT – 'completed' | 'pending' | 'cancelled'",
        "order_date TEXT – ISO date"
    ]
}

TASK1_SAMPLE_DATA = {
    "customers": [
        {"id": 1, "name": "Alice Sharma",  "email": "alice@example.com", "city": "Mumbai",    "joined": "2022-01-15"},
        {"id": 2, "name": "Bob Mehta",     "email": "bob@example.com",   "city": "Delhi",     "joined": "2022-03-10"},
        {"id": 3, "name": "Carla D Cruz",  "email": "carla@example.com", "city": "Bangalore", "joined": "2023-06-01"},
    ],
    "orders": [
        {"id": 1, "customer_id": 1, "amount": 1200.00, "status": "completed", "order_date": "2023-02-01"},
        {"id": 2, "customer_id": 1, "amount": 350.50,  "status": "completed", "order_date": "2023-05-10"},
        {"id": 3, "customer_id": 2, "amount": 899.99,  "status": "pending",   "order_date": "2023-07-22"},
    ]
}


def grade_task1(conn: sqlite3.Connection, submitted_sql: str) -> Tuple[float, str]:
    """Grade the submitted query for task 1."""
    expected_rows, _ = _run_query(conn, TASK1_EXPECTED_SQL)
    agent_rows, err = _run_query(conn, submitted_sql)

    if err:
        return 0.0, f"Query error: {err}"
    if agent_rows is None:
        return 0.0, "No result returned."

    expected_norm = _normalise(expected_rows)
    agent_norm    = _normalise(agent_rows)

    if agent_norm == expected_norm:
        return 1.0, "Perfect! Correct customers returned."

    # Partial credit
    expected_set = set(expected_norm)
    agent_set    = set(agent_norm)
    correct_hits = len(expected_set & agent_set)
    precision    = correct_hits / len(agent_set)  if agent_set    else 0.0
    recall       = correct_hits / len(expected_set) if expected_set else 0.0

    if precision + recall == 0:
        return 0.0, "No overlap with expected results."

    f1 = 2 * precision * recall / (precision + recall)
    score = round(f1 * 0.8, 3)   # cap partial at 0.8 to incentivise full solve

    missing = expected_set - agent_set
    extra   = agent_set - expected_set
    parts = []
    if missing:
        parts.append(f"Missing {len(missing)} row(s).")
    if extra:
        parts.append(f"Extra {len(extra)} row(s) not expected.")
    return score, " ".join(parts) or f"F1={f1:.2f}"


# ──────────────────────────────────────────────────────────────────────────────
# TASK 2 – MEDIUM
# Domain: SaaS product usage (events + users + plans)
# Objective: Per plan, compute total events last 90 days & avg per user
# ──────────────────────────────────────────────────────────────────────────────

TASK2_SCHEMA_SQL = """
CREATE TABLE users (
    id       INTEGER PRIMARY KEY,
    username TEXT NOT NULL,
    plan     TEXT NOT NULL,   -- 'free', 'pro', 'enterprise'
    created  TEXT
);

CREATE TABLE events (
    id        INTEGER PRIMARY KEY,
    user_id   INTEGER REFERENCES users(id),
    event     TEXT,           -- 'login', 'export', 'share', 'api_call'
    ts        TEXT            -- ISO datetime
);

INSERT INTO users VALUES
 (1, 'alpha',   'free',       '2023-01-01'),
 (2, 'beta',    'pro',        '2023-02-01'),
 (3, 'gamma',   'pro',        '2023-03-01'),
 (4, 'delta',   'enterprise', '2023-04-01'),
 (5, 'epsilon', 'enterprise', '2023-05-01'),
 (6, 'zeta',    'free',       '2023-06-01');

-- Reference date: 2024-04-01 (so "last 90 days" = >= 2024-01-01)
INSERT INTO events VALUES
 (1,  1, 'login',    '2024-01-05 10:00:00'),
 (2,  1, 'export',   '2024-01-06 11:00:00'),
 (3,  2, 'login',    '2024-01-10 09:00:00'),
 (4,  2, 'api_call', '2024-02-01 08:00:00'),
 (5,  2, 'api_call', '2024-02-15 08:30:00'),
 (6,  3, 'share',    '2024-01-20 14:00:00'),
 (7,  4, 'login',    '2024-01-25 16:00:00'),
 (8,  4, 'api_call', '2024-02-10 10:00:00'),
 (9,  4, 'api_call', '2024-03-01 11:00:00'),
 (10, 4, 'export',   '2024-03-15 12:00:00'),
 (11, 5, 'login',    '2024-02-20 09:00:00'),
 (12, 5, 'api_call', '2024-03-10 10:00:00'),
 (13, 6, 'login',    '2023-12-01 10:00:00'),  -- BEFORE 90 days window
 (14, 6, 'share',    '2023-11-15 14:00:00');  -- BEFORE 90 days window
"""

TASK2_DESCRIPTION = """
## Task: Plan-level Usage Report

You have two tables:
- **users** (id, username, plan, created)
- **events** (id, user_id, event, ts)

**Goal:** For each **plan** ('free', 'pro', 'enterprise'), compute:
1. `total_events` – count of events with `ts >= '2024-01-01'`
2. `avg_events_per_user` – total_events divided by number of users on that plan (round to 2 decimal places)

**Expected output columns:** `plan`, `total_events`, `avg_events_per_user`  
**Order by:** `plan` ascending  
**Note:** Users with zero recent events should still count in the denominator for avg.
"""

TASK2_EXPECTED_SQL = """
WITH recent AS (
    SELECT u.plan, u.id AS uid, COUNT(e.id) AS evts
    FROM users u
    LEFT JOIN events e ON e.user_id = u.id AND e.ts >= '2024-01-01'
    GROUP BY u.plan, u.id
)
SELECT
    plan,
    SUM(evts) AS total_events,
    ROUND(CAST(SUM(evts) AS REAL) / COUNT(uid), 2) AS avg_events_per_user
FROM recent
GROUP BY plan
ORDER BY plan;
"""

TASK2_SCHEMA_INFO = {
    "users": [
        "id INTEGER PRIMARY KEY",
        "username TEXT",
        "plan TEXT – 'free' | 'pro' | 'enterprise'",
        "created TEXT – ISO date"
    ],
    "events": [
        "id INTEGER PRIMARY KEY",
        "user_id INTEGER – FK → users.id",
        "event TEXT – 'login' | 'export' | 'share' | 'api_call'",
        "ts TEXT – ISO datetime e.g. '2024-01-05 10:00:00'"
    ]
}

TASK2_SAMPLE_DATA = {
    "users": [
        {"id": 1, "username": "alpha",   "plan": "free",       "created": "2023-01-01"},
        {"id": 2, "username": "beta",    "plan": "pro",        "created": "2023-02-01"},
        {"id": 4, "username": "delta",   "plan": "enterprise", "created": "2023-04-01"},
    ],
    "events": [
        {"id": 1,  "user_id": 1, "event": "login",    "ts": "2024-01-05 10:00:00"},
        {"id": 3,  "user_id": 2, "event": "login",    "ts": "2024-01-10 09:00:00"},
        {"id": 7,  "user_id": 4, "event": "login",    "ts": "2024-01-25 16:00:00"},
    ]
}


def grade_task2(conn: sqlite3.Connection, submitted_sql: str) -> Tuple[float, str]:
    """Grade task 2: aggregation correctness."""
    expected_rows, _ = _run_query(conn, TASK2_EXPECTED_SQL)
    agent_rows, err = _run_query(conn, submitted_sql)

    if err:
        return 0.0, f"Query error: {err}"
    if not agent_rows:
        return 0.0, "Empty result."

    # Build comparable dicts
    def to_dict(rows):
        return {str(r[0]).lower(): (int(r[1]), float(r[2])) for r in rows if len(r) >= 3}

    exp = to_dict(expected_rows)
    got = to_dict(agent_rows)

    if not got:
        return 0.0, "Could not parse result rows."

    correct_plans = 0
    partial_plans = 0
    feedback_parts = []

    for plan, (exp_total, exp_avg) in exp.items():
        if plan not in got:
            feedback_parts.append(f"Missing plan '{plan}'.")
            continue
        got_total, got_avg = got[plan]
        total_ok = got_total == exp_total
        avg_ok   = abs(got_avg - exp_avg) <= 0.05

        if total_ok and avg_ok:
            correct_plans += 1
        elif total_ok or avg_ok:
            partial_plans += 1
            feedback_parts.append(f"Plan '{plan}': {'total OK' if total_ok else 'total wrong'}, {'avg OK' if avg_ok else f'avg expected {exp_avg} got {got_avg}'}.")
        else:
            feedback_parts.append(f"Plan '{plan}': both total and avg wrong (expected total={exp_total}, avg={exp_avg}).")

    n = len(exp)
    score = (correct_plans + 0.5 * partial_plans) / n
    score = round(score, 3)

    if score == 1.0:
        return 1.0, "Perfect aggregation across all plans!"
    return score, " ".join(feedback_parts) if feedback_parts else f"Score: {score}"


# ──────────────────────────────────────────────────────────────────────────────
# TASK 3 – HARD
# Domain: financial transactions with running balances
# Objective: Per account, find the day their balance first exceeded a threshold
#            AND compute 7-day rolling average spend using window functions
# ──────────────────────────────────────────────────────────────────────────────

TASK3_SCHEMA_SQL = """
CREATE TABLE accounts (
    id       INTEGER PRIMARY KEY,
    holder   TEXT NOT NULL,
    currency TEXT DEFAULT 'USD'
);

CREATE TABLE transactions (
    id         INTEGER PRIMARY KEY,
    account_id INTEGER REFERENCES accounts(id),
    amount     REAL,     -- positive = credit, negative = debit
    category   TEXT,     -- 'salary', 'rent', 'food', 'transfer', 'misc'
    tx_date    TEXT      -- ISO date
);

INSERT INTO accounts VALUES
 (1, 'Alice',   'USD'),
 (2, 'Bob',     'USD'),
 (3, 'Charlie', 'USD');

INSERT INTO transactions VALUES
 -- Alice: running balance crosses 5000 on 2024-01-15
 (1,  1,  3000.0, 'salary',   '2024-01-01'),
 (2,  1,  -800.0, 'rent',     '2024-01-05'),
 (3,  1,  -200.0, 'food',     '2024-01-10'),
 (4,  1,  3100.0, 'salary',   '2024-01-15'),  -- cumsum=5100, crosses 5000
 (5,  1,  -150.0, 'misc',     '2024-01-20'),
 -- Bob: running balance crosses 5000 on 2024-01-20
 (6,  2,  2000.0, 'salary',   '2024-01-03'),
 (7,  2, -1500.0, 'rent',     '2024-01-07'),
 (8,  2,  4600.0, 'transfer', '2024-01-20'),  -- cumsum=5100, crosses 5000
 -- Charlie: never crosses 5000
 (9,  3,  1000.0, 'salary',   '2024-01-02'),
 (10, 3,  -500.0, 'food',     '2024-01-08'),
 (11, 3,  1200.0, 'salary',   '2024-01-16'),
 (12, 3,  -300.0, 'misc',     '2024-01-22');
"""

TASK3_DESCRIPTION = """
## Task: Account Balance Milestones (Window Functions)

You have:
- **accounts** (id, holder, currency)
- **transactions** (id, account_id, amount, category, tx_date)

**Goal:** Using window functions, produce a result with **one row per account** containing:
1. `holder` – account holder name
2. `first_milestone_date` – the earliest `tx_date` on which the **running cumulative sum** of `amount` (ordered by `tx_date`) **first exceeded 5000**. NULL if it never exceeded 5000.
3. `total_credits` – sum of all positive `amount` values for the account

**Expected output columns:** `holder`, `first_milestone_date`, `total_credits`  
**Order by:** `holder` ascending

This requires a CTE or subquery with `SUM(amount) OVER (PARTITION BY account_id ORDER BY tx_date)`.
"""

TASK3_EXPECTED_SQL = """
WITH running AS (
    SELECT
        a.holder,
        t.tx_date,
        SUM(t.amount) OVER (
            PARTITION BY t.account_id
            ORDER BY t.tx_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_balance,
        t.amount
    FROM transactions t
    JOIN accounts a ON a.id = t.account_id
),
milestones AS (
    SELECT holder, MIN(tx_date) AS first_milestone_date
    FROM running
    WHERE running_balance > 5000
    GROUP BY holder
),
credits AS (
    SELECT a.holder, SUM(t.amount) AS total_credits
    FROM transactions t
    JOIN accounts a ON a.id = t.account_id
    WHERE t.amount > 0
    GROUP BY a.holder
)
SELECT
    c.holder,
    m.first_milestone_date,
    c.total_credits
FROM credits c
LEFT JOIN milestones m ON c.holder = m.holder
ORDER BY c.holder;
"""

TASK3_SCHEMA_INFO = {
    "accounts": [
        "id INTEGER PRIMARY KEY",
        "holder TEXT – account holder name",
        "currency TEXT – e.g. 'USD'"
    ],
    "transactions": [
        "id INTEGER PRIMARY KEY",
        "account_id INTEGER – FK → accounts.id",
        "amount REAL – positive=credit, negative=debit",
        "category TEXT – 'salary'|'rent'|'food'|'transfer'|'misc'",
        "tx_date TEXT – ISO date e.g. '2024-01-01'"
    ]
}

TASK3_SAMPLE_DATA = {
    "accounts": [
        {"id": 1, "holder": "Alice",   "currency": "USD"},
        {"id": 2, "holder": "Bob",     "currency": "USD"},
        {"id": 3, "holder": "Charlie", "currency": "USD"},
    ],
    "transactions": [
        {"id": 1, "account_id": 1, "amount": 3000.0,  "category": "salary",   "tx_date": "2024-01-01"},
        {"id": 2, "account_id": 1, "amount": -800.0,  "category": "rent",     "tx_date": "2024-01-05"},
        {"id": 6, "account_id": 2, "amount": 2000.0,  "category": "salary",   "tx_date": "2024-01-03"},
    ]
}


def grade_task3(conn: sqlite3.Connection, submitted_sql: str) -> Tuple[float, str]:
    """Grade task 3: window functions & milestones."""
    expected_rows, _ = _run_query(conn, TASK3_EXPECTED_SQL)
    agent_rows, err  = _run_query(conn, submitted_sql)

    if err:
        return 0.0, f"Query error: {err}"
    if not agent_rows:
        return 0.0, "Empty result."

    # expected: {holder: (first_milestone_date|None, total_credits)}
    def parse(rows):
        result = {}
        for r in rows:
            if len(r) < 3:
                continue
            holder   = str(r[0]).strip().lower()
            date_val = str(r[1]).strip() if r[1] is not None else None
            try:
                credits = float(r[2]) if r[2] is not None else 0.0
            except (TypeError, ValueError):
                return {}
            result[holder] = (date_val, credits)
        return result

    exp = parse(expected_rows)
    got = parse(agent_rows)

    if not got:
        return 0.0, "Could not parse output."

    scores = []
    feedback_parts = []
    for holder, (exp_date, exp_credits) in exp.items():
        if holder not in got:
            scores.append(0.0)
            feedback_parts.append(f"Missing holder '{holder}'.")
            continue
        got_date, got_credits = got[holder]
        date_ok    = got_date == exp_date
        credits_ok = abs(got_credits - exp_credits) < 0.01

        row_score = (0.5 if date_ok else 0.0) + (0.5 if credits_ok else 0.0)
        scores.append(row_score)
        if not date_ok:
            feedback_parts.append(f"{holder}: milestone date expected={exp_date} got={got_date}.")
        if not credits_ok:
            feedback_parts.append(f"{holder}: total_credits expected={exp_credits} got={got_credits}.")

    score = round(sum(scores) / len(exp), 3) if exp else 0.0

    if score == 1.0:
        return 1.0, "Perfect! Correct milestones and credits for all accounts."
    return score, " ".join(feedback_parts) if feedback_parts else f"Partial score: {score}"


# ──────────────────────────────────────────────────────────────────────────────
# Task registry
# ──────────────────────────────────────────────────────────────────────────────

TASKS = {
    "task_1_easy": {
        "id": "task_1_easy",
        "difficulty": "easy",
        "description": TASK1_DESCRIPTION,
        "schema_sql": TASK1_SCHEMA_SQL,
        "schema_info": TASK1_SCHEMA_INFO,
        "sample_data": TASK1_SAMPLE_DATA,
        "grader": grade_task1,
        "hint": "Try a JOIN between customers and orders, then use DISTINCT to avoid duplicates.",
        "max_steps": 10,
    },
    "task_2_medium": {
        "id": "task_2_medium",
        "difficulty": "medium",
        "description": TASK2_DESCRIPTION,
        "schema_sql": TASK2_SCHEMA_SQL,
        "schema_info": TASK2_SCHEMA_INFO,
        "sample_data": TASK2_SAMPLE_DATA,
        "grader": grade_task2,
        "hint": "Use a LEFT JOIN so users with no recent events still appear. Then GROUP BY plan.",
        "max_steps": 12,
    },
    "task_3_hard": {
        "id": "task_3_hard",
        "difficulty": "hard",
        "description": TASK3_DESCRIPTION,
        "schema_sql": TASK3_SCHEMA_SQL,
        "schema_info": TASK3_SCHEMA_INFO,
        "sample_data": TASK3_SAMPLE_DATA,
        "grader": grade_task3,
        "hint": "Use SUM(...) OVER (PARTITION BY account_id ORDER BY tx_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) inside a CTE.",
        "max_steps": 15,
    },
}
