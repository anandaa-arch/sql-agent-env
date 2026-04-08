"""
test_local.py — local pre-submission validation

Validates:
  1. openenv.yaml structure
  2. All task schemas load and seed data inserts cleanly
  3. All graders return scores in [0.0, 1.0]
  4. Perfect answers score 1.0 on every task
  5. SQL errors score 0.0
  6. Partial answers score between 0 and 1
  7. Environment class: reset / step / state lifecycle
  8. Hint injection after 3 failed steps
  9. Episode terminates correctly on submit and on max_steps

Run with:
  python test_local.py
"""

import sqlite3
import sys
import traceback

# ── Colour helpers ─────────────────────────────────────────────────────────────

def ok(msg):  print(f"  \033[32m✓\033[0m {msg}")
def fail(msg):print(f"  \033[31m✗\033[0m {msg}"); _FAILURES.append(msg)

_FAILURES = []

# ──────────────────────────────────────────────────────────────────────────────
# 1. openenv.yaml
# ──────────────────────────────────────────────────────────────────────────────

print("\n── 1. openenv.yaml ──")
try:
    import yaml
    with open("openenv.yaml") as f:
        meta = yaml.safe_load(f)
    for field in ("name", "version", "tasks", "observation", "action", "reward"):
        assert field in meta, f"Missing field: {field}"
    assert len(meta["tasks"]) >= 3, "Need at least 3 tasks"
    for t in meta["tasks"]:
        assert "id" in t and "difficulty" in t
    ok("openenv.yaml is valid and has 3+ tasks")
except Exception as e:
    fail(f"openenv.yaml: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 2 + 3. Tasks load and graders are range-safe
# ──────────────────────────────────────────────────────────────────────────────

print("\n── 2+3. Task schemas + grader range safety ──")
try:
    from app.tasks import TASKS
    for tid, task in TASKS.items():
        conn = sqlite3.connect(":memory:")
        conn.executescript(task["schema_sql"])
        # empty result
        s, _ = task["grader"](conn, "SELECT 1 WHERE 1=0")
        assert 0.0 <= s <= 1.0, f"Out of range: {s}"
        # syntax error
        s, _ = task["grader"](conn, "INVALID SQL !!!")
        assert s == 0.0, f"Syntax error should give 0.0, got {s}"
        ok(f"{tid}: schema loads, grader range-safe")
except Exception as e:
    fail(f"Tasks: {e}\n{traceback.format_exc()}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Perfect answers score 1.0
# ──────────────────────────────────────────────────────────────────────────────

print("\n── 4. Perfect answers → 1.0 ──")

PERFECT_ANSWERS = {
    "task_1_easy": """
        SELECT DISTINCT c.name, c.email
        FROM customers c
        JOIN orders o ON c.id = o.customer_id
    """,
    "task_2_medium": """
        WITH recent AS (
            SELECT u.plan, u.id AS uid, COUNT(e.id) AS evts
            FROM users u
            LEFT JOIN events e ON e.user_id = u.id AND e.ts >= '2024-01-01'
            GROUP BY u.plan, u.id
        )
        SELECT plan,
               SUM(evts) AS total_events,
               ROUND(CAST(SUM(evts) AS REAL) / COUNT(uid), 2) AS avg_events_per_user
        FROM recent
        GROUP BY plan
        ORDER BY plan
    """,
    "task_3_hard": """
        WITH running AS (
            SELECT a.holder, t.tx_date,
                   SUM(t.amount) OVER (
                       PARTITION BY t.account_id ORDER BY t.tx_date
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS running_balance, t.amount
            FROM transactions t
            JOIN accounts a ON a.id = t.account_id
        ),
        milestones AS (
            SELECT holder, MIN(tx_date) AS first_milestone_date
            FROM running WHERE running_balance > 5000 GROUP BY holder
        ),
        credits AS (
            SELECT a.holder, SUM(t.amount) AS total_credits
            FROM transactions t
            JOIN accounts a ON a.id = t.account_id
            WHERE t.amount > 0 GROUP BY a.holder
        )
        SELECT c.holder, m.first_milestone_date, c.total_credits
        FROM credits c LEFT JOIN milestones m ON c.holder = m.holder
        ORDER BY c.holder
    """,
}

try:
    from app.tasks import TASKS
    for tid, sql in PERFECT_ANSWERS.items():
        conn = sqlite3.connect(":memory:")
        conn.executescript(TASKS[tid]["schema_sql"])
        s, fb = TASKS[tid]["grader"](conn, sql)
        assert s == 1.0, f"Expected 1.0, got {s}. Feedback: {fb}"
        ok(f"{tid}: perfect answer → 1.0")
except Exception as e:
    fail(f"Perfect answers: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Partial answers give 0 < score < 1
# ──────────────────────────────────────────────────────────────────────────────

print("\n── 5. Partial answers → (0, 1) ──")
try:
    from app.tasks import TASKS
    conn = sqlite3.connect(":memory:")
    conn.executescript(TASKS["task_1_easy"]["schema_sql"])
    # Returns all customers — includes 2 extras
    s, _ = TASKS["task_1_easy"]["grader"](conn, "SELECT name, email FROM customers")
    assert 0.0 < s < 1.0, f"Expected partial, got {s}"
    ok(f"task_1_easy: partial answer → {s:.3f}")

    conn2 = sqlite3.connect(":memory:")
    conn2.executescript(TASKS["task_2_medium"]["schema_sql"])
    # Correct totals but avg wrong (returns total instead of avg) → 0.5 partial
    s2, _ = TASKS["task_2_medium"]["grader"](conn2, """
        WITH recent AS (
            SELECT u.plan, u.id AS uid, COUNT(e.id) AS evts
            FROM users u
            LEFT JOIN events e ON e.user_id = u.id AND e.ts >= '2024-01-01'
            GROUP BY u.plan, u.id
        )
        SELECT plan,
               SUM(evts) AS total_events,
               CAST(SUM(evts) AS REAL) AS avg_events_per_user
        FROM recent GROUP BY plan ORDER BY plan
    """)
    assert 0.0 < s2 < 1.0, f"Expected partial (wrong avg), got {s2}"
    ok(f"task_2_medium: partial answer (correct total, wrong avg) → {s2:.3f}")
except Exception as e:
    fail(f"Partial answers: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 6+7. Environment lifecycle
# ──────────────────────────────────────────────────────────────────────────────

print("\n── 6+7. Environment lifecycle ──")
try:
    from app.environment import SQLEnvironment
    from app.models import SQLAction

    env = SQLEnvironment("task_1_easy")
    obs = env.reset()
    assert obs.task_id == "task_1_easy"
    assert obs.steps_taken == 0
    assert not obs.done
    ok("reset() returns clean observation")

    obs2, r, done, info = env.step(SQLAction(mode="sql", query="SELECT * FROM customers LIMIT 3"))
    assert obs2.steps_taken == 1
    assert obs2.last_result is not None
    assert obs2.last_result.row_count == 3
    assert r.score >= 0.0
    assert not done
    ok(f"step(sql) works, result has 3 rows, reward={r.score}")

    st = env.state()
    assert st.steps_taken == 1
    assert len(st.query_history) == 1
    ok("state() returns correct step count and history")

    obs3, r3, done3, info3 = env.step(SQLAction(
        mode="submit",
        query="SELECT DISTINCT c.name, c.email FROM customers c JOIN orders o ON c.id = o.customer_id"
    ))
    assert done3, "Episode should be done after submit"
    assert r3.score == 1.0, f"Expected 1.0, got {r3.score}"
    assert r3.is_final
    ok(f"submit() → score=1.0, done=True, is_final=True")

    # Step after done should be no-op
    obs4, r4, done4, _ = env.step(SQLAction(mode="sql", query="SELECT 1"))
    assert done4
    assert r4.score == 0.0
    ok("step() after done is a no-op")
except Exception as e:
    fail(f"Environment lifecycle: {e}\n{traceback.format_exc()}")

# ──────────────────────────────────────────────────────────────────────────────
# 8. Hint injection
# ──────────────────────────────────────────────────────────────────────────────

print("\n── 8. Hint injection after 3 failures ──")
try:
    from app.environment import SQLEnvironment
    from app.models import SQLAction

    env = SQLEnvironment("task_3_hard")
    env.reset()
    hint_seen = False
    for i in range(5):
        obs, r, done, _ = env.step(SQLAction(mode="sql", query="SELECT 1"))
        if obs.hint:
            hint_seen = True
            ok(f"Hint appeared at step {i+1}: '{obs.hint[:60]}…'")
            break
    assert hint_seen, "Hint never appeared after 5 wrong steps"
except Exception as e:
    fail(f"Hint injection: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 9. Max steps exhaustion
# ──────────────────────────────────────────────────────────────────────────────

print("\n── 9. Episode terminates at max_steps ──")
try:
    from app.environment import SQLEnvironment
    from app.models import SQLAction

    env = SQLEnvironment("task_1_easy")
    env.reset()
    done = False
    for i in range(env.max_steps + 2):
        _, _, done, _ = env.step(SQLAction(mode="sql", query="SELECT 1"))
        if done:
            ok(f"Episode terminated at step {i+1} (max={env.max_steps})")
            break
    assert done, "Episode should have terminated"
except Exception as e:
    fail(f"Max steps: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

print()
if _FAILURES:
    print(f"\033[31m✗ {len(_FAILURES)} check(s) FAILED:\033[0m")
    for f in _FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("\033[32m✓ All checks passed — submission is valid!\033[0m")
