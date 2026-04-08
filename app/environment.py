"""
Environment engine: manages per-session SQLite databases and episode state.
"""
import sqlite3
import uuid
from typing import Dict, Optional, Tuple

from app.models import (
    SQLAction, SQLObservation, SQLReward, SQLState, QueryResult
)
from app.tasks import TASKS


class SQLEnvironment:
    """
    One instance per HTTP session (identified by session_id).
    Holds an in-memory SQLite connection and all episode state.
    """

    def __init__(self, task_id: str):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASKS)}")
        self.task_id     = task_id
        self.task        = TASKS[task_id]
        self.session_id  = str(uuid.uuid4())
        self.conn: Optional[sqlite3.Connection] = None
        self._init_db()

        self.steps_taken        = 0
        self.max_steps          = self.task["max_steps"]
        self.query_history      = []
        self.best_score         = 0.0
        self.done               = False
        self.submitted          = False
        self.last_query: Optional[str]        = None
        self.last_result: Optional[QueryResult] = None
        self.failed_attempts    = 0

    # ── Database ──────────────────────────────────────────────────────────────

    def _init_db(self):
        """Create a fresh in-memory SQLite DB and run the task schema SQL."""
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.executescript(self.task["schema_sql"])
        self.conn.commit()

    # ── API ───────────────────────────────────────────────────────────────────

    def reset(self) -> SQLObservation:
        """Re-initialize the episode. Returns the initial observation."""
        self._init_db()
        self.steps_taken   = 0
        self.query_history = []
        self.best_score    = 0.0
        self.done          = False
        self.submitted     = False
        self.last_query    = None
        self.last_result   = None
        self.failed_attempts = 0
        return self._build_observation(hint=None)

    def step(self, action: SQLAction) -> Tuple[SQLObservation, SQLReward, bool, dict]:
        """
        Process one agent action.
        Returns (observation, reward, done, info).
        """
        if self.done:
            obs = self._build_observation()
            reward = SQLReward(score=0.0, feedback="Episode already done.", is_final=False)
            return obs, reward, True, {"message": "Episode already finished."}

        self.steps_taken += 1
        self.query_history.append(action.query)

        # ── Mode: submit ──────────────────────────────────────────────────────
        if action.mode == "submit":
            score, feedback = self.task["grader"](self.conn, action.query)
            self.best_score = max(self.best_score, score)
            self.done       = True
            self.submitted  = True
            self.last_query = action.query

            # Run the query to show result in final obs
            self.last_result = self._execute(action.query)

            reward = SQLReward(
                score=score,
                breakdown={"final_grade": score},
                feedback=feedback,
                is_final=True,
            )
            obs = self._build_observation()
            return obs, reward, True, {"final_score": score, "feedback": feedback}

        # ── Mode: sql (exploration) ───────────────────────────────────────────
        result = self._execute(action.query)
        self.last_query  = action.query
        self.last_result = result

        # Soft reward: partial grading on exploration queries
        if result.error:
            self.failed_attempts += 1
            step_score = 0.0
            feedback   = f"SQL error: {result.error}"
        else:
            step_score, feedback = self.task["grader"](self.conn, action.query)
            if step_score > 0:
                self.failed_attempts = 0  # reset on progress
            else:
                self.failed_attempts += 1
            self.best_score = max(self.best_score, step_score)

        # Penalty for running out of steps
        steps_left = self.max_steps - self.steps_taken
        if steps_left <= 0:
            self.done = True

        # Provide hint after 3+ failed attempts
        hint = None
        if self.failed_attempts >= 3 and not self.done:
            hint = self.task["hint"]

        # Step reward is discounted (full score only on submit)
        discounted = round(step_score * 0.5, 3)

        reward = SQLReward(
            score=discounted,
            breakdown={"exploration_partial": discounted},
            feedback=feedback + (f" ({steps_left} steps left.)" if steps_left > 0 else " No steps left."),
            is_final=self.done,
        )
        obs = self._build_observation(hint=hint)
        return obs, reward, self.done, {}

    def state(self) -> SQLState:
        return SQLState(
            task_id         = self.task_id,
            difficulty      = self.task["difficulty"],
            steps_taken     = self.steps_taken,
            max_steps       = self.max_steps,
            query_history   = list(self.query_history),
            best_score_so_far = self.best_score,
            done            = self.done,
            submitted       = self.submitted,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _execute(self, sql: str) -> QueryResult:
        try:
            cur  = self.conn.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            return QueryResult(
                columns   = cols,
                rows      = [list(r) for r in rows[:50]],  # cap preview at 50
                row_count = len(rows),
            )
        except Exception as e:
            return QueryResult(error=str(e))

    def _build_observation(self, hint: Optional[str] = None) -> SQLObservation:
        return SQLObservation(
            task_id          = self.task_id,
            task_description = self.task["description"],
            difficulty       = self.task["difficulty"],
            schema_info      = self.task["schema_info"],
            sample_data      = self.task["sample_data"],
            last_query       = self.last_query,
            last_result      = self.last_result,
            steps_taken      = self.steps_taken,
            max_steps        = self.max_steps,
            hint             = hint,
            done             = self.done,
        )


# ── Session store (in-process; fine for single-worker Docker) ─────────────────

_sessions: Dict[str, SQLEnvironment] = {}


def get_or_create(session_id: str, task_id: str) -> SQLEnvironment:
    if session_id not in _sessions:
        _sessions[session_id] = SQLEnvironment(task_id)
    return _sessions[session_id]


def create_session(task_id: str) -> SQLEnvironment:
    env = SQLEnvironment(task_id)
    _sessions[env.session_id] = env
    return env


def get_session(session_id: str) -> Optional[SQLEnvironment]:
    return _sessions.get(session_id)


def delete_session(session_id: str):
    _sessions.pop(session_id, None)
