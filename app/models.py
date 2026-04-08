"""
Typed Pydantic models for the SQL Agent OpenEnv environment.
Compliant with OpenEnv spec: Action, Observation, Reward, State.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Action ──────────────────────────────────────────────────────────────────

class SQLAction(BaseModel):
    """
    Action the agent submits each step.

    Two modes:
      - sql:    Execute a SQL statement against the environment database.
                The result is returned in the next observation.
      - submit: Mark the agent's final answer query. Triggers the grader.
    """
    mode: str = Field(
        default="sql",
        description="'sql' to run a query, 'submit' to finalize the answer."
    )
    query: str = Field(
        description="The SQL string to execute or submit as the final answer."
    )


# ─── Observation ─────────────────────────────────────────────────────────────

class QueryResult(BaseModel):
    columns: List[str] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)
    row_count: int = 0
    error: Optional[str] = None


class SQLObservation(BaseModel):
    """
    What the agent sees after each step.
    Contains the database schema, task description, and last query result.
    """
    task_id: str
    task_description: str
    difficulty: str                        # easy | medium | hard
    schema_info: Dict[str, List[str]]      # table_name -> [col descriptions]
    sample_data: Dict[str, List[Dict]]     # table_name -> first 3 rows
    last_query: Optional[str] = None
    last_result: Optional[QueryResult] = None
    steps_taken: int = 0
    max_steps: int = 10
    hint: Optional[str] = None            # Progressive hint after 3+ failed attempts
    done: bool = False


# ─── Reward ──────────────────────────────────────────────────────────────────

class SQLReward(BaseModel):
    """
    Per-step reward signal. Provides meaningful partial progress.
    """
    score: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    feedback: str = ""
    is_final: bool = False


# ─── State ───────────────────────────────────────────────────────────────────

class SQLState(BaseModel):
    """
    Full internal environment state (returned by state() endpoint).
    """
    task_id: str
    difficulty: str
    steps_taken: int
    max_steps: int
    query_history: List[str] = Field(default_factory=list)
    best_score_so_far: float = 0.0
    done: bool = False
    submitted: bool = False
