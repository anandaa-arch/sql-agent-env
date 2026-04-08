"""
FastAPI application — exposes the OpenEnv HTTP interface:

  POST /reset           → initial observation
  POST /step            → (observation, reward, done, info)
  GET  /state           → current state
  GET  /tasks           → list available tasks
  GET  /health          → liveness check
  DELETE /session       → cleanup
  GET  /ui              → interactive Gradio demo (HF Spaces landing page)

All OpenEnv endpoints require a session_id header (except /reset which creates one).
"""
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse
from typing import Optional
import uvicorn

from app.models import SQLAction, SQLObservation, SQLReward, SQLState
from app.environment import create_session, get_session, delete_session
from app.tasks import TASKS

app = FastAPI(
    title="SQL Agent OpenEnv",
    description=(
        "An OpenEnv-compliant environment for training and evaluating AI agents "
        "on SQL query generation tasks ranging from simple JOINs to window functions."
    ),
    version="1.0.0",
)

# Mount Gradio UI (gracefully skip if gradio is unavailable)
try:
    import gradio as gr
    from app.ui import demo as gradio_demo
    app = gr.mount_gradio_app(app, gradio_demo, path="/ui")
except Exception:
    pass  # Gradio optional — API still fully functional


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to the interactive UI on HF Spaces."""
    return RedirectResponse(url="/ui")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "environment": "sql-agent-env", "version": "1.0.0"}


# ── Task listing ──────────────────────────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id":         tid,
                "difficulty": t["difficulty"],
                "description": t["description"][:200] + "...",
            }
            for tid, t in TASKS.items()
        ]
    }


# ── OpenEnv core endpoints ────────────────────────────────────────────────────

@app.post("/reset")
def reset(task_id: str = Query(default="task_1_easy", description="Which task to load")):
    """
    Start a new episode. Creates a fresh session with a clean database.
    Returns the initial observation and a session_id for subsequent calls.
    """
    if task_id not in TASKS:
        raise HTTPException(400, detail=f"Unknown task_id '{task_id}'. Choose from: {list(TASKS)}")
    env = create_session(task_id)
    obs = env.reset()
    return {
        "session_id":  env.session_id,
        "observation": obs.model_dump(),
        "reward":      None,
        "done":        False,
        "info":        {"message": f"Episode started. Task: {task_id}"},
    }


@app.post("/step")
def step(
    action: SQLAction,
    session_id: str = Header(..., description="Session ID returned by /reset"),
):
    """
    Submit one action (SQL query or final submit).
    Returns observation, reward, done flag, and info dict.
    """
    env = get_session(session_id)
    if env is None:
        raise HTTPException(404, detail=f"Session '{session_id}' not found. Call /reset first.")

    obs, reward, done, info = env.step(action)
    return {
        "session_id":  session_id,
        "observation": obs.model_dump(),
        "reward":      reward.model_dump(),
        "done":        done,
        "info":        info,
    }


@app.get("/state")
def state(
    session_id: str = Header(..., description="Session ID returned by /reset"),
):
    """Return the full internal state of the current episode."""
    env = get_session(session_id)
    if env is None:
        raise HTTPException(404, detail=f"Session '{session_id}' not found.")
    return env.state().model_dump()


@app.delete("/session")
def close_session(
    session_id: str = Header(..., description="Session ID to delete"),
):
    """Clean up a session (optional — sessions are also cleaned up on server restart)."""
    delete_session(session_id)
    return {"message": f"Session {session_id} deleted."}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=False)
