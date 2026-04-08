"""
inference.py — Baseline inference script for SQL Agent OpenEnv
=============================================================

Runs a language model against all three tasks and emits structured
stdout logs in [START] / [STEP] / [END] format as required by the
OpenEnv evaluation harness.

Environment variables required:
  API_BASE_URL   e.g. https://api.groq.com/openai/v1
  MODEL_NAME     e.g. llama-3.1-8b-instant
  HF_TOKEN       your Groq / OpenAI API key
  ENV_BASE_URL   base URL of the running OpenEnv server (default: http://localhost:7860)

Usage:
  python inference.py
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.1-8b-instant")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS_PER_TASK = 10
SUCCESS_THRESHOLD  = 0.8
TASKS = ["task_1_easy", "task_2_medium", "task_3_hard"]

# ── Structured log helpers ─────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(json.dumps({
        "event":      "START",
        "task":       task,
        "env":        env,
        "model":      model,
        "timestamp":  time.time(),
    }), flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(json.dumps({
        "event":   "STEP",
        "step":    step,
        "action":  action[:500],
        "reward":  reward,
        "done":    done,
        "error":   error,
    }), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(json.dumps({
        "event":   "END",
        "success": success,
        "steps":   steps,
        "score":   score,
        "rewards": rewards,
    }), flush=True)


# ── LLM helper ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert SQL engineer. Your job is to write correct SQLite SQL queries.

You will receive:
- A task description with the goal
- Table schema definitions
- Sample data (first few rows)
- Results from your previous queries (if any)
- Hints (if you've been struggling)

Strategy:
1. First, explore the data with simple SELECT * queries to understand the structure.
2. Build up your query incrementally.
3. When confident, use mode="submit" to submit your final answer.

Always respond with a JSON object:
  {"mode": "sql",    "query": "SELECT ..."}   -- to explore
  {"mode": "submit", "query": "SELECT ..."}   -- to submit final answer

Only JSON. No markdown, no explanation outside the JSON.
"""


def get_model_action(
    client: OpenAI,
    obs: Dict[str, Any],
    history: List[str],
) -> Dict[str, str]:
    """Call the LLM and return a parsed action dict."""
    obs_text = (
        f"Task: {obs.get('task_id')} ({obs.get('difficulty')})\n\n"
        f"{obs.get('task_description', '')}\n\n"
        f"Schema:\n{json.dumps(obs.get('schema_info', {}), indent=2)}\n\n"
        f"Sample data:\n{json.dumps(obs.get('sample_data', {}), indent=2)}\n"
    )
    if obs.get("last_query"):
        obs_text += f"\nYour last query:\n{obs['last_query']}\n"
    if obs.get("last_result"):
        res = obs["last_result"]
        if res.get("error"):
            obs_text += f"\nError: {res['error']}\n"
        else:
            obs_text += (
                f"\nResult ({res.get('row_count', 0)} rows):\n"
                f"Columns: {res.get('columns', [])}\n"
                f"Rows: {json.dumps(res.get('rows', [])[:5])}\n"
            )
    if obs.get("hint"):
        obs_text += f"\nHint: {obs['hint']}\n"

    steps_taken = obs.get("steps_taken", 0)
    max_steps   = obs.get("max_steps", 10)
    steps_left  = max_steps - steps_taken
    obs_text += f"\nSteps remaining: {steps_left}\n"
    if steps_left <= 2:
        obs_text += "WARNING: Few steps left — consider submitting now.\n"

    messages = [{"role": "user", "content": obs_text}]
    for h in history[-4:]:
        messages.insert(-1, {"role": "assistant", "content": h})

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            temperature=0.0,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return {"mode": "sql", "query": "SELECT 1"}


# ── Environment HTTP client ───────────────────────────────────────────────────

def env_reset(http: httpx.Client, task_id: str) -> Dict[str, Any]:
    resp = http.post(f"{ENV_BASE_URL}/reset", params={"task_id": task_id})
    resp.raise_for_status()
    return resp.json()


def env_step(http: httpx.Client, session_id: str, action: Dict) -> Dict[str, Any]:
    resp = http.post(
        f"{ENV_BASE_URL}/step",
        json=action,
        headers={"session-id": session_id},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_task(client: OpenAI, http: httpx.Client, task_id: str) -> float:
    log_start(task=task_id, env="sql-agent-env", model=MODEL_NAME)

    result     = env_reset(http, task_id)
    session_id = result["session_id"]
    obs        = result["observation"]

    history     : List[str] = []
    rewards     : List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    try:
        for step in range(1, MAX_STEPS_PER_TASK + 1):
            if obs.get("done"):
                break

            action = get_model_action(client, obs, history)
            if action.get("mode") not in ("sql", "submit"):
                action["mode"] = "sql"
            if not action.get("query"):
                action["query"] = "SELECT 1"

            step_result = env_step(http, session_id, action)

            obs     = step_result["observation"]
            reward  = (step_result.get("reward") or {}).get("score", 0.0)
            done    = step_result.get("done", False)
            info    = step_result.get("info", {})
            error   = (step_result.get("reward") or {}).get("feedback") if reward == 0 else None

            rewards.append(reward)
            steps_taken = step
            history.append(json.dumps(action))

            log_step(step=step, action=action["query"], reward=reward, done=done, error=error)

            if done:
                score = info.get("final_score", reward)
                break

        if not score and rewards:
            score = max(rewards)

        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    finally:
        try:
            http.delete(f"{ENV_BASE_URL}/session", headers={"session-id": session_id})
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN environment variable not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    with httpx.Client(timeout=60.0) as http:
        for attempt in range(10):
            try:
                r = http.get(f"{ENV_BASE_URL}/health")
                if r.status_code == 200:
                    print("[DEBUG] Environment is ready.", flush=True)
                    break
            except Exception:
                pass
            print(f"[DEBUG] Waiting for environment... attempt {attempt+1}", flush=True)
            time.sleep(3)

        all_scores = {}
        for task_id in TASKS:
            print(f"\n{'='*60}", flush=True)
            print(f"[DEBUG] Starting task: {task_id}", flush=True)
            score = run_task(client, http, task_id)
            all_scores[task_id] = score
            print(f"[DEBUG] Task {task_id} finished. Score: {score:.3f}", flush=True)

    print("\n" + "="*60, flush=True)
    print("[DEBUG] FINAL RESULTS:", flush=True)
    for tid, s in all_scores.items():
        print(f"  {tid}: {s:.3f}", flush=True)
    avg = sum(all_scores.values()) / len(all_scores)
    print(f"  AVERAGE: {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()