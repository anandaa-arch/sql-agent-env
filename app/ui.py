"""
Landing page for the SQL Agent OpenEnv HF Space.
Mounts a minimal Gradio UI at /ui so judges can interact manually.
The core OpenEnv API stays at /reset, /step, /state, /health.
"""
import gradio as gr
import httpx
import json

BASE = "http://localhost:7860"


def do_reset(task_id: str) -> tuple:
    try:
        r = httpx.post(f"{BASE}/reset", params={"task_id": task_id}, timeout=10)
        data = r.json()
        sid  = data.get("session_id", "")
        obs  = data.get("observation", {})
        desc = obs.get("task_description", "")
        schema = json.dumps(obs.get("schema_info", {}), indent=2)
        return sid, desc, schema, "✅ Episode started.", ""
    except Exception as e:
        return "", "", "", f"❌ Error: {e}", ""


def do_step(session_id: str, mode: str, query: str) -> tuple:
    if not session_id:
        return "⚠️ Call Reset first.", "", ""
    try:
        r = httpx.post(
            f"{BASE}/step",
            json={"mode": mode, "query": query},
            headers={"session-id": session_id},
            timeout=15,
        )
        data    = r.json()
        obs     = data.get("observation", {})
        reward  = data.get("reward", {})
        result  = obs.get("last_result") or {}
        done    = data.get("done", False)
        score   = reward.get("score", 0.0) if reward else 0.0
        fb      = reward.get("feedback", "") if reward else ""
        hint    = obs.get("hint") or ""

        result_text = ""
        if result.get("error"):
            result_text = f"❌ SQL Error: {result['error']}"
        else:
            cols = result.get("columns", [])
            rows = result.get("rows", [])
            lines = [" | ".join(str(c) for c in cols)]
            lines.append("-" * max(len(lines[0]), 20))
            for row in rows[:10]:
                lines.append(" | ".join(str(v) for v in row))
            if result.get("row_count", 0) > 10:
                lines.append(f"… ({result['row_count']} total rows)")
            result_text = "\n".join(lines)

        status = f"{'🏁 Done  ' if done else '▶ Running'} | Score: {score:.3f} | {fb}"
        if hint:
            status += f"\n💡 Hint: {hint}"
        return status, result_text, f"Steps: {obs.get('steps_taken',0)}/{obs.get('max_steps',10)}"
    except Exception as e:
        return f"❌ Error: {e}", "", ""


INTRO = """
# 🗄️ SQL Agent OpenEnv

An **OpenEnv-compliant RL environment** for training and evaluating agents on SQL generation.

### How it works
1. **Reset** an episode with a task (easy / medium / hard).
2. The agent sees the **table schema** and a natural-language goal.
3. The agent submits **SQL queries** to explore the database.
4. When confident, the agent **submits** the final answer for grading.
5. The grader returns a deterministic score **0.0 – 1.0** with partial credit.

### API endpoints (for programmatic use)
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset?task_id=...` | Start episode |
| `POST` | `/step` (header: `session-id`) | Submit action |
| `GET`  | `/state` (header: `session-id`) | Full state |
| `GET`  | `/health` | Liveness check |
| `GET`  | `/tasks` | List tasks |
"""

with gr.Blocks(title="SQL Agent OpenEnv", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1️⃣  Start Episode")
            task_dd   = gr.Dropdown(
                choices=["task_1_easy", "task_2_medium", "task_3_hard"],
                value="task_1_easy", label="Task"
            )
            reset_btn = gr.Button("🔄 Reset", variant="primary")
            sid_box   = gr.Textbox(label="Session ID", interactive=False)
            status_box = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 📋  Task Description")
            desc_box   = gr.Markdown()
            schema_box = gr.Code(label="Schema", language="json", interactive=False)

    gr.Markdown("---")
    gr.Markdown("### 2️⃣  Send a Query")
    with gr.Row():
        mode_radio = gr.Radio(
            choices=["sql", "submit"], value="sql",
            label="Mode  (sql = explore, submit = final answer)"
        )
        steps_lbl  = gr.Textbox(label="Steps", interactive=False, scale=0)

    query_box  = gr.Code(
        label="SQL Query", language="sql", interactive=True,
        value="SELECT * FROM customers LIMIT 5"
    )
    step_btn   = gr.Button("▶ Execute", variant="primary")

    with gr.Row():
        step_status = gr.Textbox(label="Result / Score", interactive=False)
        result_box  = gr.Code(label="Query Output", language="sql", interactive=False)

    reset_btn.click(
        do_reset,
        inputs=[task_dd],
        outputs=[sid_box, desc_box, schema_box, status_box, steps_lbl],
    )
    step_btn.click(
        do_step,
        inputs=[sid_box, mode_radio, query_box],
        outputs=[step_status, result_box, steps_lbl],
    )


# This module is imported by main.py and mounted at /ui
app = demo.app
