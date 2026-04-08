---
title: SQL Agent OpenEnv
emoji: 🗄️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - sql
  - agent
  - text-to-sql
app_port: 7860
---

# 🗄️ SQL Agent OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/openenv)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)

An **OpenEnv-compliant reinforcement learning environment** for training and evaluating AI agents on SQL query generation — one of the most practically important tasks in data engineering.

Agents must write correct SQLite queries from natural-language descriptions, iterating with exploratory queries before submitting a final answer.

---

## 🎯 Why SQL Generation?

Text-to-SQL is a real, high-value industry problem. Enterprises spend millions on analysts writing SQL by hand. This environment provides:
- **Deterministic grading** — SQL result sets are objectively comparable
- **Real-world schemas** — e-commerce, SaaS analytics, and financial data
- **Meaningful partial credit** — F1-score on result set overlap
- **Progressive difficulty** — from single JOIN to window functions

---

## 📐 Environment Design

### Action Space

```json
{
  "mode": "sql",
  "query": "SELECT name FROM customers WHERE ..."
}
```

Two modes:
- **`sql`**: Run any SQL against the task database. Results returned immediately. Reward is discounted (0.5×) to encourage exploration before committing.
- **`submit`**: Mark final answer. Full grader runs and episode ends.

### Observation Space

```json
{
  "task_id":          "task_1_easy",
  "task_description": "## Task: ...",
  "difficulty":       "easy",
  "schema_info":      { "customers": ["id INTEGER", ...] },
  "sample_data":      { "customers": [{...}, ...] },
  "last_query":       "SELECT ...",
  "last_result":      { "columns": [...], "rows": [[...]], "row_count": 3, "error": null },
  "steps_taken":      2,
  "max_steps":        10,
  "hint":             null,
  "done":             false
}
```

### Reward Function

| Situation | Reward |
|---|---|
| Exploration query — full match | 0.5 |
| Exploration query — partial match (F1) | 0 – 0.4 |
| Exploration query — SQL error | 0.0 |
| **Submit — full match** | **1.0** |
| Submit — partial match (F1) | 0 – 0.8 |
| Steps exhausted without submit | Episode ends, 0.0 |

After **3+ consecutive failed attempts**, a natural-language hint is injected into the observation.

---

## 📋 Tasks

### Task 1 — Easy: Customers With Orders
**Schema:** `customers`, `orders`  
**Goal:** Return name & email of every customer with ≥ 1 order (any status).  
**Challenge:** Simple JOIN + DISTINCT. Filter out non-ordering customers.  
**Expected baseline score (gpt-4o-mini):** ~0.95

### Task 2 — Medium: Plan-level Usage Report
**Schema:** `users`, `events`  
**Goal:** Per plan, compute total events since 2024-01-01 and avg per user (including 0-event users).  
**Challenge:** LEFT JOIN to preserve zero-event users, date filtering, GROUP BY + ROUND.  
**Expected baseline score (gpt-4o-mini):** ~0.72

### Task 3 — Hard: Account Balance Milestones
**Schema:** `accounts`, `transactions`  
**Goal:** Per account, find the first date running balance exceeded 5000 (NULL if never) + total credits.  
**Challenge:** Window functions (`SUM OVER`), CTEs, LEFT JOIN for NULL milestones.  
**Expected baseline score (gpt-4o-mini):** ~0.48

---

## 🚀 Setup

### Local Development

```bash
git clone <repo-url>
cd sql-agent-env
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash
docker build -t sql-agent-env .
docker run -p 7860:7860 sql-agent-env
```

### Docker Compose (env + inference together)

```bash
export HF_TOKEN=sk-...
export MODEL_NAME=gpt-4o-mini
export API_BASE_URL=https://api.openai.com/v1
docker compose up
```

### Running the Baseline Manually

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-...
export ENV_BASE_URL=http://localhost:7860

python inference.py
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/tasks` | GET | List all tasks with descriptions |
| `/reset?task_id=task_1_easy` | POST | Start new episode, returns `session_id` |
| `/step` | POST | Submit action (header: `session-id`) |
| `/state` | GET | Full episode state (header: `session-id`) |
| `/session` | DELETE | Clean up session |

### Quick Example

```python
import httpx

http = httpx.Client(base_url="http://localhost:7860")

# 1. Start episode
r = http.post("/reset", params={"task_id": "task_1_easy"})
session_id = r.json()["session_id"]

# 2. Explore
r = http.post("/step",
    json={"mode": "sql", "query": "SELECT * FROM customers LIMIT 3"},
    headers={"session-id": session_id})
print(r.json()["observation"]["last_result"])

# 3. Submit final answer
r = http.post("/step",
    json={"mode": "submit",
          "query": "SELECT DISTINCT c.name, c.email FROM customers c JOIN orders o ON c.id = o.customer_id"},
    headers={"session-id": session_id})
print(r.json()["reward"])  # {"score": 1.0, "feedback": "Perfect!", ...}
```

---

## 📊 Baseline Scores

Scores with `gpt-4o-mini` at `temperature=0`:

| Task | Difficulty | Score |
|---|---|---|
| task_1_easy | easy | 0.95 |
| task_2_medium | medium | 0.72 |
| task_3_hard | hard | 0.48 |
| **Average** | | **0.72** |

---

## 🏗️ Project Structure

```
sql-agent-env/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app & OpenEnv HTTP endpoints
│   ├── environment.py   # Episode logic & in-memory SQLite session management
│   ├── models.py        # Pydantic: Action, Observation, Reward, State
│   └── tasks.py         # Task schemas, seed data, grader functions
├── inference.py         # Baseline inference script (START/STEP/END logging)
├── openenv.yaml         # OpenEnv spec metadata
├── docker-compose.yml   # Local dev: env + inference runner
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚖️ License

MIT
