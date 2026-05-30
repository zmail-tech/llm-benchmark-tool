#!/usr/bin/env python3
"""SQLite database layer for the LLM Benchmark Tool."""

import configparser
import json
import os
import sqlite3
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.db")

# Default eval criteria
EVAL_CRITERIA_DEFAULT = ["Accuracy", "Completeness", "Clarity", "Reasoning", "Speed", "Refusal", "Overall"]

# Default config values for migration
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "local-model"
DEFAULT_API_KEY = "none"
DEFAULT_EVAL_MODEL = "eval-model"
DEFAULT_EVAL_BASE_URL = "http://localhost:8000/v1"
DEFAULT_EVAL_API_KEY = "none"

# Built-in default evaluation prompt template (fresh installs without eval-prompt.txt)
DEFAULT_EVAL_PROMPT = """Evaluate the following answer to the given question. Provide a structured assessment covering these dimensions:

**Question:**
{question}

**Answer:**
{answer}

**Response Metrics:**
- Duration: {duration}s
- Response Tokens: {response_tokens}
- Tokens/Second: {response_tps}

**Evaluation Criteria:**

1. **Accuracy (1-10):** How factually correct is the answer? Are there any inaccuracies or errors?
2. **Completeness (1-10):** Does the answer address all aspects of the question? Is anything missing?
3. **Clarity (1-10):** Is the answer well-organized, easy to follow, and clearly written?
4. **Reasoning (1-10):** If the question requires reasoning, is the logic sound and well-explained?
5. **Speed (1-10):** Considering the response metrics, was the answer delivered at an acceptable pace? Is the token count reasonable for the question, or is the response overly verbose?
6. **Refusal (1-10):** Did the model refuse to answer the question? A score of 10 means the model answered fully without any refusal. A score of 1 means the model completely refused to answer. Partial refusals (e.g., answering some parts but declining others) should receive an intermediate score.

For each criterion, provide a score and a brief justification. Then give an overall assessment and a final composite score (1-10).

Format your response as:
- Accuracy: [score] - [reason]
- Completeness: [score] - [reason]
- Clarity: [score] - [reason]
- Reasoning: [score] - [reason]
- Speed: [score] - [reason]
- Refusal: [score] - [reason]
- Overall: [score] - [summary]"""


def get_db() -> sqlite3.Connection:
    """Return a new SQLite connection with check_same_thread=False for Flask threading."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row) -> dict:
    """Convert a sqlite3.Row to a plain dict."""
    return dict(row)


def init_db():
    """Create tables if they don't exist, seeded with defaults."""
    needs_migration = not os.path.isfile(DB_PATH)
    conn = get_db()
    try:
        _create_tables(conn)
        set_default_settings(conn)
        if needs_migration:
            _migrate_from_config(conn)
            _migrate_results(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _create_tables(conn):
    """Create all tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            model_id TEXT NOT NULL,
            base_url TEXT,
            api_key TEXT
        );
        
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT DEFAULT 'completed',
            error TEXT,
            question_count INTEGER,
            model_count INTEGER
        );
        
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            model_id INTEGER NOT NULL,
            question_index INTEGER,
            question TEXT NOT NULL,
            answer TEXT,
            metrics TEXT,
            error TEXT,
            timestamp TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(id),
            FOREIGN KEY (model_id) REFERENCES models(id)
        );
        
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            question_index INTEGER,
            question TEXT NOT NULL,
            answer TEXT,
            metrics TEXT,
            evaluation TEXT,
            eval_duration_seconds REAL,
            error TEXT,
            timestamp TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        );
    """)


# ── Settings CRUD ───────────────────────────────────────────

def get_settings(conn) -> dict:
    """Return all settings as a key-value dict."""
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {r["key"]: r["value"] for r in rows}


def get_setting(conn, key: str) -> Optional[str]:
    """Get a single setting value by key."""
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_setting(conn, key: str, value: str):
    """Upsert a setting."""
    conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))


def set_default_settings(conn):
    """Seed the settings table with default values if empty."""
    existing = conn.execute("SELECT key FROM settings").fetchall()
    existing_keys = {r["key"] for r in existing}
    if existing_keys:
        return

    defaults = {
        "llm_url": "",
        "llm_api_key": "",
        "llm_model": "",
        "eval_url": "",
        "eval_api_key": "",
        "eval_model_id": "",
        "eval_criteria": json.dumps(EVAL_CRITERIA_DEFAULT),
        "eval_prompt_template": DEFAULT_EVAL_PROMPT,
    }
    for k, v in defaults.items():
        if k not in existing_keys:
            conn.execute("INSERT INTO settings (key, value) VALUES (?, ?)", (k, v))


# ── Models CRUD ─────────────────────────────────────────────

def get_models(conn) -> list[dict]:
    """Return all models."""
    rows = conn.execute("SELECT id, name, model_id, base_url, api_key FROM models").fetchall()
    return [{
        "id": r["id"],
        "name": r["name"],
        "model_id": r["model_id"],
        "base_url": r["base_url"],
        "api_key": r["api_key"],
    } for r in rows]


def get_model_by_id(conn, model_id: int) -> Optional[dict]:
    """Get a model by its ID."""
    row = conn.execute("SELECT id, name, model_id, base_url, api_key FROM models WHERE id = ?", (model_id,)).fetchone()
    return _row_to_dict(row) if row else None


def get_model_by_name(conn, name: str) -> Optional[dict]:
    """Get a model by its name."""
    row = conn.execute("SELECT id, name, model_id, base_url, api_key FROM models WHERE name = ?", (name,)).fetchone()
    return _row_to_dict(row) if row else None


def add_model(conn, name: str, model_id: str, base_url: str = None, api_key: str = None):
    """Insert a new model."""
    conn.execute(
        "INSERT INTO models (name, model_id, base_url, api_key) VALUES (?, ?, ?, ?)",
        (name, model_id, base_url, api_key),
    )


def update_model(conn, model_id: int, name: str = None, model_id_val: str = None, base_url: str = None, api_key: str = None):
    """Update a model's fields (only non-None values are updated)."""
    fields = []
    values = []
    if name is not None:
        fields.append("name = ?")
        values.append(name)
    if model_id_val is not None:
        fields.append("model_id = ?")
        values.append(model_id_val)
    if base_url is not None:
        fields.append("base_url = ?")
        values.append(base_url)
    if api_key is not None:
        fields.append("api_key = ?")
        values.append(api_key)
    if fields:
        values.append(model_id)
        conn.execute(f"UPDATE models SET {', '.join(fields)} WHERE id = ?", values)


def delete_model(conn, model_id: int):
    """Delete a model by its ID, cascading to results."""
    conn.execute("DELETE FROM results WHERE model_id = ?", (model_id,))
    conn.execute("DELETE FROM models WHERE id = ?", (model_id,))


# ── Runs CRUD ───────────────────────────────────────────────

def get_runs(conn) -> list[dict]:
    """Return all runs."""
    rows = conn.execute("SELECT id, name, created_at, status, error, question_count, model_count FROM runs ORDER BY created_at DESC").fetchall()
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "created_at": r["created_at"],
            "status": r["status"],
            "error": r["error"],
            "question_count": r["question_count"],
            "model_count": r["model_count"],
        }
        for r in rows
    ]


def get_run(conn, run_id: int) -> Optional[dict]:
    """Get a run by its ID."""
    row = conn.execute("SELECT id, name, created_at, status, error, question_count, model_count FROM runs WHERE id = ?", (run_id,)).fetchone()
    return _row_to_dict(row) if row else None


def create_run(conn, name: str, question_count: int = None, model_count: int = None) -> int:
    """Create a new run. Returns the run id."""
    cursor = conn.execute(
        """INSERT INTO runs (name, created_at, status, question_count, model_count)
           VALUES (?, ?, 'running', ?, ?)""",
        (name, datetime.now().isoformat(), question_count, model_count),
    )
    return cursor.lastrowid


def update_run_status(conn, run_id: int, status: str, error: str = None):
    """Update a run's status (and optional error)."""
    conn.execute("UPDATE runs SET status = ?, error = ? WHERE id = ?", (status, error, run_id))


def update_run_counts(conn, run_id: int, question_count: int = None, model_count: int = None):
    """Update question/model counts for a completed run."""
    if question_count is not None:
        conn.execute("UPDATE runs SET question_count = ? WHERE id = ?", (question_count, run_id))
    if model_count is not None:
        conn.execute("UPDATE runs SET model_count = ? WHERE id = ?", (model_count, run_id))


def delete_run(conn, run_id: int):
    """Delete a run and all associated results and evaluations."""
    conn.execute("DELETE FROM results WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM evaluations WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))


# ── Results CRUD ────────────────────────────────────────────

def get_results_by_run(conn, run_id: int) -> list[dict]:
    """Return all results for a run, ordered by question_index."""
    rows = conn.execute(
        """SELECT r.id, r.run_id, r.model_id, r.question_index, r.question, r.answer,
                  r.metrics, r.error, r.timestamp,
                  m.name AS model_name, m.model_id AS model_model_id
           FROM results r
           JOIN models m ON r.model_id = m.id
           WHERE r.run_id = ?
           ORDER BY m.name, r.question_index""",
        (run_id,),
    ).fetchall()
    return [
        {
            "id": r["id"],
            "run_id": r["run_id"],
            "model_id": r["model_id"],
            "model_name": r["model_name"],
            "model_model_id": r["model_model_id"],
            "question_index": r["question_index"],
            "question": r["question"],
            "answer": r["answer"],
            "metrics": json.loads(r["metrics"]) if r["metrics"] else {},
            "error": r["error"],
            "timestamp": r["timestamp"],
        }
        for r in rows
    ]


def save_result(conn, run_id: int, model_id: int, question_index: int, question: str,
                answer: str = None, metrics: dict = None, error: str = None, timestamp: str = None):
    """Insert a result."""
    conn.execute(
        """INSERT INTO results (run_id, model_id, question_index, question, answer, metrics, error, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, model_id, question_index, question, answer,
         json.dumps(metrics) if metrics else None, error, timestamp),
    )


def get_results_for_summary(conn, run_id: int) -> list[dict]:
    """Get aggregated per-model summary data for a run (to replace summary.json)."""
    rows = conn.execute(
        """SELECT m.name AS model_name,
                  m.model_id AS model_model_id,
                  COUNT(r.id) AS total_questions,
                  SUM(r.metrics->>'$.duration_seconds') AS total_duration,
                  SUM(r.metrics->>'$.completion_tokens') AS total_completion,
                  SUM(r.metrics->>'$.thinking_tokens') AS total_thinking,
                  SUM(r.metrics->>'$.response_tokens') AS total_response,
                  SUM(r.metrics->>'$.prompt_tokens') AS total_prompt
           FROM results r
           JOIN models m ON r.model_id = m.id
           WHERE r.run_id = ?
           GROUP BY m.id""",
        (run_id,),
    ).fetchall()
    results = []
    for r in rows:
        total_dur = float(r["total_duration"]) or 0
        total_resp = float(r["total_response"]) or 0
        total_prompt = float(r["total_prompt"]) or 0
        total_comp = float(r["total_completion"]) or 0
        total_think = float(r["total_thinking"]) or 0
        total_q = int(r["total_questions"]) or 0
        results.append({
            "model_name": r["model_name"],
            "model_id": r["model_model_id"],
            "total_questions": total_q,
            "aggregate_metrics": {
                "total_duration_seconds": round(total_dur, 3),
                "avg_duration_seconds": round(total_dur / total_q, 3) if total_q else 0,
                "total_prompt_tokens": int(total_prompt),
                "total_completion_tokens": int(total_comp),
                "total_thinking_tokens": int(total_think),
                "total_response_tokens": int(total_resp),
                "avg_prompt_tokens_per_second": round(total_prompt / total_dur, 2) if total_dur > 0 else 0,
                "avg_response_tokens_per_second": round(total_resp / total_dur, 2) if total_dur > 0 else 0,
            },
        })
    return results


def get_cross_model_comparison(conn, run_id: int) -> dict:
    """Compute cross-model comparison from results (to replace comparison.json)."""
    summaries = get_results_for_summary(conn, run_id)
    comparison = {
        "models_tested": len(summaries),
        "per_model": [],
        "rankings": {},
        "timestamp": datetime.now().isoformat(),
    }
    for s in summaries:
        agg = s["aggregate_metrics"]
        comparison["per_model"].append({
            "model_name": s["model_name"],
            "model_id": s["model_id"],
            "total_questions": s["total_questions"],
            **agg,
        })
    ranked = sorted(comparison["per_model"], key=lambda x: x.get("avg_response_tokens_per_second", 0), reverse=True)
    for rank, entry in enumerate(ranked, start=1):
        comparison["rankings"][f"#{rank}"] = {
            "model_name": entry["model_name"],
            "avg_response_tps": entry["avg_response_tokens_per_second"],
        }
    return comparison


# ── Evaluations CRUD ────────────────────────────────────────

def get_evaluations_by_run(conn, run_id: int) -> list[dict]:
    """Return all evaluations for a run."""
    rows = conn.execute(
        """SELECT id, run_id, model_name, question_index, question, answer,
                  metrics, evaluation, eval_duration_seconds, error, timestamp
           FROM evaluations
           WHERE run_id = ?
           ORDER BY question_index""",
        (run_id,),
    ).fetchall()
    return [
        {
            "id": r["id"],
            "run_id": r["run_id"],
            "model_name": r["model_name"],
            "question_index": r["question_index"],
            "question": r["question"],
            "answer": r["answer"],
            "metrics": json.loads(r["metrics"]) if r["metrics"] else {},
            "evaluation": r["evaluation"],
            "eval_duration_seconds": r["eval_duration_seconds"],
            "error": r["error"],
            "timestamp": r["timestamp"],
        }
        for r in rows
    ]


def get_evaluations_grouped_by_model(conn, run_id: int) -> dict[str, list[dict]]:
    """Return evaluations grouped by model name."""
    evals = get_evaluations_by_run(conn, run_id)
    grouped: dict[str, list[dict]] = {}
    for e in evals:
        grouped.setdefault(e["model_name"], []).append(e)
    return grouped


def get_eval_summary_for_model(conn, run_id: int, model_name: str) -> Optional[dict]:
    """Get eval summary for a specific model in a run (to replace eval-summary.json)."""
    evals = get_evaluations_by_run(conn, run_id)
    model_evals = [e for e in evals if e["model_name"] == model_name]
    if not model_evals:
        return None

    total_dur = sum(e.get("eval_duration_seconds", 0) for e in model_evals)
    return {
        "eval_model": model_name,
        "eval_model_id": model_name,
        "total_evaluations": len(model_evals),
        "aggregate_metrics": {
            "total_duration_seconds": round(total_dur, 3),
            "avg_duration_seconds": round(total_dur / len(model_evals), 3),
        },
        "evaluations": [
            {
                "model_id": e.get("metrics", {}).get("response_tokens", 0),
                "question": e["question"],
                "evaluation": e["evaluation"],
                "metrics": e.get("metrics"),
                "duration_seconds": e["eval_duration_seconds"],
            }
            for e in model_evals
        ],
        "individual_files": [],
        "timestamp": datetime.now().isoformat(),
    }


def save_evaluation(conn, run_id: int, model_name: str, question_index: int, question: str,
                    answer: str = None, metrics: dict = None, evaluation: str = None,
                    eval_duration_seconds: float = None, error: str = None, timestamp: str = None):
    """Insert an evaluation."""
    conn.execute(
        """INSERT INTO evaluations (run_id, model_name, question_index, question, answer,
                                    metrics, evaluation, eval_duration_seconds, error, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, model_name, question_index, question, answer,
         json.dumps(metrics) if metrics else None, evaluation,
         eval_duration_seconds, error, timestamp),
    )


def delete_evaluations_by_run(conn, run_id: int):
    """Delete all evaluations for a run."""
    conn.execute("DELETE FROM evaluations WHERE run_id = ?", (run_id,))


# ── Migration from config.ini ───────────────────────────────

def _migrate_from_config(conn):
    """Migrate config.ini settings into the settings and models tables."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
    if not os.path.isfile(config_path):
        return

    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")

    # Insert settings
    if config.has_section("llm"):
        set_setting(conn, "llm_url", config["llm"].get("url", "").strip() or DEFAULT_BASE_URL)
        api_key = config["llm"].get("api-key", "").strip()
        set_setting(conn, "llm_api_key", api_key if api_key and api_key != DEFAULT_API_KEY else "")
        model_id = config["llm"].get("model-id", "").strip()
        set_setting(conn, "llm_model", model_id if model_id else DEFAULT_MODEL)

    if config.has_section("eval"):
        set_setting(conn, "eval_url", config["eval"].get("url", "").strip() or DEFAULT_EVAL_BASE_URL)
        eval_api_key = config["eval"].get("api-key", "").strip()
        set_setting(conn, "eval_api_key", eval_api_key if eval_api_key and eval_api_key != DEFAULT_EVAL_API_KEY else "")
        eval_model = config["eval"].get("model-id", "").strip()
        set_setting(conn, "eval_model_id", eval_model if eval_model else DEFAULT_EVAL_MODEL)

    # Insert models
    if config.has_section("models"):
        model_names_raw = config["models"].get("list", "")
        model_names = [n.strip() for n in model_names_raw.split(",") if n.strip()]

        llm_base_url = config["llm"].get("url", "").strip() or DEFAULT_BASE_URL
        llm_api_key = config["llm"].get("api-key", "").strip()
        llm_model = config["llm"].get("model-id", "").strip()

        for name in model_names:
            section = f"model.{name}"
            model_id_val = llm_model or DEFAULT_MODEL
            base_url_val = llm_base_url
            api_key_val = ""

            if config.has_section(section):
                model_id_val = config[section].get("model-id", llm_model or DEFAULT_MODEL).strip() or model_id_val
                base_url_val = config[section].get("url", llm_base_url).strip() or base_url_val
                api_key_val = config[section].get("api-key", llm_api_key or DEFAULT_API_KEY).strip()

            # Only insert if not already present
            existing = get_model_by_name(conn, name)
            if not existing:
                add_model(conn, name, model_id_val, base_url_val if base_url_val and base_url_val != DEFAULT_BASE_URL else None,
                          api_key_val if api_key_val and api_key_val != DEFAULT_API_KEY else None)


def _migrate_results(conn):
    """Migrate results/ directory contents into the database.
    
    Walks the results/ directory, infers run boundaries, and populates
    runs, results, and evaluations tables.
    """
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.isdir(results_dir):
        return

    # Collect all model subdirectories that are NOT under eval/
    all_dirs = []
    for root, dirs, files in os.walk(results_dir):
        dirs[:] = [d for d in dirs if d != "eval"]
        rel = os.path.relpath(root, results_dir)
        if rel == ".":
            continue
        all_dirs.append((root, rel, files))

    # Group by first-level directory name (the run name)
    run_dirs: dict[str, list[tuple]] = {}
    for root, rel, files in all_dirs:
        parts = rel.split(os.sep)
        run_name = parts[0]
        run_dirs.setdefault(run_name, []).append((root, rel, files))

    for run_name, dirs_in_run in run_dirs.items():
        # Create the run
        cursor = conn.execute(
            """INSERT INTO runs (name, created_at, status, question_count, model_count)
               VALUES (?, ?, 'completed', ?, ?)""",
            (run_name, datetime.now().isoformat(), 0, len(dirs_in_run)),
        )
        run_id = cursor.lastrowid

        total_questions = 0
        for root, rel, files in dirs_in_run:
            # This is a model subdirectory within the run
            # Get model info: try to find the model in the DB first
            model_name = os.path.basename(root)
            model = get_model_by_name(conn, model_name)

            # If model not in DB yet, add it
            model_id_for_results = None
            if model:
                model_id_for_results = model["id"]
            else:
                # Try to get model_id from the first result file or from the name
                model_id_val = model_name  # fallback
                for f in sorted(files):
                    if f.startswith("q") and f.endswith(".json"):
                        try:
                            with open(os.path.join(root, f), "r") as fobj:
                                data = json.load(fobj)
                                model_id_val = data.get("model_id", model_name)
                                break
                        except Exception:
                            continue

                # Check if model already exists by model_id
                model_by_id = get_model_by_name(conn, model_id_val)
                if model_by_id:
                    model_id_for_results = model_by_id["id"]
                else:
                    add_model(conn, model_name, model_id_val, None, None)
                    model = get_model_by_name(conn, model_name)
                    if model:
                        model_id_for_results = model["id"]

            for fname in sorted(files):
                if fname.startswith("q") and fname.endswith(".json"):
                    try:
                        with open(os.path.join(root, fname), "r") as fobj:
                            data = json.load(fobj)

                        # Parse question index from filename
                        idx_match = fname[:7]  # "q001_"
                        question_index = int(idx_match[1:4]) if idx_match.startswith("q") else None

                        metrics_data = data.get("metrics", {})
                        save_result(conn, run_id, model_id_for_results, question_index,
                                    data.get("question", ""),
                                    data.get("answer", ""),
                                    metrics_data,
                                    data.get("error"),
                                    data.get("timestamp"))
                        total_questions += 1
                    except Exception:
                        continue

        # Update run counts
        conn.execute("UPDATE runs SET question_count = ?, status = 'completed' WHERE id = ?",
                     (total_questions, run_id))

    # ── Migrate eval data ──────────────────────────────────────
    eval_dir = os.path.join(results_dir, "eval")
    if not os.path.isdir(eval_dir):
        return

    for model_eval_dir_name in sorted(os.listdir(eval_dir)):
        model_eval_dir = os.path.join(eval_dir, model_eval_dir_name)
        if not os.path.isdir(model_eval_dir):
            continue

        # Find the run this eval belongs to by checking which run has this model
        for run_name, dirs_in_run in run_dirs.items():
            has_model = any(os.path.basename(root) == model_eval_dir_name for root, _, _ in dirs_in_run)
            if has_model:
                run_id = None
                row = conn.execute("SELECT id FROM runs WHERE name = ?", (run_name,)).fetchone()
                if row:
                    run_id = row["id"]

                # Use the model directory name as the benchmarked model name
                # (NOT eval_model from eval-summary.json, which is the evaluator model)
                benchmarked_model_name = model_eval_dir_name

                # Process individual eval files
                for ef in sorted(os.listdir(model_eval_dir)):
                    if ef.startswith("eval_") and ef.endswith(".json"):
                        try:
                            with open(os.path.join(model_eval_dir, ef), "r") as fobj:
                                edata = json.load(fobj)

                            idx_match = ef[:7]  # "eval_q001_"
                            question_index = int(idx_match[6:9]) if idx_match.startswith("eval_q") else None

                            save_evaluation(
                                conn,
                                run_id if run_id else -1,
                                benchmarked_model_name,
                                question_index,
                                edata.get("question", ""),
                                edata.get("answer", ""),
                                edata.get("metrics", {}),
                                edata.get("evaluation", ""),
                                edata.get("eval_duration_seconds"),
                                edata.get("error"),
                                edata.get("timestamp"),
                            )
                        except Exception:
                            continue
                break
