#!/usr/bin/env python3
"""Flask API server for the LLM Benchmark Tool."""

import json
import os
import threading
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory

import benchmark as bm
import db as dbmod

app = Flask(__name__, static_folder="static", static_url_path="")

# State for long-running operations
_state = {
    "benchmark": {"running": False, "progress": None, "error": None, "output_dir": None},
    "evaluation": {"running": False, "progress": None, "error": None},
}

_RESULTS_DIR = "results"


def _write_progress(kind: str, current: int, total: int):
    """Update progress state for polling."""
    _state[kind]["progress"] = {
        "current": current,
        "total": total,
        "percent": round(current / total * 100, 1) if total else 0,
    }


def _resolve_model(model: dict, llm_url: str = "", llm_api_key: str = "") -> dict:
    """Resolve a model's base_url and api_key from DB defaults if null."""
    return {
        "name": model["name"],
        "model_id": model["model_id"],
        "base_url": model["base_url"] or llm_url or bm.DEFAULT_BASE_URL,
        "api_key": model["api_key"] or llm_api_key or bm.DEFAULT_API_KEY,
    }


def _run_benchmark_bg(output_dir, questions, model_names):
    """Background thread for benchmark runs."""
    _state["benchmark"]["running"] = True
    _state["benchmark"]["error"] = None
    _state["benchmark"]["output_dir"] = output_dir
    conn = dbmod.get_db()
    try:
        # Get settings
        llm_url = dbmod.get_setting(conn, "llm_url") or bm.DEFAULT_BASE_URL
        llm_api_key = dbmod.get_setting(conn, "llm_api_key") or bm.DEFAULT_API_KEY

        # Get models
        all_models = dbmod.get_models(conn)

        if model_names:
            models_to_test = [m for m in all_models if m["name"] in model_names]
        else:
            models_to_test = all_models

        if not models_to_test:
            llm_model = dbmod.get_setting(conn, "llm_model") or bm.DEFAULT_MODEL
            models_to_test = [{
                "id": 0,
                "name": llm_model,
                "model_id": llm_model,
                "base_url": llm_url,
                "api_key": llm_api_key,
            }]

        # Resolve models with inheritance
        resolved_models = [_resolve_model(m, llm_url, llm_api_key) for m in models_to_test]
        total_steps = len(questions) * len(resolved_models)

        _write_progress("benchmark", 0, total_steps)
        # Create run
        run_id = dbmod.create_run(conn, output_dir, len(questions), len(models_to_test))

        model_summaries = []
        q_idx = 0
        for model in resolved_models:
            # Get the DB model id for this model name
            model_row = dbmod.get_model_by_name(conn, model["name"])
            model_id_db = model_row["id"] if model_row else None

            for question in questions:
                q_idx += 1
                _write_progress("benchmark", q_idx, total_steps)

                client = bm.OpenAI(base_url=model["base_url"], api_key=model["api_key"])
                result = bm.run_question(client, model["model_id"], question)

                dbmod.save_result(
                    conn, run_id, model_id_db, q_idx,
                    result.get("question", ""),
                    result.get("answer", ""),
                    result.get("metrics", {}),
                    result.get("error"),
                    result.get("timestamp"),
                )

                # Track per-model results for summary
                existing = None
                for ms in model_summaries:
                    if ms["model_name"] == model["name"]:
                        existing = ms
                        break

                if existing is None:
                    ms_list = {"model_name": model["name"], "model_id": model["model_id"], "results": []}
                    model_summaries.append(ms_list)
                    existing = ms_list

                existing["results"].append(result)

            conn.commit()

        # Mark run as completed
        dbmod.update_run_status(conn, run_id, "completed")
        conn.commit()

    except Exception as e:
        _state["benchmark"]["error"] = str(e)
        try:
            dbmod.update_run_status(conn, run_id, "error", str(e))
            conn.commit()
        except Exception:
            pass
    finally:
        _state["benchmark"]["running"] = False
        conn.close()


def _run_evaluation_bg():
    """Background thread for evaluation runs."""
    _state["evaluation"]["running"] = True
    _state["evaluation"]["error"] = None

    conn = dbmod.get_db()
    try:
        # Get eval settings
        eval_url = dbmod.get_setting(conn, "eval_url") or bm.DEFAULT_BASE_URL
        eval_api_key = dbmod.get_setting(conn, "eval_api_key") or bm.DEFAULT_API_KEY
        eval_model_id = dbmod.get_setting(conn, "eval_model_id") or bm.DEFAULT_EVAL_MODEL
        eval_prompt_template = dbmod.get_setting(conn, "eval_prompt_template") or ""

        if not eval_prompt_template:
            # Fallback: read from file
            eval_prompt_path = "eval-prompt.txt"
            if os.path.isfile(eval_prompt_path):
                eval_prompt_template = bm.load_eval_prompt(eval_prompt_path)
            else:
                _state["evaluation"]["error"] = "Eval prompt not configured"
                return

        eval_model = {
            "name": eval_model_id,
            "model_id": eval_model_id,
            "base_url": eval_url,
            "api_key": eval_api_key,
        }

        # Collect all runs and their results
        runs = dbmod.get_runs(conn)
        total_results = 0
        for run in runs:
            results = dbmod.get_results_by_run(conn, run["id"])
            total_results += len(results)

        processed = 0

        for run in runs:
            results = dbmod.get_results_by_run(conn, run["id"])

            # Group results by model
            model_groups: dict[str, list[dict]] = {}
            for r in results:
                model_groups.setdefault(r["model_name"], []).append(r)

            for model_name, model_results in sorted(model_groups.items()):
                # Write temporary files for evaluation (benchmark.py still uses files for eval)
                tmp_dir = os.path.join(_RESULTS_DIR, ".eval-tmp", f"run_{run['id']}", model_name)
                os.makedirs(tmp_dir, exist_ok=True)

                file_results = []
                for idx, r in enumerate(model_results, start=1):
                    tmp_path = os.path.join(tmp_dir, f"q{idx:03d}_tmp.json")
                    with open(tmp_path, "w") as f:
                        json.dump({
                            "question": r["question"],
                            "answer": r["answer"],
                            "model_id": r.get("model_model_id", model_name),
                            "metrics": r.get("metrics", {}),
                        }, f)
                    file_results.append(tmp_path)

                bm.run_evaluation(
                    conn, run["id"], eval_model,
                    file_results, eval_prompt_template
                )
                processed += len(file_results)
                _write_progress("evaluation", processed, total_results)
                conn.commit()

        # Clean up temp files
        import shutil
        tmp_base = os.path.join(_RESULTS_DIR, ".eval-tmp")
        if os.path.isdir(tmp_base):
            shutil.rmtree(tmp_base)

    except Exception as e:
        _state["evaluation"]["error"] = str(e)
    finally:
        _state["evaluation"]["running"] = False
        conn.close()


# ── API Endpoints ──────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/config")
def get_config():
    """Return current configuration (without API keys)."""
    conn = dbmod.get_db()
    try:
        llm_url = dbmod.get_setting(conn, "llm_url") or ""
        llm_model = dbmod.get_setting(conn, "llm_model") or ""
        llm_api_key = dbmod.get_setting(conn, "llm_api_key") or ""

        eval_url = dbmod.get_setting(conn, "eval_url") or ""
        eval_model_id = dbmod.get_setting(conn, "eval_model_id") or ""
        eval_api_key = dbmod.get_setting(conn, "eval_api_key") or ""

        eval_criteria_json = dbmod.get_setting(conn, "eval_criteria")
        eval_criteria = json.loads(eval_criteria_json) if eval_criteria_json else bm.EVAL_CRITERIA

        models = dbmod.get_models(conn)
        safe_models = []
        for m in models:
            safe_models.append({
                "id": m["id"],
                "name": m["name"],
                "model_id": m["model_id"],
                "base_url": m["base_url"],
                "api_key_set": bool(m["api_key"] and m["api_key"] != bm.DEFAULT_API_KEY),
            })

        return jsonify({
            "llm": {
                "url": llm_url,
                "model": llm_model,
                "api_key_set": bool(llm_api_key and llm_api_key != bm.DEFAULT_API_KEY),
            },
            "models": safe_models,
            "eval": {
                "url": eval_url,
                "model_id": eval_model_id,
                "api_key_set": bool(eval_api_key and eval_api_key != bm.DEFAULT_EVAL_API_KEY),
            },
            "eval_criteria": eval_criteria,
            "eval_prompt_template": dbmod.get_setting(conn, "eval_prompt_template") or "",
        })
    finally:
        conn.close()


@app.route("/api/config", methods=["PUT"])
def update_config():
    """Update configuration (URL, model IDs, and API keys)."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    conn = dbmod.get_db()
    try:
        if "llm" in data:
            if "url" in data["llm"]:
                dbmod.set_setting(conn, "llm_url", data["llm"]["url"])
            if "model" in data["llm"]:
                dbmod.set_setting(conn, "llm_model", data["llm"]["model"])
            if "api_key" in data["llm"]:
                dbmod.set_setting(conn, "llm_api_key", data["llm"]["api_key"])

        if "eval" in data:
            if "url" in data["eval"]:
                dbmod.set_setting(conn, "eval_url", data["eval"]["url"])
            if "model_id" in data["eval"]:
                dbmod.set_setting(conn, "eval_model_id", data["eval"]["model_id"])
            if "api_key" in data["eval"]:
                dbmod.set_setting(conn, "eval_api_key", data["eval"]["api_key"])

        if "eval_criteria" in data:
            dbmod.set_setting(conn, "eval_criteria", json.dumps(data["eval_criteria"]))

        if "eval_prompt_template" in data:
            dbmod.set_setting(conn, "eval_prompt_template", data["eval_prompt_template"])

        # Handle model list changes
        if "models" in data:
            model_list = data["models"]
            existing_models = dbmod.get_models(conn)
            existing_ids = {m["id"] for m in existing_models}
            existing_by_name = {m["name"]: m for m in existing_models}

            for m in model_list:
                m_id = m.get("id")
                m_name = m.get("name", "")
                m_model_id = m.get("model_id", m_name)
                m_base_url = m.get("base_url") or None
                m_api_key = m.get("api_key") or None

                if m_id and m_id in existing_ids:
                    dbmod.update_model(conn, m_id, m_name, m_model_id, m_base_url, m_api_key)
                elif m_name in existing_by_name:
                    existing_id = existing_by_name[m_name]["id"]
                    dbmod.update_model(conn, existing_id, m_name, m_model_id, m_base_url, m_api_key)
                else:
                    dbmod.add_model(conn, m_name, m_model_id,
                                    m_base_url if m_base_url else None,
                                    m_api_key if m_api_key else None)

        conn.commit()
        return jsonify({"ok": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route("/api/results")
def list_results():
    """List available runs and their summaries."""
    conn = dbmod.get_db()
    try:
        runs = dbmod.get_runs(conn)

        result_runs = []
        for run in runs:
            summaries = dbmod.get_results_for_summary(conn, run["id"])
            comparison = dbmod.get_cross_model_comparison(conn, run["id"])

            result_runs.append({
                "id": run["id"],
                "name": run["name"],
                "created_at": run["created_at"],
                "status": run["status"],
                "error": run["error"],
                "question_count": run["question_count"],
                "model_count": run["model_count"],
                "summary": summaries[0] if summaries else None,
                "summaries": summaries,
                "comparison": comparison,
            })

        return jsonify({"runs": result_runs})
    finally:
        conn.close()


@app.route("/api/results/<int:run_id>/questions")
def list_questions(run_id):
    """List individual question results for a run."""
    conn = dbmod.get_db()
    try:
        run = dbmod.get_run(conn, run_id)
        if not run:
            return jsonify({"error": "Run not found"}), 404

        results = dbmod.get_results_by_run(conn, run_id)
        questions = []
        for r in results:
            questions.append({
                "model_name": r["model_name"],
                "question_index": r["question_index"],
                "question": r["question"],
                "answer": r["answer"],
                "metrics": r["metrics"],
            })

        return jsonify({"questions": questions})
    finally:
        conn.close()


@app.route("/api/benchmark/run", methods=["POST"])
def start_benchmark():
    """Start a benchmark run in the background."""
    if _state["benchmark"]["running"]:
        return jsonify({"error": "Benchmark already running"}), 409

    data = request.get_json() or {}
    questions = data.get("questions", [])
    output_dir = data.get("output_dir", datetime.now().strftime("%Y%m%d_%H%M%S"))
    model_names = data.get("models", [])

    if not questions:
        return jsonify({"error": "No questions provided"}), 400

    t = threading.Thread(target=_run_benchmark_bg, args=(output_dir, questions, model_names), daemon=True)
    t.start()

    return jsonify({"ok": True, "output_dir": output_dir})


@app.route("/api/benchmark/status")
def benchmark_status():
    """Poll benchmark run status."""
    return jsonify(_state["benchmark"])


@app.route("/api/evaluation/run", methods=["POST"])
def start_evaluation():
    """Start an evaluation run in the background."""
    if _state["evaluation"]["running"]:
        return jsonify({"error": "Evaluation already running"}), 409

    t = threading.Thread(target=_run_evaluation_bg, daemon=True)
    t.start()

    return jsonify({"ok": True})


@app.route("/api/evaluation/status")
def evaluation_status():
    """Poll evaluation run status."""
    return jsonify(_state["evaluation"])


@app.route("/api/evaluation/results/<int:run_id>/<model_name>")
def get_eval_results(run_id, model_name):
    """Get evaluation results for a specific model in a run."""
    conn = dbmod.get_db()
    try:
        summary = dbmod.get_eval_summary_for_model(conn, run_id, model_name)
        if not summary:
            return jsonify({"error": "Eval results not found"}), 404
        return jsonify(summary)
    finally:
        conn.close()


@app.route("/api/graph/generate", methods=["POST"])
def generate_graph():
    """Generate a comparison graph from eval results."""
    data = request.get_json() or {}

    conn = dbmod.get_db()
    try:
        output_path = data.get("output_path", os.path.join(_RESULTS_DIR, "eval", "comparison-graph.html"))
        result = bm.generate_comparison_graph(conn=conn, output_path=output_path)
        return jsonify({
            "model_averages": result.get("model_averages", {}),
            "output": result.get("output", ""),
        })
    except SystemExit:
        return jsonify({"error": "No eval data found for graph generation"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route("/api/graph")
def get_graph():
    """Return the generated graph HTML if it exists."""
    graph_path = os.path.join(_RESULTS_DIR, "eval", "comparison-graph.html")
    if os.path.isfile(graph_path):
        with open(graph_path, "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/html"}
    return jsonify({"error": "Graph not generated yet"}), 404


@app.route("/api/results/<int:run_id>", methods=["DELETE"])
def delete_run(run_id):
    """Delete a run and all associated data."""
    conn = dbmod.get_db()
    try:
        run = dbmod.get_run(conn, run_id)
        if not run:
            return jsonify({"error": "Run not found"}), 404
        dbmod.delete_run(conn, run_id)
        conn.commit()
        return jsonify({"ok": True})
    finally:
        conn.close()


@app.route("/api/results/<int:run_id>/export")
def export_run(run_id):
    """Export a run as JSON (backward-compatible format)."""
    conn = dbmod.get_db()
    try:
        run = dbmod.get_run(conn, run_id)
        if not run:
            return jsonify({"error": "Run not found"}), 404

        results = dbmod.get_results_by_run(conn, run_id)
        summaries = dbmod.get_results_for_summary(conn, run_id)
        comparison = dbmod.get_cross_model_comparison(conn, run_id)

        return jsonify({
            "run": run,
            "results": results,
            "summaries": summaries,
            "comparison": comparison,
        })
    finally:
        conn.close()


@app.route("/api/models", methods=["POST"])
def add_model_api():
    """Add a new model."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Model name is required"}), 400

    model_id_val = data.get("model_id", name)
    base_url = data.get("base_url") or None
    api_key = data.get("api_key") or None

    conn = dbmod.get_db()
    try:
        existing = dbmod.get_model_by_name(conn, name)
        if existing:
            return jsonify({"error": f"Model '{name}' already exists"}), 409

        dbmod.add_model(conn, name, model_id_val, base_url, api_key)
        conn.commit()

        # Return the newly created model with its DB ID
        new_model = dbmod.get_model_by_name(conn, name)
        return jsonify({
            "ok": True,
            "model": {
                "id": new_model["id"],
                "name": new_model["name"],
                "model_id": new_model["model_id"],
                "base_url": new_model["base_url"],
                "api_key_set": bool(new_model["api_key"] and new_model["api_key"] != bm.DEFAULT_API_KEY),
            }
        })
    finally:
        conn.close()


@app.route("/api/models/<int:model_id>", methods=["PUT"])
def update_model_api(model_id):
    """Update a model's settings."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    conn = dbmod.get_db()
    try:
        model = dbmod.get_model_by_id(conn, model_id)
        if not model:
            return jsonify({"error": "Model not found"}), 404

        dbmod.update_model(
            conn, model_id,
            name=data.get("name"),
            model_id_val=data.get("model_id"),
            base_url=data.get("base_url"),
            api_key=data.get("api_key"),
        )
        conn.commit()
        return jsonify({"ok": True})
    finally:
        conn.close()


@app.route("/api/models/<int:model_id>", methods=["DELETE"])
def delete_model_api(model_id):
    """Delete a model."""
    conn = dbmod.get_db()
    try:
        model = dbmod.get_model_by_id(conn, model_id)
        if not model:
            return jsonify({"error": "Model not found"}), 404
        dbmod.delete_model(conn, model_id)
        conn.commit()
        return jsonify({"ok": True})
    finally:
        conn.close()


if __name__ == "__main__":
    # Initialize database
    dbmod.init_db()
    app.run(host="0.0.0.0", port=5000, debug=False)
