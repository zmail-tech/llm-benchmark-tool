#!/usr/bin/env python3
"""
LLM Benchmark Tool - Test local LLMs running on an OpenAI-compatible endpoint.

Usage:
    python benchmark.py --questions questions.txt
    python benchmark.py --prompt "What is 2+2?" --model qwen3-8b
"""

import argparse
import configparser
import json
import os
import re
import sys
import time
from datetime import datetime

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package is required. Install it with: pip install openai")
    sys.exit(1)

DEFAULT_BASE_URL = "http://localhost:8080/v1"
DEFAULT_MODEL = "local-model"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_API_KEY = "none"

CONFIG_FILENAME = "config.ini"
DEFAULT_EVAL_PROMPT_FILE = "eval-prompt.txt"
DEFAULT_EVAL_MODEL = "eval-model"
DEFAULT_EVAL_API_KEY = "none"


def load_config(path: str | None = None) -> dict:
    """Load settings from config.ini. Returns empty dict if file not found."""
    config_path = path if path else CONFIG_FILENAME
    if not os.path.isfile(config_path):
        return {}

    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")

    settings: dict = {}

    if config.has_section("llm"):
        settings["base_url"] = config["llm"].get("url", "").strip()
        settings["api_key"] = config["llm"].get("api-key", "").strip()
        settings["model"] = config["llm"].get("model-id", "").strip()

    return settings


def load_models(config_path: str | None = None) -> list[dict]:
    """Load model definitions from config.ini [models] section.

    Each comma-separated entry is a model name. If a [model.<name>] section
    exists, its model-id/url/api-key are used; otherwise fall back to the
    [llm] defaults.

    Returns a list of dicts with keys: name, model_id, base_url, api_key.
    """
    cfg_path = config_path or CONFIG_FILENAME
    if not os.path.isfile(cfg_path):
        return []

    config = configparser.ConfigParser()
    config.read(cfg_path, encoding="utf-8")

    llm_defaults: dict = {}
    if config.has_section("llm"):
        llm_defaults["base_url"] = config["llm"].get("url", "").strip()
        llm_defaults["api_key"] = config["llm"].get("api-key", "").strip()
        llm_defaults["model_id"] = config["llm"].get("model-id", "").strip()

    if not config.has_section("models"):
        return []

    model_names_raw = config["models"].get("list", "")
    model_names = [n.strip() for n in model_names_raw.split(",") if n.strip()]

    models: list[dict] = []
    for name in model_names:
        section = f"model.{name}"
        entry: dict = {
            "name": name,
            "model_id": llm_defaults.get("model_id", DEFAULT_MODEL),
            "base_url": llm_defaults.get("base_url", DEFAULT_BASE_URL),
            "api_key": llm_defaults.get("api_key", DEFAULT_API_KEY),
        }

        if config.has_section(section):
            entry["model_id"] = config[section].get("model-id", entry["model_id"]).strip() or entry["model_id"]
            entry["base_url"] = config[section].get("url", entry["base_url"]).strip() or entry["base_url"]
            entry["api_key"] = config[section].get("api-key", entry["api_key"]).strip() or entry["api_key"]

        models.append(entry)

    return models


def load_eval_config(config_path: str | None = None) -> dict:
    """Load evaluation model settings from config.ini [eval] section."""
    cfg_path = config_path or CONFIG_FILENAME
    if not os.path.isfile(cfg_path):
        return {}

    config = configparser.ConfigParser()
    config.read(cfg_path, encoding="utf-8")

    settings: dict = {}
    if config.has_section("eval"):
        settings["base_url"] = config["eval"].get("url", "").strip()
        settings["api_key"] = config["eval"].get("api-key", "").strip()
        settings["model_id"] = config["eval"].get("model-id", "").strip()

    return settings


def load_eval_prompt(path: str) -> str:
    """Load the evaluation prompt template from a file.

    The template may contain {question} and {answer} placeholders that will be
    replaced with the actual question and answer during evaluation.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_questions(path: str) -> list[str]:
    """Load questions from a file. Supports plain text (one per line) or JSON."""
    questions: list[str] = []

    if path.endswith((".json", ".jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if path.endswith(".jsonl"):
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        if isinstance(data, str):
                            questions.append(data)
                        elif isinstance(data, dict) and "question" in data:
                            questions.append(data["question"])
                        else:
                            questions.append(json.dumps(data))
            else:
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            questions.append(item)
                        elif isinstance(item, dict) and "question" in item:
                            questions.append(item["question"])
                        else:
                            questions.append(json.dumps(item))
                else:
                    questions.append(content)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(line)

    return questions


def run_question(client: OpenAI, model: str, question: str) -> dict:
    """Send a single question to the LLM and return the response with metrics."""
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            stream=True,
        )

        answer_parts: list[str] = []
        thinking_tokens = 0
        response_tokens = 0
        completion_tokens = 0
        prompt_tokens = 0

        for chunk in response:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            if choice.delta and choice.delta.content:
                answer_parts.append(choice.delta.content)

            if chunk.usage:
                if chunk.usage.prompt_tokens:
                    prompt_tokens += chunk.usage.prompt_tokens
                if chunk.usage.completion_tokens:
                    completion_tokens += chunk.usage.completion_tokens
                if chunk.usage.completion_tokens_details:
                    details = chunk.usage.completion_tokens_details
                    if hasattr(details, "reasoning_tokens") and details.reasoning_tokens is not None:
                        thinking_tokens += details.reasoning_tokens
                    if hasattr(details, "accepted_prediction_tokens") and details.accepted_prediction_tokens is not None:
                        response_tokens += details.accepted_prediction_tokens

        end_time = time.time()
        duration = end_time - start_time
        answer = "".join(answer_parts)

        if completion_tokens == 0:
            usage_result = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
                stream=False,
            )
            if usage_result.usage:
                completion_tokens = usage_result.usage.completion_tokens or 0
                prompt_tokens = usage_result.usage.prompt_tokens or 0
                if usage_result.usage.completion_tokens_details:
                    details = usage_result.usage.completion_tokens_details
                    if hasattr(details, "reasoning_tokens") and details.reasoning_tokens is not None:
                        thinking_tokens = details.reasoning_tokens or 0
                    if hasattr(details, "accepted_prediction_tokens") and details.accepted_prediction_tokens is not None:
                        response_tokens = details.accepted_prediction_tokens or 0

        if completion_tokens > 0 and response_tokens == 0:
            response_tokens = completion_tokens - thinking_tokens

        prompt_tps = prompt_tokens / duration if duration > 0 else 0
        response_tps = response_tokens / duration if duration > 0 else 0

        result = {
            "question": question,
            "answer": answer,
            "model_id": model,
            "metrics": {
                "duration_seconds": round(duration, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "thinking_tokens": thinking_tokens,
                "response_tokens": response_tokens,
                "prompt_tokens_per_second": round(prompt_tps, 2),
                "response_tokens_per_second": round(response_tps, 2),
            },
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        end_time = time.time()
        result = {
            "question": question,
            "answer": f"ERROR: {e}",
            "model_id": model,
            "metrics": {
                "duration_seconds": round(end_time - start_time, 3),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "thinking_tokens": 0,
                "response_tokens": 0,
                "prompt_tokens_per_second": 0,
                "response_tokens_per_second": 0,
            },
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }

    return result


def sanitize_name(name: str, max_len: int = 80) -> str:
    """Sanitize a string for use as a filename component."""
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name[:max_len])


def save_result(result: dict, output_dir: str, question_index: int, model_name: str) -> str:
    """Save a single question/result to a file. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)

    safe_name = sanitize_name(result["question"])
    filename = f"q{question_index:03d}_{safe_name}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return filepath


def save_summary(results: list[dict], output_dir: str, model_name: str) -> str:
    """Save a summary of all results for one model to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "summary.json")

    total_duration = sum(r["metrics"]["duration_seconds"] for r in results)
    total_completion = sum(r["metrics"]["completion_tokens"] for r in results)
    total_thinking = sum(r["metrics"]["thinking_tokens"] for r in results)
    total_response = sum(r["metrics"]["response_tokens"] for r in results)
    total_prompt = sum(r["metrics"]["prompt_tokens"] for r in results)

    summary = {
        "model_name": model_name,
        "model_id": results[0].get("model_id", "") if results else "",
        "total_questions": len(results),
        "aggregate_metrics": {
            "total_duration_seconds": round(total_duration, 3),
            "avg_duration_seconds": round(total_duration / len(results), 3) if results else 0,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_thinking_tokens": total_thinking,
            "total_response_tokens": total_response,
            "avg_prompt_tokens_per_second": round(total_prompt / total_duration, 2) if total_duration > 0 else 0,
            "avg_response_tokens_per_second": round(total_response / total_duration, 2) if total_duration > 0 else 0,
        },
        "individual_files": [r.get("_file") for r in results],
        "timestamp": datetime.now().isoformat(),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return filepath


def save_cross_model_summary(model_summaries: list[dict], output_dir: str) -> str:
    """Save a cross-model comparison summary."""
    filepath = os.path.join(output_dir, "comparison.json")

    comparison = {
        "models_tested": len(model_summaries),
        "per_model": [],
        "rankings": {},
        "timestamp": datetime.now().isoformat(),
    }

    for summary in model_summaries:
        agg = summary["aggregate_metrics"]
        comparison["per_model"].append({
            "model_name": summary["model_name"],
            "model_id": summary["model_id"],
            "total_questions": summary["total_questions"],
            **agg,
        })

    # Rank by response tokens per second (highest first)
    ranked = sorted(comparison["per_model"], key=lambda x: x.get("avg_response_tokens_per_second", 0), reverse=True)
    for rank, entry in enumerate(ranked, start=1):
        comparison["rankings"][f"#{rank}"] = {
            "model_name": entry["model_name"],
            "avg_response_tps": entry["avg_response_tokens_per_second"],
        }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    return filepath


def run_model_benchmark(
    model: dict,
    questions: list[str],
    output_base_dir: str,
) -> dict | None:
    """Run all questions against a single model. Returns summary dict."""
    model_name = model["name"]
    safe_model = sanitize_name(model_name)
    output_dir = os.path.join(output_base_dir, safe_model)

    client = OpenAI(base_url=model["base_url"], api_key=model["api_key"])

    print(f"\n{'='*60}")
    print(f"MODEL: {model_name} ({model['model_id']})")
    print(f"URL:   {model['base_url']}")
    print(f"Out:   {output_dir}/")
    print(f"{'='*60}")

    results: list[dict] = []
    for idx, question in enumerate(questions, start=1):
        print(f"  [{idx}/{len(questions)}] Sending question...")

        result = run_question(client, model["model_id"], question)

        preview = result["answer"][:120]
        if len(result["answer"]) > 120:
            preview += "..."
        print(f"    Preview: {preview}")
        print(f"    Duration: {result['metrics']['duration_seconds']}s  |  "
              f"Response TPS: {result['metrics']['response_tokens_per_second']}  |  "
              f"Tokens: thinking={result['metrics']['thinking_tokens']} "
              f"response={result['metrics']['response_tokens']} "
              f"completion={result['metrics']['completion_tokens']}")

        filepath = save_result(result, output_dir, idx, model_name)
        result["_file"] = filepath
        results.append(result)
        print(f"    Saved:  {filepath}")

    if results:
        summary_path = save_summary(results, output_dir, model_name)
        print(f"\n  Summary saved: {summary_path}")

        agg = json.load(open(summary_path))["aggregate_metrics"]
        print(f"  Total: {agg['total_prompt_tokens']} prompt tokens, "
              f"{agg['total_completion_tokens']} completion tokens")
        print(f"  Thinking: {agg['total_thinking_tokens']} | Response: {agg['total_response_tokens']}")
        print(f"  Avg response TPS: {agg['avg_response_tokens_per_second']}")
        print(f"  Total duration: {agg['total_duration_seconds']}s")

        return json.load(open(summary_path))

    return None


def run_evaluation(
    conn_or_model,
    run_id_or_files,
    eval_model_or_files,
    result_files_or_prompt,
    eval_prompt_template_or_output=None,
    output_dir=None,
) -> list[dict]:
    """Evaluate previously saved benchmark results using an evaluation model.

    DB mode: run_evaluation(conn, run_id, eval_model, file_results, eval_prompt_template)
    File mode: run_evaluation(eval_model, result_files, eval_prompt_template, output_dir)
    """
    import db as dbmod

    # Detect which mode based on first argument type
    if hasattr(conn_or_model, "execute"):
        # DB mode: (conn, run_id, eval_model, file_results, eval_prompt_template)
        conn = conn_or_model
        run_id = run_id_or_files
        eval_model = eval_model_or_files
        result_files = result_files_or_prompt
        eval_prompt_template = eval_prompt_template_or_output
        is_db_mode = True
    else:
        # File mode: (eval_model, result_files, eval_prompt_template, output_dir)
        eval_model = conn_or_model
        result_files = run_id_or_files
        eval_prompt_template = eval_model_or_files
        output_dir = result_files_or_prompt
        is_db_mode = False

    client = OpenAI(base_url=eval_model["base_url"], api_key=eval_model["api_key"])

    if not is_db_mode:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"Eval model: {eval_model['name']} ({eval_model['model_id']})")
    print(f"Eval URL:   {eval_model['base_url']}")
    print(f"Results:    {len(result_files)} files")
    print(f"{'='*60}")

    evaluations: list[dict] = []

    for idx, filepath in enumerate(result_files, start=1):
        with open(filepath, "r", encoding="utf-8") as f:
            result = json.load(f)

        question = result.get("question", "")
        answer = result.get("answer", "")
        model_id = result.get("model_id", "unknown")
        metrics = result.get("metrics", {})
        duration = metrics.get("duration_seconds", 0)
        response_tokens = metrics.get("response_tokens", 0)
        response_tps = metrics.get("response_tokens_per_second", 0)

        eval_prompt = eval_prompt_template.format(
            question=question,
            answer=answer,
            duration=duration,
            response_tokens=response_tokens,
            response_tps=response_tps,
        )

        print(f"\n  [{idx}/{len(result_files)}] Evaluating: {question[:80]}...")

        start_time = time.time()
        try:
            resp = client.chat.completions.create(
                model=eval_model["model_id"],
                messages=[{"role": "user", "content": eval_prompt}],
                stream=True,
            )

            eval_parts: list[str] = []
            for chunk in resp:
                choice = chunk.choices[0] if chunk.choices else None
                if choice and choice.delta and choice.delta.content:
                    eval_parts.append(choice.delta.content)

            evaluation_text = "".join(eval_parts)
            duration = time.time() - start_time

            preview = evaluation_text[:120] + ("..." if len(evaluation_text) > 120 else "")
            print(f"    Preview: {preview}")
            print(f"    Duration: {duration:.3f}s")

            eval_entry = {
                "model_id": model_id,
                "question": question,
                "answer": answer,
                "metrics": {
                    "duration_seconds": duration,
                    "response_tokens": response_tokens,
                    "response_tokens_per_second": response_tps,
                },
                "evaluation": evaluation_text,
                "eval_duration_seconds": round(time.time() - start_time, 3),
                "source_file": filepath,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            duration = time.time() - start_time
            eval_entry = {
                "model_id": model_id,
                "question": question,
                "answer": answer,
                "metrics": {
                    "duration_seconds": duration,
                    "response_tokens": response_tokens,
                    "response_tokens_per_second": response_tps,
                },
                "evaluation": f"ERROR: {e}",
                "eval_duration_seconds": round(time.time() - start_time, 3),
                "source_file": filepath,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

        evaluations.append(eval_entry)

        if is_db_mode:
            # Write to DB
            model_name = os.path.basename(os.path.dirname(filepath))
            dbmod.save_evaluation(
                conn, run_id, model_name, idx,
                eval_entry["question"],
                eval_entry["answer"],
                eval_entry["metrics"],
                eval_entry["evaluation"],
                eval_entry["eval_duration_seconds"],
                eval_entry.get("error"),
                eval_entry.get("timestamp"),
            )
            print(f"    Saved:  db eval_id={conn.execute('SELECT last_insert_rowid()').fetchone()[0]}")
        else:
            # Save individual evaluation to file
            safe_name = sanitize_name(question)
            eval_filename = f"eval_q{idx:03d}_{safe_name}.json"
            eval_filepath = os.path.join(output_dir, eval_filename)
            with open(eval_filepath, "w", encoding="utf-8") as f:
                json.dump(eval_entry, f, indent=2, ensure_ascii=False)
            print(f"    Saved: {eval_filepath}")

    # Save evaluation summary
    if evaluations:
        total_duration = sum(e["eval_duration_seconds"] for e in evaluations)

        if is_db_mode:
            print("\n  Eval summary:")
            print(f"  Total eval duration: {total_duration:.3f}s")
        else:
            summary = {
                "eval_model": eval_model["name"],
                "eval_model_id": eval_model["model_id"],
                "total_evaluations": len(evaluations),
                "aggregate_metrics": {
                    "total_duration_seconds": round(total_duration, 3),
                    "avg_duration_seconds": round(total_duration / len(evaluations), 3),
                },
                "evaluations": [
                    {
                        "model_id": e["model_id"],
                        "question": e["question"],
                        "evaluation": e["evaluation"],
                        "metrics": e.get("metrics"),
                        "duration_seconds": e["eval_duration_seconds"],
                    }
                    for e in evaluations
                ],
                "individual_files": [e["source_file"] for e in evaluations],
                "timestamp": datetime.now().isoformat(),
            }

            summary_path = os.path.join(output_dir, "eval-summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print(f"\n  Eval summary saved: {summary_path}")
            print(f"  Total eval duration: {total_duration:.3f}s")

    return evaluations


EVAL_CRITERIA = [
    {"name": "Accuracy", "description": "How factually correct is the answer? Are there any inaccuracies or errors?"},
    {"name": "Completeness", "description": "Does the answer address all aspects of the question? Is anything missing?"},
    {"name": "Clarity", "description": "Is the answer well-organized, easy to follow, and clearly written?"},
    {"name": "Reasoning", "description": "If the question requires reasoning, is the logic sound and well-explained?"},
    {"name": "Speed", "description": "Considering the response metrics, was the answer delivered at an acceptable pace?"},
    {"name": "Refusal", "description": "Did the model refuse to answer? 10 = answered fully, 1 = completely refused."},
    {"name": "Overall", "description": "Final composite assessment of the answer quality."},
]


def _crit_name(c):
    """Extract criterion name from a dict {name, ...} or plain string."""
    return c.get("name", c) if isinstance(c, dict) else str(c)

def parse_eval_scores(evaluation_text: str, criteria: list[str] | None = None) -> dict[str, float] | None:
    """Extract numeric scores for each evaluation criterion from evaluation text.

    Returns a dict like {"Accuracy": 10, "Completeness": 9.5, ...} or None if
    no recognizable scores are found.
    """
    if not evaluation_text or evaluation_text.startswith("ERROR:"):
        return None

    crits = criteria or EVAL_CRITERIA
    scores: dict[str, float] = {}
    for criterion in crits:
        name = _crit_name(criterion)
        escaped = re.escape(name)
        match = re.search(
            rf'(?:[-•]\s*|\s|^)\**{escaped}:\s*(\d+\.?\d*?)(?:/\d+)?\**',
            evaluation_text,
        )
        if match:
            try:
                scores[name] = float(match.group(1))
            except ValueError:
                pass

    return scores if scores else None

def _html_escape(text: str) -> str:
    """Basic HTML escaping for safe embedding in generated HTML."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )

def generate_comparison_graph(eval_dir: str = None, output_path: str = None,
                               conn=None, run_ids=None) -> dict:
    """Generate an HTML comparison chart from evaluation data.

    If conn is provided and eval_dir is None, reads from DB.
    If run_ids is provided, only includes those runs (labels use "RunName - Model").
    Otherwise, reads eval-summary.json files from eval subdirectories.
    """
    import db as dbmod

    # Get criteria (dynamic — may come from DB)
    try:
        db_conn = conn if conn else dbmod.get_db()
        criteria_json = dbmod.get_setting(db_conn, "eval_criteria")
        criteria = json.loads(criteria_json) if criteria_json else EVAL_CRITERIA
        if not conn:
            db_conn.close()
    except Exception:
        criteria = EVAL_CRITERIA

    crit_names = [_crit_name(c) for c in criteria]

    model_data: dict[str, list[tuple[dict[str, float], str]]] = {}

    if conn and eval_dir is None:
        # Read from DB
        runs = dbmod.get_runs(conn)
        if run_ids:
            runs = [r for r in runs if r["id"] in run_ids]
        for run in runs:
            evals_by_model = dbmod.get_evaluations_grouped_by_model(conn, run["id"])
            for model_name, evals in evals_by_model.items():
                label = run["name"] + " - " + model_name if run_ids else model_name
                for ev in evals:
                    eval_text = ev.get("evaluation", "")
                    scores = parse_eval_scores(eval_text, criteria)
                    question = ev.get("question", "Unknown")
                    if scores:
                        model_data.setdefault(label, []).append((scores, question))
                    else:
                        print(f"  Warning: could not parse scores for {label}: {question[:60]}")
    else:
        # Legacy file-based approach
        if not eval_dir or not os.path.isdir(eval_dir):
            print(f"Error: Eval directory not found: {eval_dir}")
            sys.exit(1)

        for entry in sorted(os.listdir(eval_dir)):
            summary_path = os.path.join(eval_dir, entry, "eval-summary.json")
            if not os.path.isfile(summary_path):
                continue

            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except (json.JSONDecodeError, OSError):
                print(f"  Warning: could not read {summary_path}, skipping")
                continue

            evaluations = summary.get("evaluations", [])
            for ev in evaluations:
                eval_text = ev.get("evaluation", "")
                scores = parse_eval_scores(eval_text, criteria)
                question = ev.get("question", "Unknown")
                if scores:
                    model_data.setdefault(entry, []).append((scores, question))
                else:
                    print(f"  Warning: could not parse scores for {entry}: {question[:60]}")

    if not model_data:
        print("Error: No evaluatable data found.")
        sys.exit(1)

    # Compute averages
    model_names = sorted(model_data.keys())
    model_averages: dict[str, dict[str, float]] = {}
    for mname in model_names:
        entries = model_data[mname]
        totals: dict[str, list[float]] = {n: [] for n in crit_names}
        for scores, _ in entries:
            for n in crit_names:
                if n in scores:
                    totals[n].append(scores[n])
        model_averages[mname] = {n: (sum(v) / len(v)) if v else 0 for n, v in totals.items()}

    # Build Chart.js datasets
    COLORS = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
        "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
        "#9c755f", "#bab0ac", "#8c564b", "#c44e52",
    ]

    label_names_json = [f"'{_html_escape(n)}'" for n in crit_names]
    datasets_json: list[str] = []
    for idx, mname in enumerate(model_names):
        color = COLORS[idx % len(COLORS)]
        values = [model_averages[mname].get(n, 0) for n in crit_names]
        datasets_json.append(
            f"{{ label: {_html_escape(mname)}, backgroundColor: '{color}', "
            f"data: {values} }}"
        )

    # Build per-question collapsible details
    details_html: list[str] = []
    for mname in model_names:
        entries = model_data[mname]
        details_html.append(
            f"<details style='margin: 0.5em 0;'><summary><strong>"
            f"{_html_escape(mname)}</strong> — "
            f"{len(entries)} questions</summary>"
        )
        details_html.append(
            "<table border='1' cellpadding='4' cellspacing='0' "
            "style='border-collapse:collapse; font-size:0.85em;'>"
        )
        details_html.append("<tr>")
        details_html.append("<th>Question</th>")
        for n in crit_names:
            details_html.append(f"<th>{_html_escape(n)}</th>")
        details_html.append("</tr>")

        for scores, question in entries:
            details_html.append("<tr>")
            details_html.append(f"<td>{_html_escape(question[:100])}</td>")
            for n in crit_names:
                val = scores.get(n, "—")
                details_html.append(f"<td>{val}</td>")
            details_html.append("</tr>")

        details_html.append("</table></details>")

    # Generate HTML
    graph_source = "DB" if (conn and eval_dir is None) else _html_escape(eval_dir or "")
    avg_table_rows = "".join(
        f"<tr><td>{_html_escape(m)}</td>" +
        "".join(f"<td>{model_averages[m].get(n, 0):.1f}</td>" for n in crit_names) +
        "</tr>"
        for m in model_names
    )
    avg_table_headers = "".join(f"<th>{_html_escape(n)}</th>" for n in crit_names)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eval Score Comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 2em; background: #fafafa; color: #222; }}
  h1 {{ border-bottom: 2px solid #4e79a7; padding-bottom: 0.3em; }}
  canvas {{ max-width: 900px; }}
  details summary {{ cursor: pointer; }}
</style>
</head>
<body>
<h1>Eval Score Comparison</h1>
<p>Generated from eval results in <code>{graph_source}</code></p>
<canvas id="chart"></canvas>
<h2>Average Scores by Model</h2>
<table border="1" cellpadding="4" cellspacing="0"
       style="border-collapse:collapse;">
<tr><th>Model</th>{avg_table_headers}</tr>
{avg_table_rows}
</table>
<h2>Per-Question Breakdown</h2>
{"".join(details_html)}
<script>
const ctx = document.getElementById("chart").getContext("2d");
new Chart(ctx, {{
  type: "bar",
  data: {{
    labels: [{", ".join(label_names_json)}],
    datasets: [{"".join(datasets_json)}],
  }},
  options: {{
    responsive: true,
    scales: {{
      y: {{ beginAtZero: true, max: 10, ticks: {{ stepSize: 1 }} }},
    }},
    plugins: {{
      title: {{ display: true, text: "Average Scores Across Models" }},
      legend: {{ position: "top" }},
    }},
  }},
}});
</script>
</body>
</html>"""

    # Write output
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    return {"model_averages": model_averages, "model_names": model_names, "output": output_path}

def main():
    config_path_arg = None

    parser = argparse.ArgumentParser(
        description="Benchmark local LLMs on OpenAI-compatible endpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
  Examples:
    # Use models from config.ini [models] section
    python benchmark.py --questions questions.txt

    # Test a single model directly
    python benchmark.py --prompt "What is 2+2?" --model qwen3-8b

    # Override config with CLI flags
    python benchmark.py --prompt "Hello" -m my-model -u http://localhost:8080/v1

    # Evaluate saved results (requires [eval] in config.ini)
    python benchmark.py --eval

    # Evaluate with a custom prompt template
    python benchmark.py --eval --eval-prompt custom-eval-prompt.txt

    # Generate HTML comparison graph from eval results
    python benchmark.py --graph

    # Graph with custom output path
    python benchmark.py --graph --graph-output my-graph.html
          """,
    )

    parser.add_argument("--prompt", "-p", type=str, help="A single question to ask the model")
    parser.add_argument("--questions", "-q", type=str, help="Path to a questions file (text, JSON, or JSONL)")
    parser.add_argument("--model", "-m", type=str, default=None, help="Model ID to use (overrides config.ini)")
    parser.add_argument("--base-url", "-u", type=str, default=None, help="Endpoint URL (overrides config.ini)")
    parser.add_argument("--api-key", "-k", type=str, default=None, help="API key (overrides config.ini)")
    parser.add_argument("--output-dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save result files (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--config", "-c", type=str, default=None, help=f"Path to config file (default: {CONFIG_FILENAME})")
    parser.add_argument("--eval", "-e", action="store_true", help="Evaluate saved results using the evaluation model")
    parser.add_argument("--eval-prompt", type=str, default=None, help=f"Path to eval prompt template (default: {DEFAULT_EVAL_PROMPT_FILE})")
    parser.add_argument("--eval-dir", type=str, default=None, help="Directory containing result files to evaluate (default: results/)")
    parser.add_argument("--graph", "-g", action="store_true", help="Generate HTML comparison graph from eval results")
    parser.add_argument("--graph-output", type=str, default=None, help="Output path for graph HTML (default: results/eval/comparison-graph.html)")

    args = parser.parse_args()

    if args.config:
        config_path_arg = args.config

    config = load_config(config_path_arg)

    # Handle standalone evaluation mode
    if args.eval:
        eval_config = load_eval_config(config_path_arg)
        eval_prompt_path = args.eval_prompt or DEFAULT_EVAL_PROMPT_FILE

        if os.path.isfile(eval_prompt_path):
            eval_prompt_template = load_eval_prompt(eval_prompt_path)
        else:
            import db as dbmod
            eval_prompt_template = dbmod.DEFAULT_EVAL_PROMPT

        eval_model_id = eval_config.get("model_id", DEFAULT_EVAL_MODEL)
        eval_model = {
            "name": eval_model_id,
            "model_id": eval_model_id,
            "base_url": eval_config.get("base_url", DEFAULT_BASE_URL),
            "api_key": eval_config.get("api_key", DEFAULT_API_KEY),
        }

        eval_dir = args.eval_dir or DEFAULT_OUTPUT_DIR
        if not os.path.isdir(eval_dir):
            print(f"Error: Eval directory not found: {eval_dir}")
            sys.exit(1)

        # Collect result files grouped by model subdirectory
        from collections import defaultdict
        model_files: dict[str, list[str]] = defaultdict(list)
        for root, dirs, files in os.walk(eval_dir):
            # Skip the eval output directory itself
            dirs[:] = [d for d in dirs if d != "eval"]
            for fname in files:
                if fname.endswith(".json") and not fname.startswith(("summary", "comparison", "eval")):
                    # Determine the model subdirectory (immediate child of eval_dir)
                    rel = os.path.relpath(root, eval_dir)
                    model_name = rel.split(os.sep)[0]
                    model_files[model_name].append(os.path.join(root, fname))

        if not model_files:
            print(f"Error: No result files found in {eval_dir}/")
            sys.exit(1)

        eval_base_dir = os.path.join(eval_dir, "eval")

        for model_name, files in sorted(model_files.items()):
            model_eval_dir = os.path.join(eval_base_dir, model_name)
            run_evaluation(eval_model, files, eval_prompt_template, model_eval_dir)

        print(f"\n{'='*60}")
        print("Evaluation complete.")
        return

    # Handle standalone graph generation mode
    if args.graph:
        eval_base_dir = os.path.join(args.output_dir, "eval")
        graph_output = args.graph_output or os.path.join(eval_base_dir, "comparison-graph.html")

        res = generate_comparison_graph(eval_base_dir, graph_output)

        model_avgs = res["model_averages"]
        print(f"\n{'='*60}")
        print("EVAL SCORE COMPARISON")
        print(f"{'='*60}")

        crit_display_names = [_crit_name(c) for c in EVAL_CRITERIA]
        header = f"  {'Model':<25}" + "".join(f"{n:>8}" for n in crit_display_names)
        print(header)
        print("  " + "-" * (25 + 8 * len(crit_display_names)))
        for mname in res["model_names"]:
            row = f"  {_html_escape(mname):<25}" + "".join(
                f"{model_avgs[mname].get(n, 0):>8.1f}" for n in crit_display_names
            )
            print(row)

        print(f"\n  Graph saved: {graph_output}")
        print("  Open in browser to see interactive chart.")
        return

    # Benchmark mode (requires --prompt or --questions)
    models_from_config = load_models(config_path_arg)

    base_url = args.base_url or config.get("base_url") or DEFAULT_BASE_URL
    api_key = args.api_key or config.get("api_key") or DEFAULT_API_KEY

    questions: list[str] = []
    if args.prompt:
        questions = [args.prompt]
    elif args.questions:
        if not os.path.isfile(args.questions):
            print(f"Error: Questions file not found: {args.questions}")
            sys.exit(1)
        questions = load_questions(args.questions)
    else:
        print("Error: Provide --prompt or --questions")
        sys.exit(1)

    if not questions:
        print("Error: No questions to process")
        sys.exit(1)

    models_to_test: list[dict] = []

    if models_from_config:
        models_to_test = models_from_config
    elif args.model:
        models_to_test = [{
            "name": args.model,
            "model_id": args.model,
            "base_url": base_url,
            "api_key": api_key,
        }]
    else:
        model_id = config.get("model") or DEFAULT_MODEL
        models_to_test = [{
            "name": model_id,
            "model_id": model_id,
            "base_url": base_url,
            "api_key": api_key,
        }]

    print(f"Models:    {len(models_to_test)}")
    for m in models_to_test:
        print(f"  - {m['name']} ({m['model_id']}) @ {m['base_url']}")
    print(f"Questions: {len(questions)}")
    print(f"Output:    {args.output_dir}/")

    model_summaries: list[dict] = []
    for model in models_to_test:
        summary = run_model_benchmark(model, questions, args.output_dir)
        if summary:
            model_summaries.append(summary)

    # Cross-model comparison
    if len(model_summaries) > 1:
        comp_path = save_cross_model_summary(model_summaries, args.output_dir)
        print(f"\n{'='*60}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'='*60}")

        comp = json.load(open(comp_path))
        for rank_key, rank_info in comp["rankings"].items():
            marker = " <-- FASTEST" if rank_key == "#1" else ""
            print(f"  {rank_key} {rank_info['model_name']}: "
                  f"{rank_info['avg_response_tps']} tokens/s{marker}")

        print(f"\n  Saved: {comp_path}")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
