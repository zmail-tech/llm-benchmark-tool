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

    args = parser.parse_args()

    if args.config:
        config_path_arg = args.config

    config = load_config(config_path_arg)
    models_from_config = load_models(config_path_arg)

    # Merge: CLI overrides, then config, then defaults
    base_url = args.base_url or config.get("base_url") or DEFAULT_BASE_URL
    api_key = args.api_key or config.get("api_key") or DEFAULT_API_KEY

    # Determine questions
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

    # Determine which models to test
    models_to_test: list[dict] = []

    if models_from_config:
        # Use models from config.ini [models] section
        models_to_test = models_from_config
    elif args.model:
        # Single model from CLI
        models_to_test = [{
            "name": args.model,
            "model_id": args.model,
            "base_url": base_url,
            "api_key": api_key,
        }]
    else:
        # Fall back to model from [llm] section
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

    # Run benchmarks for each model
    model_summaries: list[dict] = []
    for model in models_to_test:
        summary = run_model_benchmark(model, questions, args.output_dir)
        if summary:
            model_summaries.append(summary)

    # Cross-model comparison
    if len(model_summaries) > 1:
        comp_path = save_cross_model_summary(model_summaries, args.output_dir)
        print(f"\n{'='*60}")
        print(f"CROSS-MODEL COMPARISON")
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
