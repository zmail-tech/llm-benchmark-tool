#!/usr/bin/env python3
"""
LLM Benchmark Tool - Test local LLMs running on an OpenAI-compatible endpoint.

Usage:
    python benchmark.py --questions questions.txt --model my-model [--output-dir results]
    python benchmark.py --questions-file questions.json --model my-model
    python benchmark.py --prompt "What is 2+2?" --model my-model
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
        settings["model"] = config["llm"].get("model-id", "").strip()
        settings["base_url"] = config["llm"].get("url", "").strip()
        settings["api_key"] = config["llm"].get("api-key", "").strip()

    return settings


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


def run_question(
    client: OpenAI,
    model: str,
    question: str,
    question_index: int,
) -> dict:
    """Send a single question to the LLM and return the response with metrics."""
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": question},
            ],
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

            # Accumulate answer text
            if choice.delta and choice.delta.content:
                answer_parts.append(choice.delta.content)

            # Extract usage from chunk (some servers include per-chunk usage)
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

        # If usage was not available per-chunk, try to get it from a non-streaming call
        # Some endpoints only return usage on the final chunk or not at all with streaming
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

        # Calculate response tokens as completion - thinking (if we have both)
        if completion_tokens > 0 and response_tokens == 0:
            response_tokens = completion_tokens - thinking_tokens

        prompt_tps = prompt_tokens / duration if duration > 0 else 0
        response_tps = response_tokens / duration if duration > 0 else 0

        result = {
            "question": question,
            "answer": answer,
            "model": model,
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
            "model": model,
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


def save_result(result: dict, output_dir: str, question_index: int) -> str:
    """Save a single question/result to a file. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)

    # Sanitize filename from the question
    safe_name = "".join(
        c if c.isalnum() or c in ("-", "_", ".") else "_"
        for c in result["question"][:80]
    )
    filename = f"q{question_index:03d}_{safe_name}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return filepath


def save_summary(results: list[dict], output_dir: str) -> str:
    """Save a summary of all results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "summary.json")

    total_duration = sum(r["metrics"]["duration_seconds"] for r in results)
    total_completion = sum(r["metrics"]["completion_tokens"] for r in results)
    total_thinking = sum(r["metrics"]["thinking_tokens"] for r in results)
    total_response = sum(r["metrics"]["response_tokens"] for r in results)
    total_prompt = sum(r["metrics"]["prompt_tokens"] for r in results)

    summary = {
        "total_questions": len(results),
        "model": results[0]["model"] if results else "",
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


def main():
    config = load_config()

    parser = argparse.ArgumentParser(
        description="Benchmark a local LLM running on an OpenAI-compatible endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Single question (uses config.ini for model/endpoint if present)
  python benchmark.py --prompt "What is 2+2?"

  # Questions from a text file (one per line)
  python benchmark.py --questions questions.txt --model qwen3-8b

  # Override config.ini with CLI flags
  python benchmark.py --prompt "Hello" -m my-model -u http://localhost:8080/v1
        """,
    )

    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="A single question to ask the model",
    )
    parser.add_argument(
        "--questions", "-q",
        type=str,
        help="Path to a file with questions (text: one per line, or JSON/JSONL)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model ID to use (overrides config.ini)",
    )
    parser.add_argument(
        "--base-url", "-u",
        type=str,
        default=None,
        help="OpenAI-compatible endpoint URL (overrides config.ini)",
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=None,
        help="API key (overrides config.ini)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save result files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help=f"Path to config file (default: {CONFIG_FILENAME})",
    )

    args = parser.parse_args()

    # Merge: CLI args take precedence over config, then defaults
    model = args.model or config.get("model") or DEFAULT_MODEL
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

    # Initialize client
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    print(f"Endpoint: {base_url}")
    print(f"Model:    {model}")
    print(f"Questions: {len(questions)}")
    print(f"Output:    {args.output_dir}/")
    print("-" * 60)

    results: list[dict] = []
    for idx, question in enumerate(questions, start=1):
        print(f"[{idx}/{len(questions)}] Sending question...")

        result = run_question(client, model, question, idx)

        # Print truncated answer preview
        preview = result["answer"][:120]
        if len(result["answer"]) > 120:
            preview += "..."
        print(f"  Preview: {preview}")
        print(f"  Duration: {result['metrics']['duration_seconds']}s  |  "
              f"Response TPS: {result['metrics']['response_tokens_per_second']}  |  "
              f"Tokens: thinking={result['metrics']['thinking_tokens']} "
              f"response={result['metrics']['response_tokens']} "
              f"completion={result['metrics']['completion_tokens']}")

        filepath = save_result(result, args.output_dir, idx)
        result["_file"] = filepath
        results.append(result)

        print(f"  Saved:  {filepath}")
        print()

    # Save summary
    if results:
        summary_path = save_summary(results, args.output_dir)
        print(f"Summary saved: {summary_path}")

        # Print aggregate stats
        agg = json.load(open(summary_path))["aggregate_metrics"]
        print("-" * 60)
        print(f"Total questions:   {agg['total_prompt_tokens']} prompt tokens, "
              f"{agg['total_completion_tokens']} completion tokens")
        print(f"Total thinking:    {agg['total_thinking_tokens']} | "
              f"Total response:    {agg['total_response_tokens']}")
        print(f"Avg response TPS:  {agg['avg_response_tokens_per_second']}")
        print(f"Total duration:    {agg['total_duration_seconds']}s")

    print("-" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
