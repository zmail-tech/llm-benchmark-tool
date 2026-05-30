# LLM Benchmark Tool

A Python script for benchmarking local LLMs running on OpenAI-compatible endpoints. Sends a series of questions, captures responses, and records detailed performance metrics.

## Features

- Configure model, endpoint URL, and API key in a `config.ini` file (or override with CLI flags)
- Load questions from plain text (one per line), JSON, or JSONL files
- Per-question metrics: duration, prompt/response tokens per second, thinking tokens, response tokens, and completion tokens
- Each question/response saved as a separate JSON file
- Aggregate summary across all questions

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+.

## Quick Start

1. Copy `config.ini.example` to `config.ini` and fill in your settings:

```bash
cp config.ini.example config.ini
```

2. Edit `config.ini` with your LLM endpoint settings:

```ini
[llm]
model-id = Qwen-Lite-Deepseek
url = http://192.168.1.210:8000/v1
api-key = your-api-key
```

2. Add questions to `questions.txt` (one per line), or create your own question file.

3. Run:

```bash
python benchmark.py -q questions.txt
```

## Usage

### Single Question

```bash
python benchmark.py -p "What is 2+2?"
```

### Questions from a File

```bash
# Plain text (one question per line)
python benchmark.py -q questions.txt

# JSON array
python benchmark.py -q questions.json

# JSONL (one JSON object per line)
python benchmark.py -q questions.jsonl
```

### Override Config with CLI Flags

```bash
python benchmark.py -p "Hello" -m other-model -u http://localhost:8080/v1
```

### Custom Config File

```bash
python benchmark.py -q questions.txt -c /path/to/other.ini
```

### Custom Output Directory

```bash
python benchmark.py -q questions.txt -o my_results
```

## CLI Reference

| Flag | Short | Description |
|------|-------|-------------|
| `--prompt` | `-p` | A single question to ask |
| `--questions` | `-q` | Path to a questions file |
| `--model` | `-m` | Model ID (overrides config.ini) |
| `--base-url` | `-u` | Endpoint URL (overrides config.ini) |
| `--api-key` | `-k` | API key (overrides config.ini) |
| `--output-dir` | `-o` | Output directory (default: `results/`) |
| `--config` | `-c` | Path to config file (default: `config.ini`) |

## Output

### Per-Question Files

Each question is saved as a separate file in the output directory:

```
results/
  q001_Whats_a_good_simple_dinner_recipe.json
  q002_My_8_year_old_keeps_asking.json
  q003_Were_planning_a_weekend_trip.json
  q004_Can_you_suggest_some_fun.json
  summary.json
```

Each file contains:

```json
{
  "question": "What's a good, simple dinner recipe...",
  "answer": "Here's a simple pasta dish...",
  "model": "Qwen-Lite-Deepseek",
  "metrics": {
    "duration_seconds": 4.523,
    "prompt_tokens": 25,
    "completion_tokens": 312,
    "thinking_tokens": 180,
    "response_tokens": 132,
    "prompt_tokens_per_second": 5.53,
    "response_tokens_per_second": 29.18
  },
  "timestamp": "2025-05-30T11:00:00"
}
```

### Summary

`summary.json` contains aggregate metrics across all questions, including total/average duration, total token counts, and average tokens per second.

## Token Counting

The script attempts to extract token counts from the streaming response. If the endpoint doesn't include usage data in streamed chunks, it makes a second non-streaming request to retrieve token counts. Thinking (reasoning) tokens are detected from `completion_tokens_details.reasoning_tokens` when available.
