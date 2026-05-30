# LLM Benchmark Tool

A Python script for benchmarking local LLMs running on OpenAI-compatible endpoints. Sends a series of questions, captures responses, and records detailed performance metrics. Supports testing multiple models in a single run with a cross-model comparison.

## Features

- **Multi-model benchmarking** — test multiple models back-to-back and get a ranked comparison
- Per-question metrics: duration, prompt/response tokens per second, thinking tokens, response tokens, and completion tokens
- Each question/response saved as a separate JSON file per model
- Aggregate summary per model + cross-model comparison file
- Configure models, endpoints, and API keys in `config.ini` (or override with CLI flags)
- Load questions from plain text (one per line), JSON, or JSONL files

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

2. Edit `config.ini` with your LLM endpoint settings and model list:

```ini
[llm]
url = http://192.168.1.210:8000/v1
api-key = your-api-key

[models]
list = ModelA, ModelB

[model.ModelA]
model-id = ModelA

[model.ModelB]
model-id = ModelB
```

3. Add questions to `questions.txt` (one per line).

4. Run:

```bash
python benchmark.py -q questions.txt
```

## Configuring Multiple Models

Add a `[models]` section with a `list` of model names (comma-separated), then define each model in a `[model.<name>]` section. Models inherit the URL and API key from the `[llm]` section by default but can override them individually.

```ini
[llm]
url = http://192.168.1.210:8000/v1
api-key = your-api-key

[models]
list = Qwen-Lite-Deepseek, DeepSeek-R1

[model.Qwen-Lite-Deepseek]
model-id = Qwen-Lite-Deepseek

[model.DeepSeek-R1]
model-id = DeepSeek-R1
url = http://192.168.1.210:8001/v1
```

If no `[models]` section is found, the tool falls back to a single model from the `[llm]` section's `model-id`.

## Usage

### Single Model (CLI override)

```bash
python benchmark.py -p "What is 2+2?" -m qwen3-8b
```

### Multiple Models (from config)

```bash
python benchmark.py -q questions.txt
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

### Directory Structure

Each model gets its own subdirectory under the output folder:

```
results/
  Qwen-Lite-Deepseek/
    q001_Whats_a_good_simple_dinner_recipe.json
    q002_My_8_year_old_keeps_asking.json
    summary.json
  DeepSeek-R1/
    q001_Whats_a_good_simple_dinner_recipe.json
    q002_My_8_year_old_keeps_asking.json
    summary.json
  comparison.json
```

### Per-Question File

```json
{
  "question": "What's a good, simple dinner recipe...",
  "answer": "Here's a simple pasta dish...",
  "model_id": "Qwen-Lite-Deepseek",
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

### Per-Model Summary

`summary.json` inside each model's folder contains aggregate metrics: total/average duration, token counts, and average tokens per second.

### Cross-Model Comparison

`comparison.json` at the output root contains a ranked comparison of all models tested, ordered by response tokens per second (highest first).

## Token Counting

The script attempts to extract token counts from the streaming response. If the endpoint doesn't include usage data in streamed chunks, it makes a second non-streaming request to retrieve token counts. Thinking (reasoning) tokens are detected from `completion_tokens_details.reasoning_tokens` when available.
