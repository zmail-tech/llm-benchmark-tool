# LLM Benchmark Tool

A Python script for benchmarking local LLMs running on OpenAI-compatible endpoints. Sends a series of questions, captures responses, and records detailed performance metrics. Supports testing multiple models in a single run with a cross-model comparison.

## Features

- **Multi-model benchmarking** — test multiple models back-to-back and get a ranked comparison
- **Evaluation mode** — evaluate saved benchmark results with a separate eval model and a customizable scoring prompt
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

## Evaluating Results

After running a benchmark, you can evaluate the saved results using a separate evaluation model. This is useful for quality assessment of model responses using a stronger model as a judge.

### Configuring the Eval Model

Add an `[eval]` section to `config.ini` with the eval model's endpoint and model ID:

```ini
[eval]
url = http://192.168.1.210:8000/v1
api-key = your-api-key
model-id = qwen3-32b
```

### Evaluation Prompt Template

The eval prompt is stored in `eval-prompt.txt` and uses `{question}` and `{answer}` placeholders that are replaced with the actual content from each result file. The template also receives response metrics (duration, response tokens, tokens/second) to support speed evaluation.

The default template scores on five dimensions: Accuracy, Completeness, Clarity, Reasoning, and Speed (each 1-10), plus an overall score.

You can create custom evaluation prompts for different tasks. For example, a code-focused eval prompt might score on correctness, style, and efficiency.

```text
Evaluate the following code answer. Use {question} and {answer} placeholders.

**Question:**
{question}

**Answer:**
{answer}

Score: Correctness (1-10), Style (1-10), Efficiency (1-10)
```

Use `--eval-prompt` to specify a custom template file.

### Running Evaluation

```bash
# Evaluate saved results using default settings
python benchmark.py --eval

# Evaluate with a custom prompt template
python benchmark.py --eval --eval-prompt code-eval-prompt.txt

# Evaluate results from a specific directory
python benchmark.py --eval --eval-dir my_results/
```

The eval process reads all result JSON files from the target directory, populates the eval prompt template, sends it to the eval model, and writes results to a subdirectory named `eval/` inside the results folder.

### Eval Output

Each evaluation is saved as an individual JSON file, plus an `eval-summary.json` with all assessments aggregated:

```
results/
  eval/
    eval_q001_Whats_a_good_simple_dinner_recipe.json
    eval_q002_My_8_year_old_keeps_asking.json
    eval-summary.json
```

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

### Full Workflow: Benchmark Then Evaluate

```bash
# Step 1: Run a benchmark against multiple models
python benchmark.py -q questions.txt

# Step 2: Evaluate all saved results
python benchmark.py --eval
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
| `--eval` | `-e` | Evaluate saved results using the eval model |
| `--eval-prompt` | | Path to eval prompt template (default: `eval-prompt.txt`) |
| `--eval-dir` | | Directory with result files to evaluate (default: `results/`) |

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
  eval/
    eval_q001_Whats_a_good_simple_dinner_recipe.json
    eval_q002_My_8_year_old_keeps_asking.json
    eval-summary.json
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

### Eval Summary

`eval-summary.json` inside the `eval/` folder contains all evaluation assessments, including the original response metrics (duration, tokens, tokens/second) and the eval model's scored feedback.

## Token Counting

The script attempts to extract token counts from the streaming response. If the endpoint doesn't include usage data in streamed chunks, it makes a second non-streaming request to retrieve token counts. Thinking (reasoning) tokens are detected from `completion_tokens_details.reasoning_tokens` when available.
