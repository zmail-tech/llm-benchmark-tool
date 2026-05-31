# LLM Benchmark Tool

A web application for benchmarking local LLMs running on OpenAI-compatible endpoints. Configure models, run benchmarks, evaluate results with a judge model, and visualize comparisons — all through a browser-based UI backed by a SQLite database. Also supports traditional CLI usage for scripting and automation.

## Features

- **Web UI** — single-page application with tabs for configuration, benchmarking, results, evaluation, and graph visualization
- **Multi-model benchmarking** — test multiple models back-to-back in a single run with per-question metrics
- **Evaluation mode** — evaluate saved benchmark results with a separate eval model and a customizable scoring prompt
- **Interactive comparison graphs** — Chart.js bar charts with per-model averages and per-question breakdowns, with run selection for cross-run comparison
- **SQLite persistence** — all runs, results, evaluations, and settings stored in a single database file with automatic migration from legacy file-based results
- **Per-question metrics** — duration, prompt/response/completion/thinking tokens, and tokens per second
- **Cross-model comparison** — ranked performance comparison across all models in a run
- **Model inheritance** — models inherit endpoint URL and API key from global defaults but can override individually
- **Configure everything in-browser** — connection settings, model list, eval model, eval criteria, and prompt template
- **CLI interface** — traditional command-line usage for scripting and automation

## Quick Start

```bash
pip install -r requirements.txt
python server.py
```

Then open http://localhost:5000 in your browser.

Requires Python 3.9+.

## Web UI

The web application provides five tabs:

### Configure

Set up all project settings:

- **Connection Settings** — default LLM endpoint URL, model ID, and API key
- **Models to Test** — add, edit, and delete models with individual overrides for URL and API key
- **Evaluation Settings** — eval endpoint, eval model ID, eval API key, scoring criteria, and eval prompt template

### Run

Start a benchmark:

- Enter questions (one per line)
- Optionally name the run (auto-generated timestamp if left blank)
- Select which models to test (all selected by default)
- Progress bar updates in real time during the run

### Results

View and manage past runs:

- Status badges (Completed, Running, Error)
- Per-run model rankings by response tokens/second
- Per-model aggregate metrics (duration, tokens, TPS)
- Expandable question-level detail with individual Q&A pairs
- Export runs as JSON or delete them

### Evaluation

Run and view evaluations:

- Execute evaluation across all saved benchmark results using the configured eval model
- Progress bar updates in real time
- Per-model evaluation summaries with averaged criterion scores
- Expandable individual evaluation details per question

### Graph

Generate comparison charts:

- Select one or more runs to include (models labeled as "RunName - Model" for cross-run distinction)
- Generates an interactive Chart.js grouped bar chart embedded in an iframe
- Average scores table sorted by model
- Collapsible per-question breakdown tables

## Configuration (CLI)

Copy `config.ini.example` to `config.ini` and fill in your settings:

```bash
cp config.ini.example config.ini
```

On first launch, the web UI auto-migrates settings from `config.ini` into the SQLite database (`benchmark.db`). After migration, all configuration is managed through the web UI.

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
url = http://192.168.1.210:8001/v1
```

If no `[models]` section is found, the tool falls back to a single model from the `[llm]` section's `model-id`.

### Evaluation Model Configuration

```ini
[eval]
url = http://192.168.1.210:8000/v1
api-key = your-api-key
model-id = qwen3-32b
```

### Evaluation Prompt Template

The eval prompt is stored in `eval-prompt.txt` (or in the database via the web UI) and uses `{question}`, `{answer}`, `{duration}`, `{response_tokens}`, and `{response_tps}` placeholders. Default criteria are: Accuracy, Completeness, Clarity, Reasoning, Speed, Refusal, and Overall (each scored 1-10).

## CLI Reference

| Flag | Short | Description |
|------|-------|-------------|
| `--prompt` | `-p` | A single question to ask |
| `--questions` | `-q` | Path to a questions file (text, JSON, or JSONL) |
| `--model` | `-m` | Model ID (overrides config.ini) |
| `--base-url` | `-u` | Endpoint URL (overrides config.ini) |
| `--api-key` | `-k` | API key (overrides config.ini) |
| `--output-dir` | `-o` | Output directory (default: `results/`) |
| `--config` | `-c` | Path to config file (default: `config.ini`) |
| `--eval` | `-e` | Evaluate saved results using the eval model |
| `--eval-prompt` | | Path to eval prompt template (default: `eval-prompt.txt`) |
| `--eval-dir` | | Directory with result files to evaluate (default: `results/`) |
| `--graph` | `-g` | Generate HTML comparison graph from eval results |
| `--graph-output` | | Output path for graph HTML (default: `results/eval/comparison-graph.html`) |

### CLI Examples

```bash
# Single prompt
python benchmark.py -p "What is 2+2?" -m qwen3-8b

# Multiple questions from file
python benchmark.py -q questions.txt

# Full workflow: benchmark, evaluate, visualize
python benchmark.py -q questions.txt
python benchmark.py --eval
python benchmark.py --graph
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/config` | Get current configuration (API keys redacted) |
| PUT | `/api/config` | Update configuration |
| POST | `/api/models` | Add a new model |
| PUT | `/api/models/<id>` | Update a model |
| DELETE | `/api/models/<id>` | Delete a model |
| POST | `/api/benchmark/run` | Start a benchmark run |
| GET | `/api/benchmark/status` | Poll benchmark status and progress |
| POST | `/api/evaluation/run` | Start evaluation across all runs |
| GET | `/api/evaluation/status` | Poll evaluation status and progress |
| GET | `/api/evaluation/results/<run_id>/<model>` | Get eval results for a model in a run |
| GET | `/api/results` | List all runs with summaries |
| GET | `/api/results/<id>/questions` | List question/result pairs for a run |
| GET | `/api/results/<id>/export` | Export a run as JSON |
| DELETE | `/api/results/<id>` | Delete a run and all associated data |
| POST | `/api/graph/generate` | Generate comparison graph (body: `{ run_ids: [...] }`) |
| GET | `/api/graph` | Fetch the generated graph HTML |

## Project Structure

```
benchmark.py       Core benchmark, evaluation, and graph generation logic
server.py          Flask API server + background run threads
db.py              SQLite database layer (models, runs, results, evaluations, settings)
static/index.html  Single-page React-free web UI
requirements.txt   Python dependencies
config.ini.example Example configuration file
benchmark.db       SQLite database (created on first run)
results/           Legacy file-based output (auto-migrated to DB)
```

## Database Schema

- **settings** — key-value pairs for LLM, eval, and prompt configuration
- **models** — model definitions with optional per-model URL and API key overrides
- **runs** — benchmark runs with status, question/model counts, and timestamps
- **results** — individual Q&A results per model per run with metrics JSON
- **evaluations** — eval model assessments with scored feedback per question

## Output

### Per-Question Result Structure

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

### Comparison Graph

The comparison graph is a self-contained HTML file using Chart.js (loaded via CDN) with:

- **Grouped bar chart** — average scores per model across all criteria
- **Summary table** — numeric averages for each model and criterion
- **Collapsible per-question breakdown** — individual scores for each question, organized by model

## Token Counting

The script attempts to extract token counts from the streaming response. If the endpoint doesn't include usage data in streamed chunks, it makes a second non-streaming request to retrieve token counts. Thinking (reasoning) tokens are detected from `completion_tokens_details.reasoning_tokens` when available.
