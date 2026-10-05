# Games in Context

Code and scenarios for evaluating LLM decisions across game contexts and personas.

## Install

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate with `.venv\Scripts\activate` instead. Run the commands
below from the repository root.

## Configure models

Edit `EVAL_MODELS` in [src/config.py](src/config.py) to choose your models and
their providers, for example:

```python
EVAL_MODELS = [
    {"name": "qwen3:32b", "provider": "ollama"},
    {"name": "gemini-2.5-flash", "provider": "google"},
]
```

Models must support tool calling. For local models, keep Ollama running and
download the model first, for example `ollama pull qwen3:32b`. For API providers,
copy `.env.example` to `.env` and fill in the relevant credentials.

## Run

Evaluate the included scenarios with all configured models:

```bash
python main.py --evaluate --all-scenarios
```

To select specific models from `EVAL_MODELS`, add `--models` followed by their
names:

```bash
python main.py --evaluate --all-scenarios --models qwen3:32b gemini-2.5-flash
```

For a small check, replace `--all-scenarios` with `--limit 1` to evaluate one
scenario per file. Without either flag, the default is two scenarios per file.

After evaluation, generate plots, summaries, and regression results:

```bash
python main.py --analyze
```

Other options are listed by `python main.py --help`.

## Output

- Evaluation CSVs: `results/evaluations/results_<model>.csv`.
- Plots and analysis tables: `results/<model>/`.

Colons in model names become underscores in output paths. Repeated evaluations
append to the same CSV, so move previous results before starting a fresh run.
