# Games in Context

Benchmark for measuring how narrative context and assistant persona affect LLM
decisions in six games: Prisoner's Dilemma, Chicken, Stag Hunt, El Farol Bar,
Beauty Contest, and Traveler's Dilemma.

The repository includes the runner, analysis code, six game templates, and the
900 fixed main-benchmark scenarios. There are 30 scenarios for each of six games
and five contexts (`abstract_game_theory`, `war`, `business`, `game_show`, and
`family fight`). Each is evaluated with three personas (`generic`, `rational`,
and `emotional`), giving **2,700 decisions per model** in a full run. Binary-game
option order alternates across variations; results map choices back to their
underlying actions.

## Install

Use Python 3.10 or newer. From the root of this checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python main.py --help
```

On Windows, activate with `.venv\Scripts\activate` instead. Run all commands
below from the repository root because paths are relative to the working
directory.

## Run the fixed benchmark

Choose a model that supports tool calling. The active names and providers are
defined in `EVAL_MODELS` in [src/config.py](src/config.py). `--models` selects
from that list; add or uncomment an entry before selecting another model.
Omitting `--models` evaluates all active entries. A configured entry alone does
not guarantee that its provider supports the required tool-calling interface.

For the included [qwen3:32b](https://ollama.com/library/qwen3:32b) configuration,
install and start [Ollama](https://ollama.com/), then download the model:

```bash
ollama pull qwen3:32b
```

Keep the Ollama service running (`ollama serve` if it is not already running).
This model requires enough local memory for its weights and inference. For a
smaller local model, uncomment `qwen3:8b` in `EVAL_MODELS`, pull that model, and
use its name in the commands below. Local evaluation requires no API key.

For Google evaluation, copy `.env.example` to `.env` and set `GOOGLE_API_KEY`;
then select `gemini-2.5-flash`. The example also lists credentials for the other
supported providers. Credentials and model weights are not included.

Start with a small run (one scenario per file, 90 decisions per model):

```bash
python main.py --evaluate --models qwen3:32b --limit 1
```

For the full 900-scenario benchmark, use:

```bash
python main.py --evaluate --models qwen3:32b --all-scenarios
```

For Google, the equivalent full run is:

```bash
python main.py --evaluate --models gemini-2.5-flash --all-scenarios
```

**Archive or move the previous results CSV before starting a new run of the
same model.** Evaluation appends rows to an existing file; it does not resume
from completed scenarios or deduplicate results. This also applies when moving
from a small check to a full benchmark run.

Without `--limit` or `--all-scenarios`, the existing `TEST_FIRST_N = 2` setting
evaluates two scenarios per JSON file (180 decisions per model). `--limit`
affects evaluation only, not scenario generation. Generation temperature is
0.7, evaluation temperature is 0.0, and failed tool calls are retried up to two
times. These settings and the exact persona prompts are in `src/config.py`.
Hosted-model runs require access to the chosen model and incur provider usage.

## Results and analysis

Evaluation writes `results/evaluations/results_<model>.csv`; characters such as
`:` in model names become `_`. Each row records the scenario, game, context,
variation, persona, model, action, action label, numeric decision where relevant,
tool arguments, and reasoning. `FAILED` and `ERROR` actions identify unsuccessful
evaluations; inspect these and the logged counts before interpreting results.

After evaluation:

```bash
python main.py --analyze
```

This runs the existing plots, summaries, and regression analysis over all main
evaluation CSVs. Outputs go to `results/<model>/`. Regression alone is available
with `python main.py --regress`. Results are generated locally and ignored by
Git. Analysis applies the filtering rules in `src/analyse.py`; it does not
reproduce a separate paper-specific data-cleaning workflow.

## Generate new scenarios (optional)

The committed scenarios are ready to evaluate. To create new narratives, set
the generator provider/model in `src/config.py` and configure its credentials
(Google by default), then run:

```bash
python main.py --generate
```

This overwrites the main scenario JSON files using the included `templates/`.
Generated narratives are stochastic and can differ from the fixed benchmark;
keep a separate checkout for new generations if you need both sets. Check the
generation logs and resulting narratives for failures before evaluation.

Running `python main.py` without stage flags runs generation, evaluation, and
analysis in sequence, including overwriting the committed scenarios. Use
`--evaluate` to run the fixed benchmark directly.

## Generator-prompt sensitivity study (optional)

The runner also supports four prompt variants: `baseline`, `block_text`,
`no_numbering`, and `role_card`. Generate a study, then evaluate it separately:

```bash
python -m src.sensitivity_analysis --generate \
  --games prisoners_dilemma beauty_contest \
  --contexts war business --variations 30
python main.py --evaluate-ablation --models gemini-2.5-flash --all-scenarios
```

Use `python -m src.sensitivity_analysis --help` for generation filters and the
optional abstract control. Study scenarios go to `scenarios/sensitivity_analysis/`
and results to `results/evaluations/sensitivity_analysis/`; both are ignored by
Git. Main evaluation excludes this study directory. The standard `--analyze`
stage reads the main evaluation CSVs only.

## Contents

- `main.py`: pipeline command-line interface.
- `src/`: configuration, model adapters, generation, evaluation, and analysis.
- `templates/`: game rules and action schemas required for generation.
- `scenarios/`: 30 JSON files containing the 900 fixed main scenarios.
- `pyproject.toml`, `.env.example`: dependencies and credential configuration.

MIT license; see [LICENSE](LICENSE).

