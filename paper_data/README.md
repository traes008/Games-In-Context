# Data used in the paper

This directory contains the curated analysis data behind the paper's reported
results. It deliberately excludes unrelated exploratory runs, superseded
cleaned data, provider logs, model reasoning text, and repair/debug artifacts.

## Analysis datasets

- `main_evaluations.csv` contains the 21,519 valid evaluation records used for
  the main analysis. It covers the eight retained models, six games, five
  contexts, and three personas. The reconstruction excludes 38 logged failures
  and 43 invalid numerical submissions from the 21,600 planned evaluations.
- `llama1b_binary_evaluations.csv` contains only the 1,800 valid binary-game
  records for the excluded Llama 3.2 1B model. These records are included solely
  because they underlie that model's position-bias result; they are not part of
  the eight-model main analysis.
- `generator_prompt_sensitivity.csv` contains the separate 960-record
  exploratory generator-prompt sensitivity analysis. It must not be combined
  with `main_evaluations.csv`.
- `screening_summary.csv` contains the screening counts and exclusion reasons
  for all 13 screened models.

Each row in `main_evaluations.csv` represents one valid model--scenario--persona
evaluation. Binary choices are mapped from the presented option back to their
semantic action using the scenario metadata. `is_prosocial` is the paper's
operational binary indicator. Numerical-game rows instead use
`numeric_decision`. Scenario IDs refer to the fixed records in `../scenarios/`.

## Reported outputs

- `statistical_tests.csv` gives the pooled and per-model context/persona tests
  reported in the paper and appendix, including raw and adjusted p-values.
- `generator_prompt_sensitivity_tests.csv` gives the two tests for the separate
  sensitivity analysis.
- `position_bias_summary.csv` gives the standard/reversed option counts and PBI
  values used for model screening.
- `reported_metrics.csv` lists the calculated values used in the paper's tables
  and numerical statements.
- `manifest.json` records row counts, byte sizes, and SHA-256 hashes for all CSV
  files in this directory.

The statistical calculations treat evaluation records as the unit of analysis.
Because vignettes are reused across models and personas, the resulting tests do
not by themselves establish independence of all observations; this limitation
is also stated in the paper.
