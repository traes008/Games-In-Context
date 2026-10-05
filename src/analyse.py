"""
Results analysis and exploratory data analysis module.

Dynamically reads all evaluation CSV files from the evaluations directory,
combines them, and produces visualizations across models, games, personas,
and contexts -- driven by config.py settings.

Output layout (figures + CSVs together):
    results/<model_name>/       per-model plots & summary CSVs
    results/                    cross-model comparison plots

Usage:
    python main.py --analyze      # standard report
    python -m src.analyse         # standalone
"""

import glob
import logging
import os
import sys
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap

# Optional dependencies -- graceful fallback
try:
    from scipy import stats as _scipy_stats
except ImportError:
    _scipy_stats = None

from . import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
_PERSONA_DEFAULTS = {
    "generic": "#7fb3d8",
    "rational": "#ff9966",
    "emotional": "#77dd77",
}


def _get_persona_palette() -> dict:
    """Return a colour map keyed by persona name, auto-extending for unknowns."""
    personas = list(getattr(config, "PERSONAS", _PERSONA_DEFAULTS).keys())
    cmap = plt.colormaps["tab10"]
    palette = {}
    for i, name in enumerate(personas):
        palette[name] = _PERSONA_DEFAULTS.get(
            name, cmap(i / max(len(personas) - 1, 1))
        )
    return palette


PERSONA_PALETTE = _get_persona_palette()

_PROSOCIAL_ACTION_MAP = {
    "prisoners_dilemma": "cooperate",
    "chicken": "yield",
    "stag_hunt": "hunt_stag",
    "el_farol_bar": "stay_home",
}

_CHOICE_HEATMAP_CMAP = ListedColormap([
    "#c94c4c",
    "#f0c05a",
    "#6aa56a",
    "#d9d9d9",
])
_CHOICE_HEATMAP_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], _CHOICE_HEATMAP_CMAP.N)

# Common plot theme -- applied once
_THEME_APPLIED = False

_REQUIRED_RESULT_COLUMNS = {
    "model",
    "game_type",
    "context",
    "persona",
    "action",
}


def _apply_theme() -> None:
    """Apply a consistent, publication-ready Seaborn / matplotlib theme."""
    global _THEME_APPLIED
    if _THEME_APPLIED:
        return
    sns.set_theme(
        style="whitegrid",
        font_scale=1.15,
        rc={
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        },
    )
    _THEME_APPLIED = True


def _prepare_results_dataframe(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Validate and normalize one results DataFrame.

    Raises:
        ValueError: If required columns are missing or numeric fields are malformed.
    """
    missing_cols = sorted(_REQUIRED_RESULT_COLUMNS - set(df.columns))
    if missing_cols:
        raise ValueError(f"missing required columns: {missing_cols}")

    prepared = df.copy()

    if "decision_value" not in prepared.columns:
        logger.warning("%s has no 'decision_value' column; filling with NaN.", source_name)
        prepared["decision_value"] = np.nan

    prepared["numeric_decision"] = pd.to_numeric(
        prepared["decision_value"], errors="coerce"
    )
    prepared["is_failed"] = prepared["action"].isin(["FAILED", "ERROR"])

    if "action_label" not in prepared.columns:
        prepared["action_label"] = prepared["action"]
    else:
        prepared["action_label"] = prepared["action_label"].fillna(
            prepared["action"]
        )

    if "variation" in prepared.columns:
        variation_numeric = pd.to_numeric(prepared["variation"], errors="coerce")
        bad_variation = prepared["variation"].notna() & variation_numeric.isna()
        if bad_variation.any():
            examples = prepared.loc[bad_variation, "variation"].head(3).tolist()
            raise ValueError(
                "non-numeric values found in 'variation' "
                f"(examples: {examples})"
            )
        prepared["swapped"] = variation_numeric.fillna(0).astype(int) % 2 == 1
    else:
        prepared["swapped"] = False

    return prepared


# ===================================================================
# 1. DATA LOADING
# ===================================================================

def load_all_results(evaluations_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Discover and load every ``results_*.csv`` in *evaluations_dir*.

    Files are concatenated into a single DataFrame so that multi-model
    comparisons work automatically.

    Also supports legacy ``final_evaluation_*.csv`` for backward compat.

    Args:
        evaluations_dir: Directory to scan.  Defaults to
            ``config.EVALUATIONS_PATH``.

    Returns:
        Combined DataFrame with all evaluation results.

    Raises:
        FileNotFoundError: If no result files are found.
    """
    evaluations_dir = evaluations_dir or getattr(
        config, "EVALUATIONS_PATH", "results/evaluations"
    )

    # Try new naming first, fall back to legacy
    csv_files = sorted(glob.glob(os.path.join(evaluations_dir, "results_*.csv")))
    if not csv_files:
        legacy_dir = getattr(config, "RESULTS_PATH", "results")
        csv_files = sorted(
            glob.glob(os.path.join(legacy_dir, "final_evaluation_*.csv"))
        )

    if not csv_files:
        raise FileNotFoundError(
            f"No result CSVs found in '{evaluations_dir}'.  Run evaluation first."
        )

    frames: List[pd.DataFrame] = []
    for path in csv_files:
        try:
            raw_df = pd.read_csv(path)
            df = _prepare_results_dataframe(raw_df, os.path.basename(path))
            logger.info("Loaded %d rows from %s", len(df), os.path.basename(path))
            frames.append(df)
        except Exception as exc:
            logger.warning("Skipping %s due to invalid content: %s", path, exc)

    if not frames:
        raise FileNotFoundError("All result CSVs failed to load.")

    combined = pd.concat(frames, ignore_index=True)

    logger.info(
        "Combined dataset: %d rows, %d model(s), %d game(s)",
        len(combined),
        combined["model"].nunique(),
        combined["game_type"].nunique(),
    )
    return combined


# ===================================================================
# 2. OVERVIEW
# ===================================================================

def print_overview(df: pd.DataFrame) -> None:
    """Print a concise summary of the dataset."""
    print("\n" + "=" * 60)
    print("  DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Total rows        : {len(df)}")
    print(f"  Models            : {', '.join(df['model'].unique())}")
    print(f"  Games             : {', '.join(df['game_type'].unique())}")
    print(f"  Contexts          : {', '.join(df['context'].unique())}")
    print(f"  Personas          : {', '.join(df['persona'].unique())}")
    failures = df["is_failed"].sum()
    print(f"  Failures / Errors : {failures} ({failures / len(df) * 100:.1f}%)")
    print("=" * 60)


# ===================================================================
# 3. HELPERS: NUMERIC vs CATEGORICAL DETECTION
# ===================================================================

def _is_numeric_game(game_df: pd.DataFrame, threshold: float = 0.5) -> bool:
    """Return True when >threshold of non-failed rows have numeric values."""
    valid = game_df[~game_df["is_failed"]]
    if valid.empty:
        return False
    return valid["numeric_decision"].notna().sum() / len(valid) >= threshold


def _run_step(step_name: str, func, *args, **kwargs) -> None:
    """Run one analysis step and continue on failure."""
    try:
        func(*args, **kwargs)
    except Exception as exc:
        logger.warning("Skipping '%s' due to error: %s", step_name, exc)
        print(f"  Skipping {step_name}: {exc}")


def _ordered_present(preferred_order, values) -> list:
    """Return values ordered by preferred_order, followed by remaining values."""
    present = [value for value in preferred_order if value in values]
    remaining = sorted(value for value in values if value not in present)
    return present + remaining


def _sanitize_filename_component(value: str) -> str:
    """Convert a label into a filesystem-safe filename component."""
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _dominant_action_label(series: pd.Series) -> str:
    """Return the unique dominant action label, or MIXED/N_A when ambiguous."""
    cleaned = series.dropna()
    if cleaned.empty:
        return "N/A"
    counts = cleaned.value_counts()
    top_count = counts.iloc[0]
    top_actions = sorted(counts[counts == top_count].index.tolist())
    if len(top_actions) > 1:
        return "MIXED"
    return str(top_actions[0])


def _choice_heatmap_code(action_label: str, prosocial_action: Optional[str]) -> int:
    """Map an action label to a discrete heatmap code."""
    if action_label == "N/A":
        return 3
    if action_label == "MIXED":
        return 1
    if prosocial_action and action_label.lower() == prosocial_action.lower():
        return 2
    return 0


def _display_action_label(action_label: str) -> str:
    """Convert stored action labels into compact plot annotations."""
    return action_label.replace("_", " ").title()


def _remove_scaled_stories(df: pd.DataFrame, multiplier: float = 100.0) -> pd.DataFrame:
    """
    Detect and remove scenario-level groups whose numeric decisions are
    a large multiplicative factor larger than other scenarios within the
    same game_type. This handles cases where one story mistakenly contains
    values scaled by e.g. 1000x.

    Heuristic:
    - For each (game_type, scenario_id) compute the median of non-na
      `numeric_decision` values.
    - For each scenario, compare its median to the median of other
      scenario medians in the same game_type. If it's greater than
      `multiplier * other_median` (and there is at least one other
      non-zero median), mark it for removal.

    Returns a cleaned DataFrame and prints removed scenario ids.
    """
    if "scenario_id" not in df.columns or "numeric_decision" not in df.columns:
        return df

    cleaned = df.copy()
    removed = []

    for game in sorted(cleaned["game_type"].unique()):
        g = cleaned[cleaned["game_type"] == game]
        # compute per-scenario medians
        medians = (
            g.dropna(subset=["numeric_decision"]) 
            .groupby("scenario_id")["numeric_decision"].median()
        )
        if medians.empty or len(medians) < 2:
            continue

        for sid, sid_median in medians.items():
            others = medians.drop(sid)
            others_nonzero = others[others > 0]
            if others_nonzero.empty:
                continue
            other_median = float(others_nonzero.median())
            try:
                if sid_median >= multiplier * other_median:
                    removed.append((game, sid, float(sid_median), other_median))
            except Exception:
                continue

    if removed:
        print("\n-- Removing scaled scenario(s):")
        for game, sid, sid_med, other_med in removed:
            print(f"  {game}: {sid}  median={sid_med:.3f} vs peers_median={other_med:.3f}")
            cleaned = cleaned[cleaned["scenario_id"] != sid]
        print(f"  Removed {len(removed)} scenario(s) due to large scale multiplier > {multiplier}.\n")

    return cleaned


def _remove_out_of_range_guesses(
    df: pd.DataFrame,
    min_value: float = 0.0,
    max_value: float = 100.0,
) -> pd.DataFrame:
    """
    Remove numeric guesses outside [min_value, max_value] for selected games.

    Applied to:
    - beauty_contest
    - travelers_dilemma
    """
    target_games = {"beauty_contest", "travelers_dilemma"}

    if "game_type" not in df.columns or "numeric_decision" not in df.columns:
        return df

    out_of_range = (
        df["game_type"].isin(target_games)
        & df["numeric_decision"].notna()
        & (
            (df["numeric_decision"] < min_value)
            | (df["numeric_decision"] > max_value)
        )
    )

    removed = int(out_of_range.sum())
    if removed == 0:
        return df

    by_game = (
        df.loc[out_of_range, "game_type"]
        .value_counts()
        .to_dict()
    )
    logger.info(
        "Removed %d out-of-range guesses outside [%s, %s] for %s",
        removed,
        min_value,
        max_value,
        by_game,
    )
    print(
        "  Removed "
        f"{removed} out-of-range guesses outside [{min_value:.0f}, {max_value:.0f}] "
        f"for beauty_contest/travelers_dilemma: {by_game}"
    )

    return df.loc[~out_of_range].copy()


# ===================================================================
# 4. PER-GAME PLOTTING HELPERS
# ===================================================================

def _plot_numeric_game(
    game_df: pd.DataFrame, game_title: str, axes: list
) -> None:
    """Top bar distribution (0/1..100) + bottom boxplot for numeric games."""
    ax = axes[0]
    numeric_vals = game_df["numeric_decision"].dropna()
    if numeric_vals.empty:
        ax.text(0.5, 0.5, "No numeric decisions", ha="center", va="center")
        ax.set_title(f"{game_title} -- Value Distribution (N/A)")
        ax.set_axis_off()
    else:
        # Keep a full k-level axis (1..100 or 0..100) so every possible
        # number has an explicit bar, even if count is zero.
        rounded_vals = numeric_vals.round().astype(int)
        lower_bound = 0 if rounded_vals.min() <= 0 else 1
        full_range = pd.Index(range(lower_bound, 101), name="number")
        value_counts = (
            rounded_vals.value_counts().reindex(full_range, fill_value=0).sort_index()
        )

        value_counts.plot.bar(ax=ax, color="#7fb3d8", width=0.9)
        ax.set_title(f"{game_title} -- Value Distribution (full 1..100 scale)")
        ax.set_ylabel("Count")
        ax.set_xlabel("Number")
        tick_positions = list(range(len(value_counts)))
        tick_labels = [
            str(val) if (val % 5 == 0 or val in (lower_bound, 100)) else ""
            for val in value_counts.index
        ]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0)

    ax = axes[1]
    sns.boxplot(
        x="context", y="numeric_decision", data=game_df, ax=ax,
        hue="context", palette="Set3", legend=False,
    )
    sns.stripplot(
        x="context", y="numeric_decision", data=game_df, ax=ax,
        color="black", alpha=0.25, jitter=True, size=2,
    )
    ax.set_title(f"{game_title} -- Boxplot by Context")
    ax.set_ylabel("Numeric Value")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)

    print(f"\n-- {game_title} (numeric / k-level) --")
    print(
        game_df.groupby(["context", "persona"])["numeric_decision"]
        .agg(["mean", "median", "std", "min", "max"])
        .round(2)
        .to_string()
    )


def _plot_categorical_game(
    game_df: pd.DataFrame, game_title: str, axes: list
) -> None:
    """Stacked-bar plots for a categorical (2-action) game."""
    unique_actions = sorted(game_df["action_label"].unique())
    cmap = plt.colormaps["tab20"]
    colours = [
        cmap(i / max(len(unique_actions) - 1, 1))
        for i in range(len(unique_actions))
    ]

    ax = axes[0]
    ct = (
        game_df.groupby(["context", "action_label"]).size().unstack(fill_value=0)
    )
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot.bar(stacked=True, ax=ax, color=colours[: ct_pct.shape[1]])
    ax.set_title(f"{game_title} -- Actions by Context")
    ax.set_ylabel("Percentage")
    ax.set_xlabel("")
    ax.legend(title="Action", fontsize=8)
    ax.tick_params(axis="x", rotation=25)

    ax = axes[1]
    pt = (
        game_df.groupby(["persona", "action_label"]).size().unstack(fill_value=0)
    )
    pt_pct = pt.div(pt.sum(axis=1), axis=0) * 100
    pt_pct.plot.bar(stacked=True, ax=ax, color=colours[: pt_pct.shape[1]])
    ax.set_title(f"{game_title} -- Actions by Persona")
    ax.set_ylabel("Percentage")
    ax.set_xlabel("")
    ax.legend(title="Action", fontsize=8)

    print(f"\n-- {game_title} (categorical) --")
    print(
        game_df.groupby(["context", "persona"])["action_label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .rename("pct")
        .to_string()
    )


# ===================================================================
# 5. CROSS-CUTTING EDA PLOTS
# ===================================================================

def plot_persona_effect(df: pd.DataFrame, save_dir: str) -> None:
    """Action distribution and numeric decisions by persona."""
    valid = df[~df["is_failed"]].copy()

    if valid.empty:
        print("  Skipping persona effect plot (no valid non-failed rows).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Persona Effect Across Games", fontsize=14)

    ax = axes[0]
    ct = valid.groupby(["persona", "action_label"]).size().unstack(fill_value=0)
    row_totals = ct.sum(axis=1)
    ct = ct.loc[row_totals > 0]
    if ct.empty:
        ax.text(0.5, 0.5, "No action data", ha="center", va="center")
        ax.set_title("Action Distribution by Persona -- N/A")
        ax.set_axis_off()
    else:
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        ct_pct.plot.bar(stacked=True, ax=ax, colormap="tab20")
        ax.set_title("Action Distribution by Persona")
        ax.set_ylabel("Percentage")
        ax.set_xlabel("")
        ax.legend(
            title="Action", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7
        )

    ax = axes[1]
    numeric = valid.dropna(subset=["numeric_decision"]).copy()
    if not numeric.empty:
        sns.boxplot(
            x="persona", y="numeric_decision", data=numeric, ax=ax,
            hue="persona", palette=PERSONA_PALETTE, legend=False,
        )
        ax.set_title("Numeric Decisions by Persona")
        ax.set_ylabel("Numeric Decision")
    else:
        ax.text(0.5, 0.5, "No numeric decisions", ha="center", va="center")
        ax.set_title("Numeric Decisions -- N/A")
    ax.set_xlabel("")

    plt.tight_layout()
    path = os.path.join(save_dir, "persona_effect.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_context_effect(df: pd.DataFrame, save_dir: str) -> None:
    """Action distribution and numeric decisions by context."""
    valid = df[~df["is_failed"]].copy()

    if valid.empty:
        print("  Skipping context effect plot (no valid non-failed rows).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Context / Framing Effect Across Games", fontsize=14)

    ax = axes[0]
    ct = valid.groupby(["context", "action_label"]).size().unstack(fill_value=0)
    row_totals = ct.sum(axis=1)
    ct = ct.loc[row_totals > 0]
    if ct.empty:
        ax.text(0.5, 0.5, "No action data", ha="center", va="center")
        ax.set_title("Action Distribution by Context -- N/A")
        ax.set_axis_off()
    else:
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        ct_pct.plot.bar(stacked=True, ax=ax, colormap="tab20")
        ax.set_title("Action Distribution by Context")
        ax.set_ylabel("Percentage")
        ax.set_xlabel("")
        ax.legend(
            title="Action", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7
        )
        ax.tick_params(axis="x", rotation=25)

    ax = axes[1]
    numeric = valid.dropna(subset=["numeric_decision"]).copy()
    if not numeric.empty:
        sns.boxplot(
            x="context", y="numeric_decision", data=numeric, ax=ax,
            hue="context", palette="Set3", legend=False,
        )
        ax.set_title("Numeric Decisions by Context")
        ax.set_ylabel("Numeric Decision")
    else:
        ax.text(0.5, 0.5, "No numeric decisions", ha="center", va="center")
        ax.set_title("Numeric Decisions -- N/A")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    path = os.path.join(save_dir, "context_effect.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_model_comparison(df: pd.DataFrame, save_dir: str) -> None:
    """Compare behaviour across models (only meaningful with 2+ models)."""
    models = df["model"].unique()
    if len(models) < 2:
        print("  Skipping model comparison (only 1 model in data).")
        return

    valid = df[~df["is_failed"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Comparison", fontsize=14)

    ax = axes[0]
    ct = valid.groupby(["model", "action_label"]).size().unstack(fill_value=0)
    row_totals = ct.sum(axis=1)
    ct = ct.loc[row_totals > 0]
    if ct.empty:
        ax.text(0.5, 0.5, "No action data", ha="center", va="center")
        ax.set_title("Action Distribution by Model -- N/A")
        ax.set_axis_off()
    else:
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        ct_pct.plot.bar(stacked=True, ax=ax, colormap="tab20")
        ax.set_title("Action Distribution by Model")
        ax.set_ylabel("Percentage")
        ax.set_xlabel("")
        ax.legend(
            title="Action", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7
        )

    ax = axes[1]
    numeric = valid.dropna(subset=["numeric_decision"]).copy()
    if not numeric.empty:
        sns.boxplot(
            x="model", y="numeric_decision", data=numeric, ax=ax,
            hue="model", legend=False,
        )
        ax.set_title("Numeric Decisions by Model")
        ax.set_ylabel("Numeric Decision")
    else:
        ax.text(0.5, 0.5, "No numeric decisions", ha="center", va="center")
        ax.set_title("Numeric Decisions -- N/A")
    ax.set_xlabel("")

    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_failure_overview(df: pd.DataFrame, save_dir: str) -> None:
    """Show success / failure / error rates."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Response Quality", fontsize=14)

    ax = axes[0]
    status = df["action"].apply(
        lambda a: (
            "Error" if a == "ERROR" else ("Failed" if a == "FAILED" else "Success")
        )
    )
    counts = status.value_counts()
    ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=["#aec7e8", "#ff9896", "#ffcc99"][: len(counts)],
        startangle=90,
    )
    ax.set_title("Overall Response Status")

    ax = axes[1]
    df_tmp = df.copy()
    df_tmp["status"] = status
    ct = df_tmp.groupby(["game_type", "status"]).size().unstack(fill_value=0)
    ct.plot.bar(stacked=True, ax=ax, color=["#ff9896", "#ffcc99", "#aec7e8"])
    ax.set_title("Response Status by Game")
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Status")

    plt.tight_layout()
    path = os.path.join(save_dir, "failure_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# 6. GAME-SPECIFIC DASHBOARDS
# ===================================================================

def plot_game_dashboards(df: pd.DataFrame, save_dir: str) -> None:
    """Two-panel dashboard per game (auto-detects numeric vs categorical)."""
    for game in sorted(df["game_type"].unique()):
        game_df = df[(df["game_type"] == game) & ~df["is_failed"]].copy()
        game_title = game.replace("_", " ").title()

        if game_df.empty:
            logger.warning("No valid data for %s, skipping dashboard.", game)
            continue

        if _is_numeric_game(game_df):
            fig, axes = plt.subplots(2, 1, figsize=(14, 9))
            fig.suptitle(game_title, fontsize=14)
            _plot_numeric_game(game_df, game_title, [axes[0], axes[1]])
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(game_title, fontsize=14)
            _plot_categorical_game(game_df, game_title, [axes[0], axes[1]])

        plt.tight_layout()
        path = os.path.join(save_dir, f"game_{game}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ===================================================================
# 7. HEATMAP: PERSONA x CONTEXT
# ===================================================================

def plot_persona_context_heatmap(
    df: pd.DataFrame,
    save_dir: str,
    model_label: Optional[str] = None,
) -> None:
    """Save one persona x context chosen-option heatmap per game."""
    valid = df[~df["is_failed"]].copy()
    games = sorted(valid["game_type"].unique())
    if not games:
        return

    if model_label is None:
        raw_model_label = str(valid["model"].iloc[0]) if not valid.empty else "unknown_model"
        model_label = _sanitize_filename_component(raw_model_label)

    persona_order = _ordered_present(config.PERSONAS.keys(), valid["persona"].dropna().unique())
    context_order = _ordered_present(config.SCENARIO_CONTEXTS, valid["context"].dropna().unique())

    for game in games:
        g = valid[valid["game_type"] == game]
        if g.empty:
            continue

        fig, ax = plt.subplots(
            figsize=(max(6.5, 1.3 * len(context_order) + 2.5), max(3.8, 0.9 * len(persona_order) + 2.0))
        )

        if g["numeric_decision"].notna().sum() > 0.5 * len(g):
            pivot = g.pivot_table(
                values="numeric_decision",
                index="persona",
                columns="context",
                aggfunc="mean",
            ).reindex(index=persona_order, columns=context_order).round(1)
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".1f",
                cmap="Blues",
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"label": "Mean choice"},
                ax=ax,
            )
        else:
            mode_table = (
                g.groupby(["persona", "context"])["action_label"]
                .apply(_dominant_action_label)
                .unstack(fill_value="N/A")
                .reindex(index=persona_order, columns=context_order, fill_value="N/A")
            )
            prosocial_action = _PROSOCIAL_ACTION_MAP.get(game)
            numeric_pivot = mode_table.apply(
                lambda column: column.map(
                    lambda action: _choice_heatmap_code(str(action), prosocial_action)
                )
            )
            annotations = mode_table.apply(
                lambda column: column.map(lambda action: _display_action_label(str(action)))
            )
            sns.heatmap(
                numeric_pivot,
                annot=annotations.values,
                fmt="",
                cmap=_CHOICE_HEATMAP_CMAP,
                norm=_CHOICE_HEATMAP_NORM,
                ax=ax,
                cbar=False,
                linewidths=0.5,
                linecolor="white",
            )

        ax.set_title(
            f"{game.replace('_', ' ').title()} | {model_label.replace('_', ' ')}",
            fontsize=11,
        )
        ax.set_xlabel("Context")
        ax.set_ylabel("Persona")
        ax.tick_params(axis="x", rotation=25)
        plt.tight_layout()

        safe_game = _sanitize_filename_component(game)
        path = os.path.join(save_dir, f"heatmap_chosen_options_{model_label}_{safe_game}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ===================================================================
# 8. STATISTICAL TESTS
# ===================================================================

def run_statistical_tests(
    df: pd.DataFrame, save_dir: str
) -> pd.DataFrame:
    """
    Per-game significance tests.

    * Categorical games  -> Chi-square of independence
    * Numeric games      -> Kruskal-Wallis H-test

    Tested across **contexts** and **personas** separately.
    Returns + saves a DataFrame of results.
    """
    if _scipy_stats is None:
        print("  Skipping statistical tests (install 'scipy' to enable).")
        return pd.DataFrame()

    valid = df[~df["is_failed"]].copy()
    rows: list[dict] = []

    for game in sorted(valid["game_type"].unique()):
        g = valid[valid["game_type"] == game]
        title = game.replace("_", " ").title()

        if _is_numeric_game(g):
            # Kruskal-Wallis across contexts
            groups = [
                grp["numeric_decision"].dropna().values
                for _, grp in g.groupby("context")
            ]
            groups = [gr for gr in groups if len(gr) > 0]
            if len(groups) >= 2:
                try:
                    stat, p = _scipy_stats.kruskal(*groups)
                    rows.append(dict(
                        game=title, test="Kruskal-Wallis", factor="context",
                        statistic=round(stat, 4), p_value=round(p, 6),
                        significant_0_05=p < 0.05,
                    ))
                except ValueError as e:
                    logger.warning("Kruskal-Wallis failed for %s (context): %s", title, e)

            # Kruskal-Wallis across personas
            groups = [
                grp["numeric_decision"].dropna().values
                for _, grp in g.groupby("persona")
            ]
            groups = [gr for gr in groups if len(gr) > 0]
            if len(groups) >= 2:
                try:
                    stat, p = _scipy_stats.kruskal(*groups)
                    rows.append(dict(
                        game=title, test="Kruskal-Wallis", factor="persona",
                        statistic=round(stat, 4), p_value=round(p, 6),
                        significant_0_05=p < 0.05,
                    ))
                except ValueError as e:
                    logger.warning("Kruskal-Wallis failed for %s (persona): %s", title, e)
        else:
            # Chi-square across contexts
            ct = pd.crosstab(g["context"], g["action_label"])
            if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                chi2, p, _dof, _ = _scipy_stats.chi2_contingency(ct)
                rows.append(dict(
                    game=title, test="Chi-square", factor="context",
                    statistic=round(chi2, 4), p_value=round(p, 6),
                    significant_0_05=p < 0.05,
                ))

            # Chi-square across personas
            ct = pd.crosstab(g["persona"], g["action_label"])
            if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                chi2, p, _dof, _ = _scipy_stats.chi2_contingency(ct)
                rows.append(dict(
                    game=title, test="Chi-square", factor="persona",
                    statistic=round(chi2, 4), p_value=round(p, 6),
                    significant_0_05=p < 0.05,
                ))

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        path = os.path.join(save_dir, "statistical_tests.csv")
        results_df.to_csv(path, index=False)
        print(f"  Saved: {path}")
        print("\n-- Statistical Tests --")
        print(results_df.to_string(index=False))
    else:
        print("  No statistical tests could be performed (insufficient data).")

    return results_df


# ===================================================================
# 9. SWAP-BIAS ANALYSIS
# ===================================================================

def plot_swap_bias(df: pd.DataFrame, save_dir: str) -> None:
    """
    Positional-bias analysis: compare swapped vs non-swapped variations.

    * Categorical games -> bar chart of Option-A choice rate
    * Numeric games     -> boxplot of values normal vs swapped
    """
    valid = df[~df["is_failed"]].copy()
    if "swapped" not in valid.columns:
        print("  Skipping swap-bias analysis (no swap info in data).")
        return

    cat_games = [
        g for g in valid["game_type"].unique()
        if not _is_numeric_game(valid[valid["game_type"] == g])
    ]
    num_games = [
        g for g in valid["game_type"].unique()
        if _is_numeric_game(valid[valid["game_type"] == g])
    ]

    total_panels = len(cat_games) + len(num_games)
    if total_panels == 0:
        print("  Skipping swap-bias plot (no valid game data).")
        return

    cols = min(total_panels, 3)
    rows_needed = (total_panels + cols - 1) // cols
    fig, axes = plt.subplots(
        rows_needed, cols, figsize=(6 * cols, 5 * rows_needed)
    )
    if total_panels == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).flatten()
    fig.suptitle("Positional Bias -- Normal vs Swapped", fontsize=14, y=1.02)

    idx = 0

    # -- Categorical: Option-A choice rate --
    for game in sorted(cat_games):
        g = valid[valid["game_type"] == game]
        ax = axes[idx]
        idx += 1

        option_a_rate = (
            g.groupby("swapped")
            .apply(
                lambda x: (x["action"] == "action_a").mean() * 100,
                include_groups=False,
            )
        )
        option_a_rate.index = option_a_rate.index.map(
            {False: "Normal", True: "Swapped"}
        )
        option_a_rate.plot.bar(ax=ax, color=["#7fb3d8", "#ff9966"])
        ax.set_title(game.replace("_", " ").title())
        ax.set_ylabel("Option A chosen (%)")
        ax.set_xlabel("")
        ax.set_ylim(0, 100)
        ax.axhline(50, ls="--", color="grey", alpha=0.5)
        ax.tick_params(axis="x", rotation=0)

    # -- Numeric: value distributions --
    for game in sorted(num_games):
        g = valid[valid["game_type"] == game]
        ax = axes[idx]
        idx += 1

        g_plot = g.copy()
        g_plot["Order"] = g_plot["swapped"].map(
            {False: "Normal", True: "Swapped"}
        )
        sns.boxplot(
            x="Order", y="numeric_decision", data=g_plot, ax=ax,
            hue="Order", palette=["#7fb3d8", "#ff9966"], legend=False,
        )
        ax.set_title(game.replace("_", " ").title())
        ax.set_ylabel("Numeric Value")
        ax.set_xlabel("")

    # Hide unused axes
    for i in range(idx, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, "swap_bias.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # -- Console: stat tests for bias --
    if _scipy_stats is not None:
        print("\n-- Swap-Bias Tests --")
        for game in sorted(cat_games):
            g = valid[valid["game_type"] == game]
            ct = pd.crosstab(g["swapped"], g["action_label"])
            if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                chi2, p, _, _ = _scipy_stats.chi2_contingency(ct)
                sig = "*" if p < 0.05 else ""
                print(f"  {game:30s}  Chi2={chi2:.3f}  p={p:.4f} {sig}")

        for game in sorted(num_games):
            g = valid[valid["game_type"] == game]
            normal = g.loc[~g["swapped"], "numeric_decision"].dropna()
            swapped_vals = g.loc[g["swapped"], "numeric_decision"].dropna()
            if len(normal) > 0 and len(swapped_vals) > 0:
                try:
                    stat, p = _scipy_stats.mannwhitneyu(
                        normal, swapped_vals, alternative="two-sided"
                    )
                    sig = "*" if p < 0.05 else ""
                    print(f"  {game:30s}  U={stat:.1f}  p={p:.4f} {sig}")
                except ValueError as e:
                    print(f"  {game:30s}  U=N/A   p=N/A   (Test failed: {e})")


# ===================================================================
# 10. COOPERATION RATE PLOT
# ===================================================================

def plot_cooperation_rate(df: pd.DataFrame, save_dir: str) -> None:
    """
    Cooperation / prosocial action rate across all categorical games,
    broken down by context.
    """
    valid = df[~df["is_failed"]].copy()

    cat_games = [
        g for g in sorted(valid["game_type"].unique())
        if not _is_numeric_game(valid[valid["game_type"] == g])
    ]
    if not cat_games:
        print("  Skipping cooperation rate plot (no categorical games).")
        return

    _PROSOCIAL = {
        "cooperate",     # prisoners_dilemma
        "swerve",        # chicken (alternative framing)
        "yield",         # chicken
        "stag",          # stag_hunt (alternative framing)
        "hunt_stag",     # stag_hunt
        "stay_home",     # el_farol_bar
        "stay home",     # el_farol_bar (with space)
    }

    rows: list[dict] = []
    for game in cat_games:
        g = valid[valid["game_type"] == game]
        labels = sorted(g["action_label"].unique())
        prosocial = next(
            (lab for lab in labels if lab.lower() in _PROSOCIAL), None
        )
        
        if not prosocial:
            logger.warning("No prosocial action found for '%s'. Skipping cooperation plot for this game.", game)
            continue

        for ctx in sorted(g["context"].unique()):
            gc = g[g["context"] == ctx]
            if len(gc) > 0:
                rate = (gc["action_label"] == prosocial).mean() * 100
                rows.append({
                    "game": game.replace("_", " ").title(),
                    "context": ctx,
                    "cooperation_rate": rate,
                    "prosocial_action": prosocial,
                })

    if not rows:
        print("  Skipping cooperation rate plot (no prosocial data matched).")
        return

    rate_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(cat_games)), 5))
    sns.barplot(
        data=rate_df, x="game", y="cooperation_rate", hue="context",
        ax=ax, palette="Set2",
    )
    ax.set_title(
        "Cooperation / Prosocial Action Rate by Game & Context", fontsize=13
    )
    ax.set_ylabel("Rate (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 105)
    ax.axhline(50, ls="--", color="grey", alpha=0.5)
    ax.legend(
        title="Context", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8
    )
    ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    path = os.path.join(save_dir, "cooperation_rate.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# 11. REASONING WORD CLOUD
# ===================================================================

def plot_reasoning_wordcloud(df: pd.DataFrame, save_dir: str) -> None:
    """Per-persona word cloud from the reasoning column (optional dep)."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("  Skipping word cloud (install 'wordcloud' package to enable).")
        return

    valid = df[~df["is_failed"]].copy()
    personas = sorted(valid["persona"].unique())
    n = len(personas)
    if n == 0 or "reasoning" not in valid.columns:
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Reasoning Word Clouds by Persona", fontsize=14)

    _STOPWORDS = {
        "the", "a", "an", "is", "it", "to", "and", "of", "in", "for",
        "this", "that", "with", "on", "as", "my", "i", "we", "you",
        "option", "action", "choose", "decision", "reasoning", "step",
        "therefore", "since", "because", "however", "also", "would",
        "could", "should", "will", "one", "two", "each", "both", "other",
        "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "but", "or", "if", "not", "no", "so", "than",
        "too", "very", "just", "about", "up", "out", "all", "its",
        "from", "by", "at", "they", "their", "them", "which", "what",
        "when", "where", "how", "who", "more", "some", "any", "our",
        "your", "there", "here", "then", "only", "into", "over", "such",
        "can", "may", "most", "own", "same", "make", "made", "get",
    }

    for ax, persona in zip(axes, personas):
        text = " ".join(
            valid.loc[valid["persona"] == persona, "reasoning"]
            .dropna()
            .astype(str)
            .values
        )
        if not text.strip():
            ax.text(0.5, 0.5, "No reasoning data", ha="center", va="center")
            ax.set_title(persona)
            ax.axis("off")
            continue

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=120,
            stopwords=_STOPWORDS,
            colormap="viridis",
        ).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(persona, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(save_dir, "reasoning_wordcloud.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# 12. SUMMARY TABLE EXPORT
# ===================================================================

def export_summary_tables(df: pd.DataFrame, save_dir: str) -> None:
    """
    Export summary statistics as CSV tables:

    * summary_by_game_context.csv
    * summary_by_game_persona.csv
    * summary_overall.csv
    """
    valid = df[~df["is_failed"]].copy()

    # -- Per game x context --
    rows: list[dict] = []
    for game in sorted(valid["game_type"].unique()):
        g = valid[valid["game_type"] == game]
        is_num = _is_numeric_game(g)
        for ctx in sorted(g["context"].unique()):
            gc = g[g["context"] == ctx]
            row: dict = {"game_type": game, "context": ctx, "n": len(gc)}
            if is_num:
                row["mean"] = gc["numeric_decision"].mean()
                row["median"] = gc["numeric_decision"].median()
                row["std"] = gc["numeric_decision"].std()
            else:
                for label in sorted(gc["action_label"].unique()):
                    row[f"pct_{label}"] = (
                        (gc["action_label"] == label).mean() * 100
                    )
            rows.append(row)

    path = os.path.join(save_dir, "summary_by_game_context.csv")
    pd.DataFrame(rows).round(2).to_csv(path, index=False)
    print(f"  Saved: {path}")

    # -- Per game x persona --
    rows = []
    for game in sorted(valid["game_type"].unique()):
        g = valid[valid["game_type"] == game]
        is_num = _is_numeric_game(g)
        for persona in sorted(g["persona"].unique()):
            gp = g[g["persona"] == persona]
            row = {"game_type": game, "persona": persona, "n": len(gp)}
            if is_num:
                row["mean"] = gp["numeric_decision"].mean()
                row["median"] = gp["numeric_decision"].median()
                row["std"] = gp["numeric_decision"].std()
            else:
                for label in sorted(gp["action_label"].unique()):
                    row[f"pct_{label}"] = (
                        (gp["action_label"] == label).mean() * 100
                    )
            rows.append(row)

    path = os.path.join(save_dir, "summary_by_game_persona.csv")
    pd.DataFrame(rows).round(2).to_csv(path, index=False)
    print(f"  Saved: {path}")

    # -- Overall summary (iterates over full df to capture 100% failure games) --
    rows = []
    for game in sorted(df["game_type"].unique()):
        g_all = df[df["game_type"] == game]
        g_valid = valid[valid["game_type"] == game]
        is_num = _is_numeric_game(g_valid) if not g_valid.empty else False
        
        row = {
            "game_type": game,
            "n": len(g_all),
            "n_contexts": g_all["context"].nunique(),
            "n_personas": g_all["persona"].nunique(),
            "failure_rate": (g_all["is_failed"].mean() * 100),
        }
        
        if is_num and not g_valid.empty:
            row["mean"] = g_valid["numeric_decision"].mean()
            row["median"] = g_valid["numeric_decision"].median()
            row["std"] = g_valid["numeric_decision"].std()
        elif not g_valid.empty:
            top = g_valid["action_label"].mode().iloc[0] if not g_valid.empty else "N/A"
            row["dominant_action"] = top
            row[f"pct_{top}"] = (g_valid["action_label"] == top).mean() * 100
            
        rows.append(row)

    path = os.path.join(save_dir, "summary_overall.csv")
    pd.DataFrame(rows).round(2).to_csv(path, index=False)
    print(f"  Saved: {path}")


# ===================================================================
# 13. ORCHESTRATION
# ===================================================================

def _run_analysis_for_subset(
    df: pd.DataFrame, save_dir: str, label: str
) -> None:
    """Run the per-model chosen-option heatmap export."""
    os.makedirs(save_dir, exist_ok=True)

    print_overview(df)

    print(f"\n-- Chosen Option Heatmaps ({label}) --")
    _run_step("persona-context heatmap", plot_persona_context_heatmap, df, save_dir, label)


def analyse(
    file_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> None:
    """
    Run the full EDA pipeline.

    If *file_path* is given only that CSV is loaded; otherwise all
    ``results_*.csv`` files in the evaluations directory are read.

    **Output layout** (figures + summary CSVs side by side):
        ``results/<model_name>/``   -- per-model outputs
        ``results/``                -- cross-model comparison plots

    Args:
        file_path:  Optional single CSV to analyse.
        model_name: Unused -- kept for backward compat with main.py.
    """
    results_dir = getattr(config, "RESULTS_PATH", "results")
    os.makedirs(results_dir, exist_ok=True)

    _apply_theme()

    # -- Load data --
    if file_path:
        try:
            raw_df = pd.read_csv(file_path)
            df = _prepare_results_dataframe(raw_df, os.path.basename(file_path))
            logger.info("Loaded %d rows from %s", len(df), file_path)
        except Exception as exc:
            logger.error("Cannot read %s: %s", file_path, exc)
            print(f"Error: {exc}")
            sys.exit(1)
    else:
        df = load_all_results()
    # Remove any scenario-level numeric scale outliers (e.g., stories with
    # values accidentally scaled by 1000x). Default multiplier=100.
    df = _remove_scaled_stories(df)
    # Remove invalid guesses for bounded games.
    df = _remove_out_of_range_guesses(df, min_value=0.0, max_value=100.0)

    # -- Per-model analysis -> results/<model_name>/ --
    models = sorted(df["model"].unique())
    for original_mdl in models:
        safe_mdl = original_mdl
        if ":" in original_mdl:
            logger.warning(
                "Model name '%s' contains ':', which may cause filesystem "
                "issues. Replacing with '_'.", original_mdl
            )
            safe_mdl = original_mdl.replace(":", "_")  # Sanitize model name for filesystem
            
        model_dir = os.path.join(results_dir, safe_mdl)
        model_df = df[df["model"] == original_mdl].copy()
        print("\n" + "#" * 60)
        print(f"  MODEL: {original_mdl}")
        print("#" * 60)
        _run_analysis_for_subset(model_df, model_dir, label=safe_mdl)
        print(f"\n  Model outputs saved to: {model_dir}/")

    print("\n" + "=" * 60)
    print(f"  All outputs saved to: {results_dir}/")
    print("=" * 60)


# Keep backward-compatible alias used by main.py / interpret_results
analyze_game_theory_results = analyse


# ===================================================================
# Standalone execution
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    analyse()