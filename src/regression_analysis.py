"""
Regression and advanced statistical analysis for game-theoretic benchmark.

This module provides thesis-grade statistical analysis beyond the exploratory
plots in analyse.py.  It produces:

1. **Logistic regression** (categorical games): persona × context interaction
   with odds ratios, significance, and pseudo-R².
2. **OLS regression** (numeric / k-level games): persona × context interaction
   with ANOVA decomposition.
3. **Effect sizes**: Cramér's V (categorical) and eta-squared (numeric) for
   every game × factor combination.
4. **Multiple-comparison correction**: Holm–Bonferroni adjustment across all
   tests to control family-wise error rate.
5. **Post-hoc pairwise tests**: Which specific contexts / personas differ from
   each other (Tukey-style for numeric, pairwise chi-square for categorical).
6. **Variance partitioning**: How much of the behavioural variance is
   attributable to context, persona, and their interaction.
7. **Reasoning language analysis**: Keyword-frequency profiles per persona.
8. **Persona consistency**: Shannon entropy per persona × context cell.

All results are saved as CSV tables and a consolidated console report is
printed.

Usage:
    python -m src.regression_analysis          # standalone
    python main.py --regress                   # if wired into the pipeline

Requires:
    statsmodels, scipy, pandas, numpy  (all already in the project env)
"""

import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.multitest import multipletests

    _HAS_SM = True
except ImportError:
    sm = None
    smf = None
    pairwise_tukeyhsd = None
    multipletests = None
    _HAS_SM = False

from . import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Map game type -> prosocial / cooperative action label
_PROSOCIAL_MAP: Dict[str, str] = {
    "prisoners_dilemma": "cooperate",
    "chicken": "yield",
    "stag_hunt": "hunt_stag",
    "el_farol_bar": "stay_home",
}

_CATEGORICAL_GAMES = tuple(_PROSOCIAL_MAP.keys())


# ===================================================================
# 0. DATA LOADING  (re-uses analyse.load_all_results when available)
# ===================================================================

def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load and prepare the evaluation results DataFrame."""
    from . import analyse

    df = analyse.load_all_results(path)
    # Derived columns (idempotent — analyse may have added some already)
    df["numeric_decision"] = pd.to_numeric(df["decision_value"], errors="coerce")
    df["is_failed"] = df["action"].isin(["FAILED", "ERROR"])
    if "action_label" not in df.columns:
        df["action_label"] = df["action"]
    else:
        df["action_label"] = df["action_label"].fillna(df["action"])
    if "variation" in df.columns:
        df["swapped"] = df["variation"].astype(int) % 2 == 1
    else:
        df["swapped"] = False
    return df


def _is_numeric_game(game_df: pd.DataFrame, threshold: float = 0.5) -> bool:
    valid = game_df[~game_df["is_failed"]]
    if valid.empty:
        return False
    return valid["numeric_decision"].notna().sum() / len(valid) >= threshold


# ===================================================================
# 1. LOGISTIC REGRESSION  (categorical games)
# ===================================================================


def logistic_regression(
    df: pd.DataFrame, save_dir: str
) -> pd.DataFrame:
    """
    Fit logistic regression for each categorical game:
        prosocial ~ C(context) * C(persona)

    Reports odds ratios, 95% CIs, p-values, and McFadden pseudo-R².
    Falls back to L2 regularization if MLE fails due to perfect separation.
    """
    if not _HAS_SM:
        print("  [skip] statsmodels not installed — skipping logistic regression.")
        return pd.DataFrame()

    valid = df[~df["is_failed"]].copy()
    all_rows: List[dict] = []
    full_summaries: List[str] = []
    matrix_rows: List[dict] = []

    for game, prosocial in _PROSOCIAL_MAP.items():
        g = valid[valid["game_type"] == game].copy()
        if g.empty:
            continue

        g["y"] = (g["action_label"] == prosocial).astype(int)

        # Skip if outcome is constant
        if g["y"].nunique() < 2:
            print(f"  {game}: skipped (constant outcome)")
            continue

        formula = "y ~ C(context, Treatment('abstract_game_theory')) * C(persona, Treatment('generic'))"
        model = None
        method_used = "MLE"

        # 1. Try standard MLE first
        try:
            model = smf.logit(formula, data=g).fit(disp=0, maxiter=200)
        except Exception:
            pass

        if model is None or not model.mle_retvals.get("converged", False):
            # 2. Fall back to L1-penalized logistic regression.
            # This is more robust than the unregularized fit when the data
            # are close to separation or the Hessian becomes singular.
            try:
                model = smf.logit(formula, data=g).fit_regularized(
                    method="l1", alpha=0.1, disp=0, maxiter=300,
                )
                method_used = "L1-penalized"
            except Exception as exc:
                logger.warning("Logistic regression failed for %s: %s", game, exc)
                continue

        summary_text = (
            model.summary() if hasattr(model, "summary") else "(regularized - no full summary)"
        )
        full_summaries.append(
            f"\n{'='*70}\n  LOGISTIC REGRESSION - {game.upper()} "
            f"(prosocial = {prosocial}, method = {method_used})\n"
            f"{'='*70}\n{summary_text}\n"
        )

        coef = model.params
        
        # 3. Safely handle Confidence Intervals without fabricating data
        try:
            ci = model.conf_int()
        except Exception:
            # If CIs cannot be computed (common in regularized statsmodels fits), output NaNs
            ci = pd.DataFrame(
                {0: np.nan, 1: np.nan},
                index=coef.index,
            )
            
        # Extract metrics safely (regularized models may lack these attributes)
        pvals = getattr(model, "pvalues", pd.Series(np.nan, index=coef.index))
        pseudo_r2 = getattr(model, "prsquared", np.nan)
        nobs = int(getattr(model, "nobs", len(g)))
        aic = getattr(model, "aic", np.nan)
        bic = getattr(model, "bic", np.nan)

        for term in coef.index:
            coef_val = coef[term]
            clipped = np.clip(coef_val, -20, 20)
            
            row = {
                "game": game,
                "term": term,
                "coef": coef_val,
                "odds_ratio": np.exp(clipped),
                "ci_lower": np.exp(np.clip(ci.loc[term, 0], -20, 20)) if pd.notna(ci.loc[term, 0]) else np.nan,
                "ci_upper": np.exp(np.clip(ci.loc[term, 1], -20, 20)) if pd.notna(ci.loc[term, 1]) else np.nan,
                "p_value": pvals.get(term, np.nan) if isinstance(pvals, pd.Series) else (pvals[term] if pd.notna(pvals).any() else np.nan),
                "pseudo_r2": pseudo_r2,
                "n": nobs,
                "aic": aic,
                "bic": bic,
                "method": method_used,
            }
            all_rows.append(row)

        # Predicted odds/probability matrix for every context x persona cell.
        ctx_levels = sorted(g["context"].dropna().unique())
        persona_levels = sorted(g["persona"].dropna().unique())
        grid = pd.DataFrame(
            [(ctx, persona) for ctx in ctx_levels for persona in persona_levels],
            columns=["context", "persona"],
        )
        try:
            pred = model.predict(grid)
            pred = pd.to_numeric(pred, errors="coerce")
            clipped_pred = np.clip(pred, 1e-12, 1 - 1e-12)
            grid["predicted_probability"] = clipped_pred
            grid["predicted_odds"] = clipped_pred / (1 - clipped_pred)

            # Baseline-relative odds ratio matrix (baseline: abstract_game_theory x generic)
            baseline_mask = (
                (grid["context"] == "abstract_game_theory")
                & (grid["persona"] == "generic")
            )
            if baseline_mask.any():
                baseline_odds = float(grid.loc[baseline_mask, "predicted_odds"].iloc[0])
                if np.isfinite(baseline_odds) and baseline_odds > 0:
                    grid["predicted_odds_ratio_vs_baseline"] = (
                        grid["predicted_odds"] / baseline_odds
                    )
                else:
                    grid["predicted_odds_ratio_vs_baseline"] = np.nan
            else:
                grid["predicted_odds_ratio_vs_baseline"] = np.nan

            grid["game"] = game
            grid["method"] = method_used
            matrix_rows.extend(grid.to_dict("records"))
        except Exception as exc:
            logger.warning(
                "Could not generate odds matrix for %s: %s", game, exc
            )

    results = pd.DataFrame(all_rows)
    if not results.empty:
        path = os.path.join(save_dir, "logistic_regression.csv")
        results.round(6).to_csv(path, index=False)
        print(f"  Saved: {path}")

    if full_summaries:
        path = os.path.join(save_dir, "logistic_regression_summaries.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(full_summaries))
        print(f"  Saved: {path}")

    if matrix_rows:
        matrix_df = pd.DataFrame(matrix_rows)

        long_path = os.path.join(save_dir, "logistic_odds_matrix.csv")
        matrix_df.round(6).to_csv(long_path, index=False)
        print(f"  Saved: {long_path}")

        for game in sorted(matrix_df["game"].unique()):
            sub = matrix_df[matrix_df["game"] == game]

            odds_piv = sub.pivot(
                index="persona", columns="context", values="predicted_odds"
            )
            odds_path = os.path.join(save_dir, f"logistic_odds_matrix_{game}.csv")
            odds_piv.round(6).to_csv(odds_path)

            prob_piv = sub.pivot(
                index="persona", columns="context", values="predicted_probability"
            )
            prob_path = os.path.join(
                save_dir, f"logistic_probability_matrix_{game}.csv"
            )
            prob_piv.round(6).to_csv(prob_path)

            or_piv = sub.pivot(
                index="persona", columns="context", values="predicted_odds_ratio_vs_baseline"
            )
            or_path = os.path.join(save_dir, f"logistic_odds_ratio_matrix_{game}.csv")
            or_piv.round(6).to_csv(or_path)

            print(f"  Saved: {odds_path}")
            print(f"  Saved: {prob_path}")
            print(f"  Saved: {or_path}")

    return results


# ===================================================================
# 2. OLS REGRESSION  (numeric / k-level games)
# ===================================================================

def ols_regression(
    df: pd.DataFrame, save_dir: str
) -> pd.DataFrame:
    """
    Fit OLS for each numeric game:
        numeric_decision ~ C(context) * C(persona)

    Reports coefficients, ANOVA table, and R².
    """
    if not _HAS_SM:
        print("  [skip] statsmodels not installed — skipping OLS regression.")
        return pd.DataFrame()

    valid = df[~df["is_failed"]].copy()
    all_rows: List[dict] = []
    full_summaries: List[str] = []

    for game in sorted(valid["game_type"].unique()):
        g = valid[valid["game_type"] == game].copy()
        if not _is_numeric_game(g):
            continue

        g = g.dropna(subset=["numeric_decision"])
        if len(g) < 10:
            continue

        try:
            formula = (
                "numeric_decision ~ "
                "C(context, Treatment('abstract_game_theory')) * "
                "C(persona, Treatment('generic'))"
            )
            model = smf.ols(formula, data=g).fit()
        except Exception as exc:
            logger.warning("OLS failed for %s: %s", game, exc)
            continue

        # ANOVA (Type II)
        try:
            anova = sm.stats.anova_lm(model, typ=2)
            anova_str = anova.to_string()
        except Exception:
            anova_str = "(ANOVA unavailable)"

        full_summaries.append(
            f"\n{'='*70}\n  OLS REGRESSION — {game.upper()}\n{'='*70}\n"
            f"{model.summary()}\n\n  Type II ANOVA:\n{anova_str}\n"
        )

        for term in model.params.index:
            ci = model.conf_int().loc[term]
            row = {
                "game": game,
                "term": term,
                "coef": model.params[term],
                "std_err": model.bse[term],
                "t_stat": model.tvalues[term],
                "p_value": model.pvalues[term],
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "n": int(model.nobs),
            }
            all_rows.append(row)

    results = pd.DataFrame(all_rows)
    if not results.empty:
        path = os.path.join(save_dir, "ols_regression.csv")
        results.round(6).to_csv(path, index=False)
        print(f"  Saved: {path}")

    if full_summaries:
        path = os.path.join(save_dir, "ols_regression_summaries.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(full_summaries))
        print(f"  Saved: {path}")
        for s in full_summaries:
            print(s)

    return results


# ===================================================================
# 3. EFFECT SIZES
# ===================================================================

def compute_effect_sizes(
    df: pd.DataFrame, save_dir: str
) -> pd.DataFrame:
    """
    Compute effect sizes for every game × factor pair.

    * Categorical: Cramér's V  (from chi-square)
    * Numeric:     Eta-squared (from Kruskal-Wallis H / one-way ANOVA SS)

    Also reports the underlying test statistic and p-value.
    """
    valid = df[~df["is_failed"]].copy()
    rows: List[dict] = []

    for game in sorted(valid["game_type"].unique()):
        g = valid[valid["game_type"] == game]
        is_num = _is_numeric_game(g)

        for factor in ["context", "persona"]:
            if is_num:
                groups = [
                    grp["numeric_decision"].dropna().values
                    for _, grp in g.groupby(factor)
                ]
                groups = [gr for gr in groups if len(gr) > 0]
                if len(groups) < 2:
                    continue

                # Kruskal-Wallis
                stat, p = sp_stats.kruskal(*groups)

                # Eta-squared (from H statistic):
                # η² = (H - k + 1) / (n - k)
                n_total = sum(len(gr) for gr in groups)
                k = len(groups)
                eta_sq = (stat - k + 1) / (n_total - k) if n_total > k else 0.0
                eta_sq = max(0.0, eta_sq)

                rows.append({
                    "game": game, "factor": factor, "test": "Kruskal-Wallis",
                    "statistic": stat, "p_value": p,
                    "effect_size_name": "eta_squared", "effect_size": eta_sq,
                    "n": n_total, "k_groups": k,
                })
            else:
                ct = pd.crosstab(g[factor], g["action_label"])
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    continue

                chi2, p, dof, _ = sp_stats.chi2_contingency(ct)
                n_total = ct.sum().sum()
                min_dim = min(ct.shape[0], ct.shape[1]) - 1

                # Cramér's V = sqrt(chi2 / (n * min(r-1, c-1)))
                cramers_v = math.sqrt(chi2 / (n_total * min_dim)) if min_dim > 0 else 0.0

                rows.append({
                    "game": game, "factor": factor, "test": "Chi-square",
                    "statistic": chi2, "p_value": p,
                    "effect_size_name": "cramers_v", "effect_size": cramers_v,
                    "n": int(n_total), "k_groups": ct.shape[0],
                })

    results = pd.DataFrame(rows)

    # --- Holm–Bonferroni correction across all tests ---
    if not results.empty and len(results) > 1 and multipletests is not None:
        reject, corrected_p, _, _ = multipletests(
            results["p_value"], method="holm", alpha=0.05
        )
        results["p_corrected_holm"] = corrected_p
        results["significant_corrected"] = reject
    elif not results.empty:
        if multipletests is None:
            logger.warning(
                "statsmodels is not installed; using uncorrected p-values for effect sizes."
            )
        results["p_corrected_holm"] = results["p_value"]
        results["significant_corrected"] = results["p_value"] < 0.05

    if not results.empty:
        path = os.path.join(save_dir, "effect_sizes.csv")
        results.round(6).to_csv(path, index=False)
        print(f"  Saved: {path}")

        print("\n-- Effect Sizes (with Holm-corrected p-values) --")
        display_cols = [
            "game", "factor", "test", "statistic", "p_value",
            "p_corrected_holm", "effect_size_name", "effect_size",
            "significant_corrected",
        ]
        print(results[display_cols].round(4).to_string(index=False))

    return results


# ===================================================================
# 4. POST-HOC PAIRWISE TESTS
# ===================================================================

def posthoc_pairwise(
    df: pd.DataFrame, save_dir: str
) -> pd.DataFrame:
    """
    Pairwise comparisons to identify which groups differ.

    * Numeric games: Tukey HSD on numeric_decision, grouped by context
      and persona.
    * Categorical games: pairwise chi-square (Bonferroni-corrected) on
      action distribution, grouped by context and persona.
    """
    valid = df[~df["is_failed"]].copy()
    rows: List[dict] = []

    for game in sorted(valid["game_type"].unique()):
        g = valid[valid["game_type"] == game]
        is_num = _is_numeric_game(g)

        for factor in ["context", "persona"]:
            levels = sorted(g[factor].unique())
            if len(levels) < 2:
                continue

            if is_num:
                vals = g.dropna(subset=["numeric_decision"])
                if len(vals) < 4:
                    continue
                try:
                    tukey = pairwise_tukeyhsd(
                        vals["numeric_decision"], vals[factor], alpha=0.05
                    )
                    for row_data in tukey.summary().data[1:]:
                        rows.append({
                            "game": game, "factor": factor, "test": "Tukey HSD",
                            "group1": row_data[0], "group2": row_data[1],
                            "mean_diff": float(row_data[2]),
                            "p_value": float(row_data[3]),
                            "significant": bool(row_data[4] == "True" or row_data[4] is True),
                        })
                except Exception as exc:
                    logger.warning("Tukey failed for %s/%s: %s", game, factor, exc)

            else:
                # Pairwise chi-square with Bonferroni
                n_tests = len(levels) * (len(levels) - 1) // 2
                for i, lev_a in enumerate(levels):
                    for lev_b in levels[i + 1:]:
                        sub = g[g[factor].isin([lev_a, lev_b])]
                        ct = pd.crosstab(sub[factor], sub["action_label"])
                        if ct.shape[0] < 2 or ct.shape[1] < 2:
                            continue
                        chi2, p, _, _ = sp_stats.chi2_contingency(ct)
                        p_adj = min(p * n_tests, 1.0)  # Bonferroni
                        rows.append({
                            "game": game, "factor": factor,
                            "test": "Chi-square (Bonferroni)",
                            "group1": lev_a, "group2": lev_b,
                            "mean_diff": chi2,  # chi2 stored here
                            "p_value": p_adj,
                            "significant": p_adj < 0.05,
                        })

    results = pd.DataFrame(rows)
    if not results.empty:
        path = os.path.join(save_dir, "posthoc_pairwise.csv")
        results.round(6).to_csv(path, index=False)
        print(f"  Saved: {path}")

        # Print only significant pairs
        sig = results[results["significant"]]
        print(f"\n-- Significant Pairwise Differences ({len(sig)}/{len(results)}) --")
        if not sig.empty:
            print(sig.to_string(index=False))
        else:
            print("  (none)")

    return results


# ===================================================================
# 5. VARIANCE PARTITIONING
# ===================================================================

def variance_partitioning(
    df: pd.DataFrame, save_dir: str
) -> pd.DataFrame:
    """
    Decompose variance in prosocial choice (categorical) or numeric
    value into:  context (main), persona (main), interaction, residual.

    Uses Type II SS from OLS on the binary/numeric outcome.
    """
    if not _HAS_SM:
        print("  [skip] statsmodels not installed.")
        return pd.DataFrame()

    valid = df[~df["is_failed"]].copy()
    rows: List[dict] = []

    for game in sorted(valid["game_type"].unique()):
        g = valid[valid["game_type"] == game].copy()
        is_num = _is_numeric_game(g)

        if is_num:
            g = g.dropna(subset=["numeric_decision"])
            dep_var = "numeric_decision"
        else:
            prosocial = _PROSOCIAL_MAP.get(game)
            if not prosocial:
                continue
            g["prosocial"] = (g["action_label"] == prosocial).astype(float)
            dep_var = "prosocial"

        if len(g) < 10:
            continue

        try:
            formula = f"{dep_var} ~ C(context) + C(persona) + C(context):C(persona)"
            model = smf.ols(formula, data=g).fit()
            anova = sm.stats.anova_lm(model, typ=2)
        except Exception as exc:
            logger.warning("Variance partitioning failed for %s: %s", game, exc)
            continue

        total_ss = anova["sum_sq"].sum()
        for source in anova.index:
            if source == "Residual":
                label = "Residual"
            elif "context" in source and "persona" in source:
                label = "Interaction"
            elif "context" in source:
                label = "Context"
            elif "persona" in source:
                label = "Persona"
            else:
                label = source

            rows.append({
                "game": game,
                "source": label,
                "sum_sq": anova.loc[source, "sum_sq"],
                "pct_variance": anova.loc[source, "sum_sq"] / total_ss * 100,
                "df": anova.loc[source, "df"],
                "F": anova.loc[source].get("F", np.nan),
                "p_value": anova.loc[source].get("PR(>F)", np.nan),
            })

    results = pd.DataFrame(rows)
    if not results.empty:
        path = os.path.join(save_dir, "variance_partitioning.csv")
        results.round(4).to_csv(path, index=False)
        print(f"  Saved: {path}")

        print("\n-- Variance Partitioning (% of total SS) --")
        pivot = results.pivot_table(
            index="game", columns="source", values="pct_variance"
        )
        col_order = [c for c in ["Context", "Persona", "Interaction", "Residual"]
                     if c in pivot.columns]
        print(pivot[col_order].round(1).to_string())

    return results


# ===================================================================
# 6. REASONING LANGUAGE ANALYSIS
# ===================================================================

_KEYWORD_CATEGORIES: Dict[str, List[str]] = {
    "game_theory": [
        "nash", "equilibrium", "dominant", "dominated", "rational",
        "game theory", "payoff",
    ],
    "cooperation": [
        "cooperat", "trust", "mutual", "together", "partner",
        "fairness", "relationship",
    ],
    "competition": [
        "maximize", "advantage", "exploit", "undercut", "dominate",
        "aggressive", "outperform",
    ],
    "risk": [
        "risk", "safe", "guaranteed", "uncertain", "worst case",
        "worst-case", "danger",
    ],
    "emotion": [
        "empathy", "empathetic", "feeling", "compassion", "care",
        "harm", "well-being", "welfare",
    ],
    "mathematics": [
        "probability", "expected value", "calculate", "optimal",
        "minimize", "utility",
    ],
}


def reasoning_language_analysis(
    df: pd.DataFrame, save_dir: str
) -> pd.DataFrame:
    """
    Compute keyword-category frequencies per persona (and per persona × game).

    Rates are reported as occurrences per 1 000 words so they are comparable
    across groups of different sizes.
    """
    valid = df[~df["is_failed"]].copy()
    rows: List[dict] = []

    for persona in sorted(valid["persona"].unique()):
        for game in [None] + sorted(valid["game_type"].unique()):
            if game is None:
                sub = valid[valid["persona"] == persona]
                game_label = "ALL"
            else:
                sub = valid[(valid["persona"] == persona) & (valid["game_type"] == game)]
                game_label = game

            text = " ".join(
                sub["reasoning"].dropna().astype(str).values
            ).lower()
            total_words = max(len(text.split()), 1)

            row: Dict[str, Any] = {
                "persona": persona,
                "game": game_label,
                "total_words": total_words,
            }
            for category, terms in _KEYWORD_CATEGORIES.items():
                count = sum(len(re.findall(t, text)) for t in terms)
                row[f"{category}_count"] = count
                row[f"{category}_per_1k"] = count / total_words * 1000

            rows.append(row)

    results = pd.DataFrame(rows)
    if not results.empty:
        path = os.path.join(save_dir, "reasoning_language.csv")
        results.round(3).to_csv(path, index=False)
        print(f"  Saved: {path}")

        # Print summary (ALL games only)
        summary = results[results["game"] == "ALL"]
        rate_cols = [c for c in summary.columns if c.endswith("_per_1k")]
        print("\n-- Reasoning Language (per 1 000 words, all games) --")
        print(summary[["persona", "total_words"] + rate_cols].to_string(index=False))

    return results


# ===================================================================
# 7. PERSONA CONSISTENCY (Shannon entropy)
# ===================================================================

def persona_consistency(
    df: pd.DataFrame, save_dir: str
) -> pd.DataFrame:
    """
    Compute Shannon entropy of the action distribution per
    (game, persona, context) cell.

    Lower entropy = more deterministic behaviour.
    For a binary choice, max entropy is 1.0 bit (50/50 split);
    0.0 means perfectly consistent.
    """
    valid = df[~df["is_failed"]].copy()
    rows: List[dict] = []

    for game in sorted(valid["game_type"].unique()):
        g = valid[valid["game_type"] == game]
        is_num = _is_numeric_game(g)

        for persona in sorted(g["persona"].unique()):
            for ctx in sorted(g["context"].unique()):
                cell = g[(g["persona"] == persona) & (g["context"] == ctx)]
                n = len(cell)
                if n == 0:
                    continue

                if is_num:
                    # Use coefficient of variation as consistency metric
                    vals = cell["numeric_decision"].dropna()
                    if len(vals) < 2:
                        continue
                    cv = vals.std() / vals.mean() if vals.mean() != 0 else np.inf
                    rows.append({
                        "game": game, "persona": persona, "context": ctx,
                        "n": n, "metric": "coeff_of_variation", "value": cv,
                    })
                else:
                    # Shannon entropy of action distribution
                    counts = cell["action_label"].value_counts(normalize=True)
                    entropy = -sum(p * math.log2(p) for p in counts if p > 0)
                    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
                    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

                    rows.append({
                        "game": game, "persona": persona, "context": ctx,
                        "n": n, "metric": "shannon_entropy",
                        "value": abs(entropy),  # avoid -0.0
                    })
                    rows.append({
                        "game": game, "persona": persona, "context": ctx,
                        "n": n, "metric": "normalised_entropy",
                        "value": abs(norm_entropy),  # avoid -0.0
                    })

    results = pd.DataFrame(rows)
    if not results.empty:
        path = os.path.join(save_dir, "persona_consistency.csv")
        results.round(4).to_csv(path, index=False)
        print(f"  Saved: {path}")

        # Print pivot for normalised entropy
        ent = results[results["metric"] == "normalised_entropy"]
        if not ent.empty:
            print("\n-- Persona Consistency (normalised entropy, 0 = deterministic) --")
            for game in sorted(ent["game"].unique()):
                sub = ent[ent["game"] == game]
                piv = sub.pivot(index="persona", columns="context", values="value")
                print(f"\n  {game}:")
                print(piv.round(3).to_string())

    return results


# ===================================================================
# 8. PERSONA × CONTEXT INTERACTION TABLE
# ===================================================================

def interaction_table(
    df: pd.DataFrame, save_dir: str
) -> pd.DataFrame:
    """
    Build a detailed persona × context rate table for all games.

    For categorical games: prosocial action rate (%).
    For numeric games: mean, median, std of numeric_decision.
    """
    valid = df[~df["is_failed"]].copy()
    rows: List[dict] = []

    for game in sorted(valid["game_type"].unique()):
        g = valid[valid["game_type"] == game]
        is_num = _is_numeric_game(g)

        for persona in sorted(g["persona"].unique()):
            for ctx in sorted(g["context"].unique()):
                cell = g[(g["persona"] == persona) & (g["context"] == ctx)]
                n = len(cell)
                if n == 0:
                    continue

                row: Dict[str, Any] = {
                    "game": game, "persona": persona, "context": ctx, "n": n,
                }

                if is_num:
                    vals = cell["numeric_decision"].dropna()
                    row["mean"] = vals.mean()
                    row["median"] = vals.median()
                    row["std"] = vals.std()
                else:
                    prosocial = _PROSOCIAL_MAP.get(game, "")
                    prosocial_rate = (cell["action_label"] == prosocial).mean() * 100
                    row["prosocial_pct"] = prosocial_rate
                    # Also store full distribution
                    for label in sorted(cell["action_label"].unique()):
                        row[f"pct_{label}"] = (
                            (cell["action_label"] == label).mean() * 100
                        )

                rows.append(row)

    results = pd.DataFrame(rows)
    if not results.empty:
        path = os.path.join(save_dir, "interaction_table.csv")
        results.round(2).to_csv(path, index=False)
        print(f"  Saved: {path}")

        # Print compact pivot for categorical games
        cat = results.dropna(subset=["prosocial_pct"])
        if not cat.empty:
            print("\n-- Persona × Context: Prosocial Rate (%) --")
            for game in sorted(cat["game"].unique()):
                sub = cat[cat["game"] == game]
                piv = sub.pivot(index="persona", columns="context", values="prosocial_pct")
                print(f"\n  {game}:")
                print(piv.round(1).to_string())

    return results


# ===================================================================
# 9. POSITIONAL BIAS (swap) ANALYSIS
# ===================================================================

def positional_bias_analysis(
    df: pd.DataFrame, save_dir: str
) -> pd.DataFrame:
    """
    Test positional bias per game × persona:
    compare action_a choice rate in normal vs swapped variations.
    """
    valid = df[~df["is_failed"]].copy()
    if "swapped" not in valid.columns:
        valid["swapped"] = valid["variation"].astype(int) % 2 == 1

    rows: List[dict] = []

    for game in sorted(valid["game_type"].unique()):
        g = valid[valid["game_type"] == game]
        if _is_numeric_game(g):
            continue  # swap analysis only meaningful for categorical

        for persona in sorted(g["persona"].unique()):
            sub = g[g["persona"] == persona]
            normal = sub[~sub["swapped"]]
            swapped = sub[sub["swapped"]]

            rate_normal = (normal["action"] == "action_a").mean() * 100
            rate_swapped = (swapped["action"] == "action_a").mean() * 100
            bias_pp = abs(rate_normal - rate_swapped)

            # Chi-square test
            ct = pd.crosstab(sub["swapped"], sub["action"])
            if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                chi2, p, _, _ = sp_stats.chi2_contingency(ct)
            else:
                chi2, p = 0.0, 1.0

            rows.append({
                "game": game, "persona": persona,
                "pct_action_a_normal": rate_normal,
                "pct_action_a_swapped": rate_swapped,
                "bias_pp": bias_pp,
                "chi2": chi2, "p_value": p,
                "significant": p < 0.05,
            })

    results = pd.DataFrame(rows)
    if not results.empty:
        path = os.path.join(save_dir, "positional_bias.csv")
        results.round(4).to_csv(path, index=False)
        print(f"  Saved: {path}")

        print("\n-- Positional Bias by Game × Persona --")
        print(results.to_string(index=False))

    return results


# ===================================================================
# ORCHESTRATOR
# ===================================================================

def run_regression_analysis(
    file_path: Optional[str] = None,
) -> None:
    """
    Run the full regression and advanced statistical analysis pipeline.

    Results are saved per-model under ``results/<model_name>/``.
    """
    results_dir = getattr(config, "RESULTS_PATH", "results")
    os.makedirs(results_dir, exist_ok=True)

    df = load_data(file_path)
    models = sorted(df["model"].unique())

    for original_mdl in models:
        safe_mdl = original_mdl
        if ":" in original_mdl:
            logger.warning(
                "Model name '%s' contains ':', which may cause filesystem "
                "issues. Replacing with '_'.", original_mdl
            )
            safe_mdl = original_mdl.replace(":", "_")

        model_dir = os.path.join(results_dir, safe_mdl)
        os.makedirs(model_dir, exist_ok=True)

        model_df = df[df["model"] == original_mdl].copy()
        valid = model_df[~model_df["is_failed"]]

        print("\n" + "#" * 70)
        print(f"  REGRESSION ANALYSIS — {original_mdl}")
        print(f"  {len(valid)} valid observations")
        print("#" * 70)

        print("\n\n=== 1. Logistic Regression (categorical games) ===")
        logistic_regression(model_df, model_dir)

        print("\n\n=== 2. OLS Regression (numeric games) ===")
        ols_regression(model_df, model_dir)

        print("\n\n=== 3. Effect Sizes (Cramer's V / eta-squared) + Holm correction ===")
        compute_effect_sizes(model_df, model_dir)

        print("\n\n=== 4. Post-hoc Pairwise Comparisons ===")
        posthoc_pairwise(model_df, model_dir)

        print("\n\n=== 5. Variance Partitioning ===")
        variance_partitioning(model_df, model_dir)

        print("\n\n=== 6. Reasoning Language Analysis ===")
        reasoning_language_analysis(model_df, model_dir)

        print("\n\n=== 7. Persona Consistency (entropy) ===")
        persona_consistency(model_df, model_dir)

        print("\n\n=== 8. Persona × Context Interaction Table ===")
        interaction_table(model_df, model_dir)

        print("\n\n=== 9. Positional Bias Analysis ===")
        positional_bias_analysis(model_df, model_dir)

        print(f"\n  All regression outputs saved to: {model_dir}/")

    print("\n" + "=" * 70)
    print("  Regression analysis complete.")
    print("=" * 70)


# ===================================================================
# Standalone
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_regression_analysis()
