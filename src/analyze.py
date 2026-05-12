"""Analyse experiment results: aggregate, run statistical tests, and produce
figures.

Usage:
    python src/analyze.py --inputs results/exp1*.jsonl results/exp2*.jsonl

Produces:
    results/aggregated.csv          per-cell aggregates
    results/per_item.csv            per-item flat table
    results/stats.json              statistical-test outputs
    figures/*.png                   plots
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

WORKSPACE = Path(__file__).resolve().parent.parent


def load_records(paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        if not p.exists():
            print(f"WARN: missing {p}")
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            init_m = r["initial_metrics"]
            row = {
                "model": r["model"],
                "dataset": r["dataset"],
                "tone_level": r["tone_level"],
                "tone_label": r["tone_label"],
                "item_id": r["item_id"],
                "answer_gold": r["answer_gold"],
                "init_correct": init_m["correct"],
                "init_apology": init_m["apology"],
                "init_deliberation": init_m["deliberation_markers"],
                "init_has_deliberation": init_m["has_deliberation"],
                "init_completion_tokens": init_m["completion_tokens"],
                "init_words": init_m["words"],
                "init_chars": init_m["chars"],
            }
            if "followup_metrics" in r:
                fu = r["followup_metrics"]
                row.update(
                    {
                        "followup_correct": fu["correct"],
                        "followup_apology": fu["apology"],
                        "followup_completion_tokens": fu["completion_tokens"],
                        "flipped": r.get("flipped", 0),
                        "flip_correct_to_wrong": r.get("flip_correct_to_wrong", 0),
                        "flip_wrong_to_correct": r.get("flip_wrong_to_correct", 0),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


# --- Statistical tests ---


def mcnemar_pair(p_baseline: list[int], p_compare: list[int]) -> dict:
    """Paired binary test (per-item): McNemar."""
    assert len(p_baseline) == len(p_compare)
    b = sum(1 for x, y in zip(p_baseline, p_compare) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(p_baseline, p_compare) if x == 0 and y == 1)
    if b + c == 0:
        return {"b": b, "c": c, "p": 1.0, "stat": 0.0}
    # exact binomial test
    p = stats.binomtest(min(b, c), n=b + c, p=0.5).pvalue
    return {"b": b, "c": c, "p": float(p), "stat": float((b - c) / max(1, math.sqrt(b + c)))}


def cochran_armitage(df: pd.DataFrame, value_col: str, level_col: str = "tone_int") -> dict:
    """Trend test for a 0/1 variable across ordered tone levels."""
    levels = sorted(df[level_col].unique())
    counts_yes = [int((df[df[level_col] == k][value_col] == 1).sum()) for k in levels]
    counts_no = [int((df[df[level_col] == k][value_col] == 0).sum()) for k in levels]
    totals = [y + n for y, n in zip(counts_yes, counts_no)]
    grand_yes = sum(counts_yes)
    grand_total = sum(totals)
    if grand_total == 0 or grand_yes == 0 or grand_yes == grand_total:
        return {"z": 0.0, "p": 1.0, "levels": levels, "p_yes": [0.0] * len(levels)}
    p_bar = grand_yes / grand_total
    # numerator: sum_k t_k * (y_k - n_k * p_bar) where t is the level value
    num = sum(k * (counts_yes[i] - totals[i] * p_bar) for i, k in enumerate(levels))
    var = (
        p_bar
        * (1 - p_bar)
        * (
            grand_total * sum(k * k * totals[i] for i, k in enumerate(levels))
            - sum(k * totals[i] for i, k in enumerate(levels)) ** 2
        )
        / grand_total
    )
    if var <= 0:
        return {"z": 0.0, "p": 1.0, "levels": levels, "p_yes": [y / max(1, t) for y, t in zip(counts_yes, totals)]}
    z = num / math.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "z": float(z),
        "p": float(p),
        "levels": levels,
        "p_yes": [y / max(1, t) for y, t in zip(counts_yes, totals)],
    }


def wilcoxon_pair(x: list[float], y: list[float]) -> dict:
    """Paired Wilcoxon signed-rank for completion-token comparison."""
    diffs = [a - b for a, b in zip(x, y)]
    if all(d == 0 for d in diffs):
        return {"p": 1.0, "stat": 0.0, "median_diff": 0.0}
    try:
        res = stats.wilcoxon(diffs)
        return {
            "p": float(res.pvalue),
            "stat": float(res.statistic),
            "median_diff": float(np.median(diffs)),
        }
    except ValueError:
        return {"p": 1.0, "stat": 0.0, "median_diff": float(np.median(diffs))}


def aggregate_cell_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, dataset, tone_level) aggregates."""
    g = df.groupby(["model", "dataset", "tone_level"], as_index=False)
    out = g.agg(
        n=("init_correct", "size"),
        accuracy=("init_correct", "mean"),
        apology_rate=("init_apology", "mean"),
        delib_rate=("init_has_deliberation", "mean"),
        delib_count_mean=("init_deliberation", "mean"),
        completion_tokens_mean=("init_completion_tokens", "mean"),
        completion_tokens_median=("init_completion_tokens", "median"),
        words_mean=("init_words", "mean"),
        chars_mean=("init_chars", "mean"),
    )
    out["accuracy_se"] = (out["accuracy"] * (1 - out["accuracy"]) / out["n"]).pow(0.5)
    out["apology_rate_se"] = (
        out["apology_rate"] * (1 - out["apology_rate"]) / out["n"]
    ).pow(0.5)
    return out


def followup_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """If follow-up cols exist, compute FlipFlop deltas."""
    if "followup_correct" not in df.columns:
        return pd.DataFrame()
    g = df.dropna(subset=["followup_correct"]).groupby(
        ["model", "dataset", "tone_level"], as_index=False
    )
    out = g.agg(
        n=("init_correct", "size"),
        acc_init=("init_correct", "mean"),
        acc_followup=("followup_correct", "mean"),
        flip_rate=("flipped", "mean"),
        flip_correct_to_wrong=("flip_correct_to_wrong", "mean"),
        flip_wrong_to_correct=("flip_wrong_to_correct", "mean"),
        followup_apology_rate=("followup_apology", "mean"),
        followup_tokens_mean=("followup_completion_tokens", "mean"),
    )
    out["flipflop_delta"] = out["acc_followup"] - out["acc_init"]
    return out


def per_tone_stats(df: pd.DataFrame) -> dict:
    """Tone-level statistical tests (vs L0) within each (model, dataset)."""
    df = df.copy()
    tone_to_int = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}
    df["tone_int"] = df["tone_level"].map(tone_to_int)
    out = {}
    for (model, dataset), sub in df.groupby(["model", "dataset"]):
        # pivot so each item appears under each tone
        pivot_acc = sub.pivot_table(
            index="item_id", columns="tone_level", values="init_correct", aggfunc="first"
        )
        pivot_apo = sub.pivot_table(
            index="item_id", columns="tone_level", values="init_apology", aggfunc="first"
        )
        pivot_tok = sub.pivot_table(
            index="item_id", columns="tone_level", values="init_completion_tokens", aggfunc="first"
        )
        # drop items not present in all tones
        cols = [c for c in ["L0", "L1", "L2", "L3", "L4", "L5"] if c in pivot_acc.columns]
        pivot_acc = pivot_acc[cols].dropna()
        pivot_apo = pivot_apo[cols].dropna()
        pivot_tok = pivot_tok[cols].dropna()
        per_pair: dict = {}
        baseline = "L0"
        if baseline not in cols:
            continue
        for c in cols:
            if c == baseline:
                continue
            per_pair[c] = {
                "mcnemar_acc": mcnemar_pair(
                    pivot_acc[baseline].astype(int).tolist(),
                    pivot_acc[c].astype(int).tolist(),
                ),
                "mcnemar_apology": mcnemar_pair(
                    pivot_apo[baseline].astype(int).tolist(),
                    pivot_apo[c].astype(int).tolist(),
                ),
                "wilcoxon_tokens": wilcoxon_pair(
                    pivot_tok[c].astype(float).tolist(),
                    pivot_tok[baseline].astype(float).tolist(),
                ),
            }
        # trend tests
        ca_acc = cochran_armitage(sub, "init_correct", "tone_int")
        ca_apo = cochran_armitage(sub, "init_apology", "tone_int")
        # Spearman for tokens
        tok_corr = stats.spearmanr(sub["tone_int"], sub["init_completion_tokens"])
        out[f"{model}::{dataset}"] = {
            "n_items": int(len(pivot_acc)),
            "vs_L0": per_pair,
            "trend_acc": ca_acc,
            "trend_apology": ca_apo,
            "spearman_tone_tokens": {
                "rho": float(tok_corr.statistic),
                "p": float(tok_corr.pvalue),
            },
        }
    return out


# --- Plotting ---


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=140)


def plot_sweet_spot(agg: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("notebook", font_scale=1.05)
    sns.set_style("whitegrid")
    tone_order = ["L0", "L1", "L2", "L3", "L4", "L5"]
    metrics = [
        ("accuracy", "Accuracy", "accuracy_se"),
        ("apology_rate", "Apology rate (%Sorry)", "apology_rate_se"),
        ("completion_tokens_mean", "Mean completion tokens", None),
    ]
    for metric, ylab, se_col in metrics:
        fig, axes = plt.subplots(
            1, agg["dataset"].nunique(), figsize=(4 * agg["dataset"].nunique(), 4),
            sharey=True,
        )
        if agg["dataset"].nunique() == 1:
            axes = [axes]
        datasets = sorted(agg["dataset"].unique())
        for ax, ds in zip(axes, datasets):
            sub = agg[agg["dataset"] == ds]
            for model in sorted(sub["model"].unique()):
                ms = sub[sub["model"] == model].set_index("tone_level").reindex(tone_order)
                xs = list(range(len(tone_order)))
                ys = ms[metric].values
                ax.plot(xs, ys, marker="o", label=model)
                if se_col is not None:
                    ses = ms[se_col].values
                    ax.fill_between(
                        xs,
                        ys - 1.96 * ses,
                        ys + 1.96 * ses,
                        alpha=0.15,
                    )
            ax.set_xticks(xs)
            ax.set_xticklabels(tone_order)
            ax.set_xlabel("Tone level")
            ax.set_title(ds)
            ax.legend(fontsize=8)
        axes[0].set_ylabel(ylab)
        fig.suptitle(f"{ylab} vs. tone intensity")
        _save(fig, out_dir / f"sweetspot_{metric}.png")
        import matplotlib.pyplot as plt
        plt.close(fig)


def plot_acc_vs_apology(agg: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("notebook", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(7, 5))
    for (model, dataset), sub in agg.groupby(["model", "dataset"]):
        sub = sub.copy().sort_values("tone_level")
        ax.plot(
            sub["apology_rate"],
            sub["accuracy"],
            marker="o",
            label=f"{model}/{dataset}",
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["tone_level"],
                (row["apology_rate"], row["accuracy"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
    ax.set_xlabel("Apology rate")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs apology rate (each point = tone level)")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir / "acc_vs_apology.png")
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_token_growth(agg: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("notebook", font_scale=1.05)
    tone_order = ["L0", "L1", "L2", "L3", "L4", "L5"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for (model, dataset), sub in agg.groupby(["model", "dataset"]):
        s = sub.set_index("tone_level").reindex(tone_order)
        baseline = s.loc["L0", "completion_tokens_mean"]
        rel = (s["completion_tokens_mean"] / baseline) * 100
        ax.plot(range(len(tone_order)), rel, marker="o", label=f"{model}/{dataset}")
    ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(tone_order)))
    ax.set_xticklabels(tone_order)
    ax.set_ylabel("Completion tokens (% of L0)")
    ax.set_xlabel("Tone level")
    ax.set_title("Reasoning length growth relative to neutral prompt")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir / "token_growth.png")
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_flipflop(fu: pd.DataFrame, out_dir: Path) -> None:
    if fu.empty:
        return
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("notebook", font_scale=1.05)
    tone_order = ["L0", "L1", "L2", "L3", "L4", "L5"]
    datasets = sorted(fu["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = fu[fu["dataset"] == ds]
        for model in sorted(sub["model"].unique()):
            s = sub[sub["model"] == model].set_index("tone_level").reindex(tone_order)
            xs = list(range(len(tone_order)))
            ax.plot(xs, s["acc_init"], marker="o", label=f"{model} pre")
            ax.plot(xs, s["acc_followup"], marker="s", linestyle="--", label=f"{model} post")
        ax.set_xticks(xs)
        ax.set_xticklabels(tone_order)
        ax.set_xlabel("Tone level")
        ax.set_title(ds)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Pre- vs post-challenge accuracy by tone (FlipFlop)")
    _save(fig, out_dir / "flipflop_pre_post.png")
    import matplotlib.pyplot as plt
    plt.close(fig)

    # Flip-rate plot
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = fu[fu["dataset"] == ds]
        for model in sorted(sub["model"].unique()):
            s = sub[sub["model"] == model].set_index("tone_level").reindex(tone_order)
            xs = list(range(len(tone_order)))
            ax.plot(xs, s["flip_rate"], marker="o", label=f"{model} flip")
            ax.plot(
                xs,
                s["flip_correct_to_wrong"],
                marker="x",
                linestyle="--",
                label=f"{model} C→W",
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(tone_order)
        ax.set_xlabel("Tone level")
        ax.set_title(ds)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Rate")
    fig.suptitle("Post-challenge flip rate by preemptive tone (FlipFlop)")
    _save(fig, out_dir / "flipflop_flip_rate.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--figures-dir", default="figures")
    args = p.parse_args()

    paths = [Path(x) for x in args.inputs]
    df = load_records(paths)
    print(f"Loaded {len(df)} records.")

    res_dir = WORKSPACE / args.results_dir
    fig_dir = WORKSPACE / args.figures_dir

    df.to_csv(res_dir / "per_item.csv", index=False)
    agg = aggregate_cell_metrics(df)
    agg.to_csv(res_dir / "aggregated.csv", index=False)
    print("\n=== Aggregated cells ===")
    print(agg.to_string(index=False))

    fu = followup_metrics(df)
    if not fu.empty:
        fu.to_csv(res_dir / "followup_aggregated.csv", index=False)
        print("\n=== FlipFlop aggregates ===")
        print(fu.to_string(index=False))

    stats_out = per_tone_stats(df)

    def _coerce(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"unencodable {type(o)}")

    (res_dir / "stats.json").write_text(json.dumps(stats_out, indent=2, default=_coerce))

    plot_sweet_spot(agg, fig_dir)
    plot_acc_vs_apology(agg, fig_dir)
    plot_token_growth(agg, fig_dir)
    plot_flipflop(fu, fig_dir)
    print(f"\nFigures written to {fig_dir}")


if __name__ == "__main__":
    main()
