"""Final analyses: one-sided pooled tests + qualitative example tables +
extra figures for the report.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

WORKSPACE = Path(__file__).resolve().parent.parent
RESULTS = WORKSPACE / "results"
FIGS = WORKSPACE / "figures"
FIGS.mkdir(exist_ok=True)

sns.set_context("notebook", font_scale=1.05)
sns.set_style("whitegrid")


def _per_item_csv() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "per_item.csv")


def one_sided_pooled_truthful_qa(df: pd.DataFrame) -> dict:
    """For each model on TruthfulQA: McNemar one-sided test of L0 vs the
    skeptical-majority of L1-L5 (per item).
    """
    out = {}
    for model in df["model"].unique():
        sub = df[(df["model"] == model) & (df["dataset"] == "truthful_qa")]
        pivot = sub.pivot_table(
            index="item_id",
            columns="tone_level",
            values="init_correct",
            aggfunc="first",
        ).dropna()
        skeptic_cols = [c for c in ["L1", "L2", "L3", "L4", "L5"] if c in pivot.columns]
        l0 = pivot["L0"].astype(int)
        skeptic_majority = (
            pivot[skeptic_cols].astype(int).mean(axis=1) >= 0.5
        ).astype(int)
        # Count: items where exactly one of (L0, skeptic_maj) is correct.
        b = int(((l0 == 1) & (skeptic_majority == 0)).sum())  # L0 only correct
        c = int(((l0 == 0) & (skeptic_majority == 1)).sum())  # Skeptic only correct
        # one-sided: alternative = skeptic majority is correct more often
        if b + c == 0:
            p_one = 1.0
        else:
            p_one = stats.binomtest(c, n=b + c, p=0.5, alternative="greater").pvalue
        out[model] = {
            "n": int(len(pivot)),
            "L0_only_correct": b,
            "skeptic_only_correct": c,
            "skeptic_minus_L0_pp": float((skeptic_majority.mean() - l0.mean()) * 100),
            "one_sided_p": float(p_one),
        }
    return out


def plot_per_item_delta_truthful_qa(df: pd.DataFrame) -> None:
    """For each item where ANY tone changed correctness vs L0, plot a small bar
    showing the count of items 'recovered' under skeptic tones vs 'regressed'."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rows = []
    for model in df["model"].unique():
        sub = df[(df["model"] == model) & (df["dataset"] == "truthful_qa")]
        pivot = sub.pivot_table(
            index="item_id",
            columns="tone_level",
            values="init_correct",
            aggfunc="first",
        ).dropna()
        for tone in ["L1", "L2", "L3", "L4", "L5"]:
            if tone not in pivot.columns:
                continue
            l0 = pivot["L0"].astype(int)
            t = pivot[tone].astype(int)
            recover = int(((l0 == 0) & (t == 1)).sum())
            regress = int(((l0 == 1) & (t == 0)).sum())
            rows.append(
                {"model": model, "tone": tone, "recover": recover, "regress": -regress}
            )
    df_r = pd.DataFrame(rows)
    bar_w = 0.35
    tones = ["L1", "L2", "L3", "L4", "L5"]
    x = np.arange(len(tones))
    models = sorted(df["model"].unique())
    colors = ["#4c72b0", "#dd8452"]
    for i, model in enumerate(models):
        s = df_r[df_r["model"] == model].set_index("tone").reindex(tones)
        ax.bar(
            x + (i - 0.5) * bar_w,
            s["recover"],
            width=bar_w,
            color=colors[i],
            label=f"{model}: recovered (wrong→right)",
        )
        ax.bar(
            x + (i - 0.5) * bar_w,
            s["regress"],
            width=bar_w,
            color=colors[i],
            alpha=0.4,
            label=f"{model}: regressed (right→wrong)",
        )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tones)
    ax.set_xlabel("Tone level (vs L0 neutral)")
    ax.set_ylabel("Item count (negative = regression)")
    ax.set_title("TruthfulQA: per-tone recovery vs regression of items relative to L0")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / "tqa_per_item_recovery.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_token_per_dataset(df: pd.DataFrame) -> None:
    """One panel per (dataset, model) of completion-token distribution by tone."""
    sns.set_style("whitegrid")
    datasets = sorted(df["dataset"].unique())
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(
        len(models), len(datasets), figsize=(4 * len(datasets), 3.4 * len(models)),
        sharey=False,
    )
    if len(models) == 1 and len(datasets) == 1:
        axes = np.array([[axes]])
    elif len(models) == 1:
        axes = np.array([axes])
    elif len(datasets) == 1:
        axes = np.array([[ax] for ax in axes])
    tone_order = ["L0", "L1", "L2", "L3", "L4", "L5"]
    for i, model in enumerate(models):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            sub = df[(df["model"] == model) & (df["dataset"] == ds)]
            sns.boxplot(
                data=sub,
                x="tone_level",
                y="init_completion_tokens",
                order=tone_order,
                ax=ax,
                showfliers=False,
                color="#9bbed6",
            )
            ax.set_title(f"{model} / {ds}")
            ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel("Completion tokens")
            else:
                ax.set_ylabel("")
    fig.suptitle("Completion-token distribution by tone level (whisker=1.5×IQR)")
    fig.tight_layout()
    fig.savefig(FIGS / "tokens_box_per_dataset.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_flipflop_truthful_qa(df: pd.DataFrame) -> None:
    if "flipped" not in df.columns:
        return
    fu = df.dropna(subset=["flipped"])
    fu = fu[fu["dataset"] == "truthful_qa"]
    if fu.empty:
        return
    tone_order = ["L0", "L1", "L2", "L3", "L4", "L5"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in sorted(fu["model"].unique()):
        sub = fu[fu["model"] == model]
        agg = (
            sub.groupby("tone_level")
            .agg(
                c2w=("flip_correct_to_wrong", "mean"),
                w2c=("flip_wrong_to_correct", "mean"),
            )
            .reindex(tone_order)
        )
        x = np.arange(len(tone_order))
        ax.plot(x, agg["c2w"], "o-", label=f"{model}: correct→wrong (bad)", color="#c44e52")
        ax.plot(x, agg["w2c"], "s--", label=f"{model}: wrong→correct (good)", color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels(tone_order)
    ax.set_xlabel("Preemptive tone level (before initial answer)")
    ax.set_ylabel("Rate after follow-up rebuttal")
    ax.set_title(
        "TruthfulQA: post-rebuttal flip rates by preemptive tone\n"
        "(does priming with skeptical tone make the model more or less robust?)"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "flipflop_truthfulqa_directional.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_tone_ladder_summary(df: pd.DataFrame) -> None:
    """Single combined plot: accuracy with CI bands + token growth, side by side, TruthfulQA only."""
    sub = df[df["dataset"] == "truthful_qa"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    tone_order = ["L0", "L1", "L2", "L3", "L4", "L5"]
    palette = ["#4c72b0", "#dd8452"]
    for i, model in enumerate(sorted(sub["model"].unique())):
        m = sub[sub["model"] == model]
        agg = m.groupby("tone_level").agg(
            acc=("init_correct", "mean"),
            tok=("init_completion_tokens", "mean"),
            n=("init_correct", "size"),
        )
        agg = agg.reindex(tone_order)
        agg["se"] = (agg["acc"] * (1 - agg["acc"]) / agg["n"]) ** 0.5
        x = np.arange(len(tone_order))
        axes[0].errorbar(
            x, agg["acc"] * 100, yerr=1.96 * agg["se"] * 100, marker="o", capsize=3,
            label=model, color=palette[i],
        )
        axes[1].plot(
            x, agg["tok"], marker="o", color=palette[i], label=model
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tone_order)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xlabel("Tone level")
    axes[0].set_title("TruthfulQA accuracy (95% CI)")
    axes[0].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tone_order)
    axes[1].set_ylabel("Mean completion tokens")
    axes[1].set_xlabel("Tone level")
    axes[1].set_title("TruthfulQA reasoning length")
    axes[1].legend()
    fig.suptitle(
        "Sweet-spot evidence on TruthfulQA: accuracy nudge + clear length growth"
    )
    fig.tight_layout()
    fig.savefig(FIGS / "tqa_summary.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    df = _per_item_csv()
    print(f"Loaded {len(df)} per-item records.")

    one_sided = one_sided_pooled_truthful_qa(df)
    print("\n=== One-sided pooled (L1..L5 majority correct vs L0 alone) on TruthfulQA ===")
    print(json.dumps(one_sided, indent=2))
    (RESULTS / "extra_one_sided_tqa.json").write_text(json.dumps(one_sided, indent=2))

    plot_per_item_delta_truthful_qa(df)
    plot_token_per_dataset(df)
    plot_flipflop_truthful_qa(df)
    plot_tone_ladder_summary(df)
    print(f"\nFigures: {sorted([p.name for p in FIGS.iterdir()])}")


if __name__ == "__main__":
    main()
