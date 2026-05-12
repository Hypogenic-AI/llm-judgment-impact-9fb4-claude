"""Additional analyses requested by Phase 5:

1. Pooled L0-vs-(L1..L5) accuracy comparison per (model, dataset). McNemar on
   pairs where L0 disagrees with the *majority* of the L1-L5 conditions.
2. Combined L0 vs best-non-L0 (L*) per item — paired binomial test.
3. Per-item delta-tokens vs delta-correctness (Spearman).
4. FlipFlop chi-square: tone level × flip event.
5. A 'judged thinks longer' effect-size table (median tokens & 95% CI of the diff).
6. Bootstrap confidence interval for accuracy at each tone level.
7. Save a single summary table for the report.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

WORKSPACE = Path(__file__).resolve().parent.parent
RESULTS = WORKSPACE / "results"


def _load() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "per_item.csv")


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def pooled_skeptic_vs_neutral(df: pd.DataFrame) -> pd.DataFrame:
    """For each (model, dataset, item): is the model correct under L0 only,
    under all of L1-L5 only, etc. Then count.
    """
    rows = []
    for (model, dataset), sub in df.groupby(["model", "dataset"]):
        pivot = sub.pivot_table(
            index="item_id",
            columns="tone_level",
            values="init_correct",
            aggfunc="first",
        ).dropna()
        if "L0" not in pivot.columns:
            continue
        l0 = pivot["L0"].astype(int)
        skeptic_cols = [c for c in ["L1", "L2", "L3", "L4", "L5"] if c in pivot.columns]
        # majority vote across L1-L5
        skeptic_majority = (
            pivot[skeptic_cols].astype(int).mean(axis=1) >= 0.5
        ).astype(int)
        # Per item: 1 if L0 correct, 1 if skeptic_majority correct.
        # b = items where L0 correct & skeptic wrong, c = opposite.
        b = int(((l0 == 1) & (skeptic_majority == 0)).sum())
        c = int(((l0 == 0) & (skeptic_majority == 1)).sum())
        if b + c > 0:
            p = stats.binomtest(min(b, c), n=b + c, p=0.5).pvalue
        else:
            p = 1.0
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "n_items": int(len(pivot)),
                "acc_L0": float(l0.mean()),
                "acc_skeptic_maj": float(skeptic_majority.mean()),
                "delta_pp": float((skeptic_majority.mean() - l0.mean()) * 100),
                "L0_only_correct": b,
                "skeptic_only_correct": c,
                "mcnemar_p": float(p),
            }
        )
    return pd.DataFrame(rows)


def best_tone_per_item(df: pd.DataFrame) -> pd.DataFrame:
    """For each (model, dataset), what fraction of items improve under any of
    L1-L5 vs L0 (and vice versa)?
    """
    rows = []
    for (model, dataset), sub in df.groupby(["model", "dataset"]):
        pivot = sub.pivot_table(
            index="item_id",
            columns="tone_level",
            values="init_correct",
            aggfunc="first",
        ).dropna()
        if "L0" not in pivot.columns:
            continue
        l0 = pivot["L0"].astype(int)
        skeptic_cols = [c for c in ["L1", "L2", "L3", "L4", "L5"] if c in pivot.columns]
        skeptic_max = pivot[skeptic_cols].astype(int).max(axis=1)
        skeptic_min = pivot[skeptic_cols].astype(int).min(axis=1)
        n = int(len(pivot))
        gained_under_skepticism = int(((l0 == 0) & (skeptic_max == 1)).sum())
        lost_under_some_skepticism = int(((l0 == 1) & (skeptic_min == 0)).sum())
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "n_items": n,
                "items_correct_under_some_L1_L5_but_wrong_at_L0": gained_under_skepticism,
                "items_correct_at_L0_but_wrong_under_some_L1_L5": lost_under_some_skepticism,
                "frac_items_with_skeptic_recovery": gained_under_skepticism / max(1, n),
                "frac_items_with_skeptic_regression": lost_under_some_skepticism / max(1, n),
            }
        )
    return pd.DataFrame(rows)


def token_delta_vs_acc_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Per item: does the change in tokens (across tones) correlate with the
    change in correctness?
    """
    rows = []
    for (model, dataset), sub in df.groupby(["model", "dataset"]):
        if sub["item_id"].nunique() < 5:
            continue
        # per item, compute std of correctness (i.e. did tone matter?) and mean
        # token delta (L4-L0).
        item_table = sub.pivot_table(
            index="item_id",
            columns="tone_level",
            values=["init_correct", "init_completion_tokens"],
            aggfunc="first",
        ).dropna()
        if ("init_correct", "L0") not in item_table.columns:
            continue
        rho_acc, p_acc = stats.spearmanr(
            item_table[("init_completion_tokens", "L4")] - item_table[("init_completion_tokens", "L0")] if ("init_completion_tokens", "L4") in item_table.columns else item_table[("init_completion_tokens", "L0")],
            item_table[("init_correct", "L4")] - item_table[("init_correct", "L0")] if ("init_correct", "L4") in item_table.columns else item_table[("init_correct", "L0")],
        ) if ("init_correct", "L4") in item_table.columns else (None, None)
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "spearman_dToken_dCorrect_L4vsL0": float(rho_acc) if rho_acc is not None else None,
                "p": float(p_acc) if p_acc is not None else None,
            }
        )
    return pd.DataFrame(rows)


def flipflop_chi(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-square of tone level vs flip outcome (correct→wrong, wrong→correct,
    no flip). Run only on rows that have followup."""
    if "flipped" not in df.columns:
        return pd.DataFrame()
    rows = []
    fu = df.dropna(subset=["flipped"]).copy()
    for (model, dataset), sub in fu.groupby(["model", "dataset"]):
        # 3-class outcome: 0=no flip, 1=C->W, 2=W->C
        out = np.where(
            sub["flipped"] == 0,
            "no_flip",
            np.where(sub["flip_correct_to_wrong"] == 1, "C2W", "W2C"),
        )
        ct = pd.crosstab(sub["tone_level"], out)
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p, dof, _ = stats.chi2_contingency(ct.values)
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "chi2": float(chi2),
                "dof": int(dof),
                "p": float(p),
                "n": int(len(sub)),
                "table": ct.to_dict(),
            }
        )
    return pd.DataFrame(rows)


def tone_level_token_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, dataset, tone_level): completion-token mean, median, and
    bootstrap 95% CI."""
    rows = []
    for (model, dataset, tone), sub in df.groupby(["model", "dataset", "tone_level"]):
        toks = sub["init_completion_tokens"].values
        lo, hi = bootstrap_ci(toks)
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "tone_level": tone,
                "n": int(len(toks)),
                "tokens_mean": float(toks.mean()),
                "tokens_median": float(np.median(toks)),
                "tokens_ci_lo": lo,
                "tokens_ci_hi": hi,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "dataset", "tone_level"])


def accuracy_with_ci(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, dataset, tone), sub in df.groupby(["model", "dataset", "tone_level"]):
        v = sub["init_correct"].values
        lo, hi = bootstrap_ci(v)
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "tone_level": tone,
                "n": int(len(v)),
                "acc": float(v.mean()),
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "dataset", "tone_level"])


def main():
    df = _load()
    print(f"Loaded {len(df)} per-item records.")

    pooled = pooled_skeptic_vs_neutral(df)
    print("\n=== Pooled skeptic-majority (L1..L5) vs neutral (L0) ===")
    print(pooled.to_string(index=False))
    pooled.to_csv(RESULTS / "extra_pooled_skeptic_vs_neutral.csv", index=False)

    bt = best_tone_per_item(df)
    print("\n=== Per-item recovery / regression under any L1..L5 ===")
    print(bt.to_string(index=False))
    bt.to_csv(RESULTS / "extra_best_tone_per_item.csv", index=False)

    tok_table = tone_level_token_table(df)
    tok_table.to_csv(RESULTS / "extra_tokens_with_ci.csv", index=False)

    acc_table = accuracy_with_ci(df)
    acc_table.to_csv(RESULTS / "extra_accuracy_with_ci.csv", index=False)
    print("\n=== Accuracy with 95% bootstrap CI ===")
    print(acc_table.to_string(index=False))

    chi = flipflop_chi(df)
    if not chi.empty:
        print("\n=== FlipFlop chi-square (tone level × flip outcome) ===")
        for _, r in chi.iterrows():
            print(
                f"  {r['model']}/{r['dataset']}: chi2={r['chi2']:.2f} dof={r['dof']} p={r['p']:.4g} n={r['n']}"
            )
            print(f"    table: {r['table']}")
        chi.drop(columns=["table"]).to_csv(
            RESULTS / "extra_flipflop_chi.csv", index=False
        )

    summary = {
        "pooled_skeptic_vs_neutral": pooled.to_dict(orient="records"),
        "best_tone_per_item": bt.to_dict(orient="records"),
    }
    (RESULTS / "extra_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved extras under results/extra_*.csv and extra_summary.json")


if __name__ == "__main__":
    main()
