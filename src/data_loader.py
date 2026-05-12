"""Data loaders that produce a uniform schema for the tone-ladder experiment.

Each dataset returns a list of dicts with at least:
    - id (str)
    - question (str)         # full question text
    - choices (list[str])    # MCQ options if applicable, else []
    - answer (str)           # canonical correct answer (letter A/B/... for MCQ;
                               numeric string for GSM8K).
    - dataset (str)
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

WORKSPACE = Path(__file__).resolve().parent.parent
CACHE_DIR = WORKSPACE / "cache" / "datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _save_cache(name: str, items: list[dict]) -> None:
    (CACHE_DIR / f"{name}.json").write_text(json.dumps(items, indent=2))


def _load_cache(name: str) -> list[dict] | None:
    p = CACHE_DIR / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def load_truthful_qa(n: int = 100, seed: int = 42) -> list[dict]:
    """TruthfulQA-MC1: pick the single correct option from a list of distractors."""
    cached = _load_cache(f"truthful_qa_n{n}_s{seed}")
    if cached is not None:
        return cached

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), k=n)
    items: list[dict] = []
    for i in indices:
        row = ds[i]
        choices = list(row["mc1_targets"]["choices"])
        labels = list(row["mc1_targets"]["labels"])
        # The correct choice is the one with label==1.
        correct_idx = labels.index(1)
        # We will randomise the option order so position bias doesn't confound
        # tone effects, but keep the mapping reproducible.
        order = list(range(len(choices)))
        rng.shuffle(order)
        shuffled = [choices[j] for j in order]
        new_correct_idx = order.index(correct_idx)
        letter = chr(ord("A") + new_correct_idx)
        items.append(
            {
                "id": f"tqa-{i}",
                "question": row["question"],
                "choices": shuffled,
                "answer": letter,
                "dataset": "truthful_qa",
            }
        )
    _save_cache(f"truthful_qa_n{n}_s{seed}", items)
    return items


def load_mmlu(
    subject: str = "high_school_mathematics", n: int = 100, seed: int = 42
) -> list[dict]:
    cached = _load_cache(f"mmlu_{subject}_n{n}_s{seed}")
    if cached is not None:
        return cached

    ds = load_dataset("cais/mmlu", subject, split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), k=min(n, len(ds)))
    items: list[dict] = []
    for i in indices:
        row = ds[i]
        choices = list(row["choices"])
        ans_idx = int(row["answer"])
        letter = chr(ord("A") + ans_idx)
        items.append(
            {
                "id": f"mmlu-{subject}-{i}",
                "question": row["question"],
                "choices": choices,
                "answer": letter,
                "dataset": f"mmlu_{subject}",
            }
        )
    _save_cache(f"mmlu_{subject}_n{n}_s{seed}", items)
    return items


_GSM8K_NUMBER_RE = re.compile(r"####\s*(-?[\d,\.]+)")


def _extract_gsm8k_answer(answer_text: str) -> str:
    m = _GSM8K_NUMBER_RE.search(answer_text)
    if not m:
        return answer_text.strip()
    return m.group(1).replace(",", "").strip()


def load_gsm8k(n: int = 100, seed: int = 42) -> list[dict]:
    cached = _load_cache(f"gsm8k_n{n}_s{seed}")
    if cached is not None:
        return cached

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), k=n)
    items: list[dict] = []
    for i in indices:
        row = ds[i]
        items.append(
            {
                "id": f"gsm8k-{i}",
                "question": row["question"],
                "choices": [],
                "answer": _extract_gsm8k_answer(row["answer"]),
                "dataset": "gsm8k",
            }
        )
    _save_cache(f"gsm8k_n{n}_s{seed}", items)
    return items


def load_all(n_per_dataset: int = 100, seed: int = 42) -> dict[str, list[dict]]:
    return {
        "truthful_qa": load_truthful_qa(n=n_per_dataset, seed=seed),
        "gsm8k": load_gsm8k(n=n_per_dataset, seed=seed),
        "mmlu_hs_math": load_mmlu("high_school_mathematics", n=n_per_dataset, seed=seed),
    }


if __name__ == "__main__":
    data = load_all(n_per_dataset=100)
    for name, items in data.items():
        print(f"{name}: {len(items)} items, example:")
        print(json.dumps(items[0], indent=2)[:400])
        print()
