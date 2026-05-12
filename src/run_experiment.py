"""Experiment 1 (single-turn) and Experiment 2 (FlipFlop follow-up) runner.

Outputs JSONL files in results/ with one record per (model, dataset, tone, item).
Resumable via the on-disk LLM cache.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE / "src"))

from data_loader import load_all  # noqa: E402
from llm_client import chat  # noqa: E402
from metrics import all_metrics, has_apology, is_correct  # noqa: E402
from prompts import (  # noqa: E402
    TONES,
    build_followup_messages,
    build_messages,
)

LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = WORKSPACE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / f"{name}.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _max_tokens_for(dataset: str) -> int:
    if dataset == "gsm8k":
        return 2048
    return 1024


def run_one_cell(
    model: str,
    item: dict,
    tone,
    do_followup: bool,
) -> dict:
    """Run a single (model, item, tone) cell. Optionally also follow up with the
    FlipFlop challenge.
    """
    initial_msgs = build_messages(item, tone)
    max_tok = _max_tokens_for(item["dataset"])
    initial = chat(model, initial_msgs, temperature=0.0, max_tokens=max_tok)
    init_resp = initial["content"]
    init_correct = is_correct(item, init_resp)
    record = {
        "model": model,
        "dataset": item["dataset"],
        "item_id": item["id"],
        "tone_level": tone.level,
        "tone_label": tone.label,
        "answer_gold": item["answer"],
        "initial_response": init_resp,
        "initial_metrics": all_metrics(item, init_resp, initial["completion_tokens"]),
        "initial_prompt_tokens": initial["prompt_tokens"],
    }
    if do_followup:
        fu_msgs = build_followup_messages(initial_msgs, init_resp, item)
        fu = chat(model, fu_msgs, temperature=0.0, max_tokens=max_tok)
        fu_resp = fu["content"]
        record["followup_response"] = fu_resp
        record["followup_metrics"] = all_metrics(item, fu_resp, fu["completion_tokens"])
        record["followup_prompt_tokens"] = fu["prompt_tokens"]
        record["flipped"] = int(init_correct != is_correct(item, fu_resp))
        record["flip_correct_to_wrong"] = int(init_correct and not is_correct(item, fu_resp))
        record["flip_wrong_to_correct"] = int((not init_correct) and is_correct(item, fu_resp))
        record["followup_apology"] = int(has_apology(fu_resp))
    return record


def run_experiment(
    model: str,
    items_by_dataset: dict[str, list[dict]],
    tones: list,
    out_path: Path,
    do_followup: bool,
    max_workers: int = 8,
    logger: logging.Logger | None = None,
) -> None:
    if logger is None:
        logger = _setup_logger(out_path.stem)
    cells = []
    for dataset, items in items_by_dataset.items():
        for item in items:
            for tone in tones:
                cells.append((dataset, item, tone))

    logger.info(
        "Running %d cells (model=%s, datasets=%s, tones=%s, followup=%s, workers=%d)",
        len(cells),
        model,
        list(items_by_dataset.keys()),
        [t.level for t in tones],
        do_followup,
        max_workers,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Append-mode so re-runs don't lose data. We rely on the LLM cache for
    # idempotency on the API side; for the JSONL we just rewrite from scratch
    # for cleanliness.
    out_path.write_text("")

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_cell = {
            pool.submit(run_one_cell, model, item, tone, do_followup): (dataset, item, tone)
            for (dataset, item, tone) in cells
        }
        with out_path.open("a") as f:
            for fut in as_completed(fut_to_cell):
                dataset, item, tone = fut_to_cell[fut]
                try:
                    rec = fut.result()
                except Exception as e:  # noqa: BLE001
                    logger.exception(
                        "Cell failed (dataset=%s item=%s tone=%s): %s",
                        dataset,
                        item["id"],
                        tone.level,
                        e,
                    )
                    continue
                f.write(json.dumps(rec) + "\n")
                done += 1
                if done % 50 == 0:
                    rate = done / (time.time() - t0)
                    logger.info(
                        "%d/%d cells done (%.1f cells/s)", done, len(cells), rate
                    )
    logger.info(
        "Finished %d/%d cells in %.1fs", done, len(cells), time.time() - t0
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n-per-dataset", type=int, default=100)
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Subset of dataset keys to run (e.g. truthful_qa gsm8k mmlu_hs_math).")
    p.add_argument("--tones", nargs="*", default=None,
                   help="Subset of tone levels (e.g. L0 L1 L2).")
    p.add_argument("--out", required=True)
    p.add_argument("--followup", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    items = load_all(n_per_dataset=args.n_per_dataset)
    if args.datasets:
        items = {k: v for k, v in items.items() if k in args.datasets}
    tones = TONES
    if args.tones:
        tones = [t for t in TONES if t.level in set(args.tones)]

    logger = _setup_logger(Path(args.out).stem)
    run_experiment(
        model=args.model,
        items_by_dataset=items,
        tones=tones,
        out_path=Path(args.out),
        do_followup=args.followup,
        max_workers=args.workers,
        logger=logger,
    )


if __name__ == "__main__":
    main()
