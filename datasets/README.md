# Datasets

Data files are **not** committed to git (see `datasets/.gitignore`). The experiment runner downloads them on demand from HuggingFace or directly from the cloned code repositories under `code/`. Small JSON samples (≤10 records each) live in `datasets/samples/` so reviewers can see the shape of each dataset.

## Dataset selection rationale

For the hypothesis ("does the *tone* of a user prompt change how LLMs reason?") we need:

1. **Objective tasks** where there is one correct answer (so we can measure flip-induced accuracy drops cleanly): TruthfulQA-MC1, MMLU, GSM8K, MATH/AMPS, BrokenMath.
2. **Subjective tasks** where the model must *defend a position* against pressure (per Feng et al., subjective tasks elicit the strongest sycophancy gradient): SYCON-Bench debate topics.
3. **High-stakes / advice tasks** to mirror real-world deployment risks: MedQuad (per SycEval).
4. **Pre-built rebuttal datasets** already paired with challenger utterances: Sharma's sycophancy-eval and AssertBench.

---

## 1. TruthfulQA (Lin et al. 2022)

- **Source:** HuggingFace `truthful_qa` (config `multiple_choice`). [Paper](https://arxiv.org/abs/2109.07958), [HF page](https://huggingface.co/datasets/truthful_qa).
- **Size:** 817 questions, validation split. Multiple-choice variant has `mc1_targets` (single correct) and `mc2_targets` (multiple correct).
- **Format:** `{question, mc1_targets: {choices, labels}, mc2_targets: {choices, labels}}`.
- **License:** Apache 2.0.
- **Used in:** Laban (FlipFlop), Sharma (sycophancy-eval), Kim & Khashabi.
- **Why include:** Has well-known truthful-vs-misconception items, so we can observe whether a skeptical tone helps the model *resist* the misconception choice (sweet-spot win) or pushes it *toward* it (sycophantic loss).

```python
from datasets import load_dataset
ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
```

Sample: `samples/truthful_qa_sample.json` (3 records).

## 2. MMLU (Hendrycks et al. 2021)

- **Source:** HuggingFace `cais/mmlu`. [Paper](https://arxiv.org/abs/2009.03300).
- **Size:** 57 subjects × ~test sets; we will subsample. Pre-pulled config: `high_school_mathematics` (270 test items).
- **Format:** `{question, choices, answer, subject}`.
- **License:** MIT.
- **Used in:** Sharma, Laban, Kim.
- **Why include:** Standard broad-coverage benchmark; lets us split by domain (humanities vs STEM vs professional) and check whether the tone effect varies.

Recommended subsamples for our experiment: `high_school_mathematics`, `professional_medicine`, `formal_logic`, `moral_scenarios`.

```python
from datasets import load_dataset
ds = load_dataset("cais/mmlu", "high_school_mathematics", split="test")
```

Sample: `samples/mmlu_sample.json` (3 records).

## 3. GSM8K (Cobbe et al. 2021)

- **Source:** HuggingFace `openai/gsm8k` config `main`. [Paper](https://arxiv.org/abs/2110.14168).
- **Size:** 7,473 train / 1,319 test. We will use the test split.
- **Format:** `{question, answer}` where `answer` contains step-by-step working and a `####` final answer.
- **License:** MIT.
- **Why include:** Lets us measure reasoning-token cost as a function of prompt tone — gold-standard for "longer reasoning" claims.

```python
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="test")
```

Sample: `samples/gsm8k_sample.json` (3 records).

## 4. BrokenMath (Petrov, Dekoninck, Vechev 2025)

- **Source:** HuggingFace `INSAIT-Institute/BrokenMath`. [Paper arXiv 2510.04721](https://arxiv.org/abs/2510.04721).
- **Size:** 451 problems from 39+ olympiads, each perturbed with a false premise + paired with its original problem.
- **Format:** `{problem_id, problem (perturbed), original_problem, gold_answer, solution, question_type, is_adversarial}`.
- **Why include:** Purpose-built for measuring sycophancy on theorem-proving tasks. Sycophantic models will "prove" the false statement. Gives us a clean signal for whether judgment tone makes models more or less willing to push back on a false premise.

```python
from datasets import load_dataset
ds = load_dataset("INSAIT-Institute/BrokenMath", split="benchmark")
```

Sample: `samples/brokenmath_sample.json` (3 records).

## 5. SYCON-Bench debate / unethical / false-presupposition prompts

- **Source:** Already cloned at `code/SYCON-Bench/`. [Paper arXiv 2505.23840](https://arxiv.org/abs/2505.23840).
- **Files:**
  - `code/SYCON-Bench/debate_setting/data/questions.txt` and `arguments.txt` — 100 debate topics with pre-written stances. The model is asked to defend the stance across 5 turns under user pressure.
  - `code/SYCON-Bench/ethical-setting/data/stereoset_intra_user_queries_api_over45.csv` — 200 stereotyping prompts.
  - `code/SYCON-Bench/debate_setting/data/topics/` — supplementary topic metadata.
- **Format:** Plain text, one item per line for the txt files.
- **Why include:** Subjective scenarios where the *expected* behaviour under pressure is to hold the line; lets us see whether judgment-tone prompts cause models to capitulate *faster* or *slower*. Reusable Turn-of-Flip / Number-of-Flip metrics already implemented.

## 6. Sharma sycophancy-eval datasets

- **Source:** Already cloned at `code/sycophancy-eval/datasets/`. Companion to [arXiv 2310.13548](https://arxiv.org/abs/2310.13548).
- **Files:**
  - `answer.jsonl` — 7,267 short-answer items (TriviaQA-derived) with `correct_answer` and `incorrect_answer` fields. Use for "biased question" sycophancy.
  - `are_you_sure.jsonl` — 4,887 MCQ samples (AQuA, MATH, MMLU, TruthfulQA, TriviaQA) pre-formatted for the two-turn challenge protocol.
  - `feedback.jsonl` — 8,500 argument-passage items with logical-error labels for measuring whether the model tailors its feedback to a stated user preference.
- **License:** MIT (per repo).
- **Why include:** Drop-in, already pre-formatted for the FlipFlop-style challenge protocol — saves us from re-implementing prompt boilerplate.

## 7. AssertBench (Lee & Chowdhary 2025)

- **Source:** Already cloned at `code/assert-bench/input_data/input.csv`. [Paper arXiv 2506.11110](https://arxiv.org/abs/2506.11110).
- **Size:** 41,836 evidence-supported claims derived from FEVEROUS.
- **Format:** Single-column CSV of factual claims. The harness in `code/assert-bench/main.py` constructs paired user framings ("the user claims this is correct" / "the user claims this is incorrect").
- **Why include:** Largest single source of evidence-paired claims; lets us measure self-assertion separately from sycophancy.

## 8. MedQuad (Ben Abacha & Demner-Fushman 2019)

- **Source:** HuggingFace `lavita/MedQuAD` or the original [MedQuad GitHub release](https://github.com/abachaa/MedQuAD). 43k+ clinical Q&A items.
- **Used in:** Fanous (SycEval).
- **Why include:** Provides the high-stakes-advice axis. Not pre-downloaded; the experiment runner can sample 200–500 items as needed. License is the original NIH NLM redistribution terms — check before publication.

```python
from datasets import load_dataset
ds = load_dataset("lavita/MedQuAD")  # verify version before use
```

---

## Quick load test (sanity check)

```python
from datasets import load_dataset

for name, args in [
    ("truthful_qa", ("truthful_qa", "multiple_choice", "validation")),
    ("mmlu_hs_math", ("cais/mmlu", "high_school_mathematics", "test")),
    ("gsm8k", ("openai/gsm8k", "main", "test")),
    ("brokenmath", ("INSAIT-Institute/BrokenMath", None, "benchmark")),
]:
    ds = load_dataset(args[0], args[1], split=args[2]) if args[1] else load_dataset(args[0], split=args[2])
    print(f"{name}: {len(ds)} rows, keys={list(ds[0].keys())}")
```

This was run during resource finding and all four datasets are accessible from this workspace.

## Storage notes

- Most of these datasets are small (TruthfulQA: ~1MB, GSM8K test: ~3MB, BrokenMath: ~5MB). MMLU is larger but we only need a few subjects.
- AssertBench input.csv is ~14MB and already in the repo (it's not large enough to gitignore aggressively, but the .gitignore excludes data subdirs by default — adjust if needed).
- HuggingFace caches under `~/.cache/huggingface/datasets/` — leave there for reuse.
