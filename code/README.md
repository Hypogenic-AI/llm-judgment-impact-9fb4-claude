# Cloned code repositories

Five shallow clones of the most relevant sycophancy / framing benchmarks. Each is documented with what it provides, install requirements, and how it maps to the research hypothesis. BIG-bench is included for completeness but is large and likely not needed.

---

## 1. `SYCON-Bench/` — multi-turn sycophancy benchmark

- **Source:** https://github.com/JiseungHong/SYCON-Bench
- **Paper:** Hong et al. 2025, [arXiv 2505.23840](https://arxiv.org/abs/2505.23840). EMNLP 2025 Findings.
- **What it provides:**
  - `debate_setting/` — 100 controversial topics with pre-paired stances. Scripts: `run_benchmark.py`, `evaluate_ToF.py`, `evaluate_oscillate.py`, `model_registry.py`.
  - `ethical-setting/` — 200 stereotype prompts derived from StereoSet, plus `evaluate_ToF.py` and `run_benchmark.py`.
  - `anova_syco_analysis.py` — statistical analysis (ANOVA) of consistency results across models, included at repo root.
- **Key reusable artifacts:**
  - Pre-written **Turn-of-Flip (ToF)** and **Number-of-Flip (NoF)** evaluation pipelines.
  - The "Andrew prompt" mitigation (third-person persona) — directly relevant to the "sweet spot" question: it reduces sycophancy by 63.8% in debate.
- **Dependencies:** `code/SYCON-Bench/debate_setting/requirements.txt` lists OpenAI / Anthropic / Google API clients plus pandas/numpy. Install in an isolated env, not the main `.venv`, if their pins conflict.
- **How we use it:** Reuse the debate prompts and ToF metric. The user pressure utterances baked into the harness (social proof, essentialism, etc.) are themselves "skeptical-tone" variants we can compare against.

## 2. `sycophancy-eval/` — Sharma et al. original eval suite

- **Source:** https://github.com/meg-tong/sycophancy-eval
- **Paper:** Sharma et al. ICLR 2024, [arXiv 2310.13548](https://arxiv.org/abs/2310.13548).
- **What it provides:**
  - `datasets/answer.jsonl` (7,267 items), `are_you_sure.jsonl` (4,887 items), `feedback.jsonl` (8,500 items) — all pre-formatted for direct LLM calls.
  - `example.ipynb` — reference implementation of how to run the eval.
  - `utils.py` — prompt-template and evaluation helpers.
- **Dependencies:** Minimal — pandas, jsonlines, an OpenAI/Anthropic client.
- **How we use it:** The `are_you_sure.jsonl` items are an off-the-shelf two-turn challenge benchmark. We can slot our tone-conditioned prompt in place of the standard challenger and measure how the FlipFlop effect changes.

## 3. `llm-conversational-judgment/` — Rabbani et al. CJT framework

- **Source:** https://github.com/LadyPary/llm-conversational-judgment
- **Paper:** Rabbani et al. IWSDS 2026, [arXiv 2511.10871](https://arxiv.org/abs/2511.10871).
- **What it provides:**
  - `scripts/run_experiment.py` — main runner for the factual vs CJT framing experiment.
  - `scripts/analyze_results.py` — analysis & plotting.
  - `data/results/*.csv` — pre-computed results for 5 models (Gemma, GPT-4o-mini, Llama-3.1-8B, Llama-3.2-3B, Mistral Small 3). Useful as a baseline reference.
  - `src/` — prompt builders for the four conditions (C1-true/false statement × C2-correct/incorrect speaker).
  - `requirements.txt` — installable dependency list.
- **How we use it:** Their two-condition framing manipulation is the closest precedent for our tone manipulation. We can extend their CJT prompts with our skepticism-intensity ladder.

## 4. `assert-bench/` — Lee & Chowdhary self-assertion benchmark

- **Source:** https://github.com/achowd32/assert-bench
- **Paper:** Lee & Chowdhary 2025, [arXiv 2506.11110](https://arxiv.org/abs/2506.11110).
- **What it provides:**
  - `main.py` — the harness that takes a claim, constructs paired user framings ("the user claims this is correct/incorrect"), and queries an LLM.
  - `input_data/input.csv` — 41,836 evidence-supported claims sourced from FEVEROUS.
  - `json_conv.py` — output formatter.
- **How we use it:** Large-scale evidence-paired claims let us scale up beyond the smaller benchmarks. The harness is minimal and easy to retrofit with our tone-intensity ladder.

## (Not cloned) BIG-bench — pull on demand

BIG-bench was initially cloned but is 5.3 GB and the only relevant task is `logical_fallacy_detection` (used in Laban's FlipFlop). If we want to reproduce that exact subset, do a sparse checkout:

```bash
git clone --filter=blob:none --sparse https://github.com/google/BIG-bench.git
cd BIG-bench && git sparse-checkout set bigbench/benchmark_tasks/logical_fallacy_detection
```

Alternatively the task is available on HuggingFace as part of `lukaemon/bbh` (`logical_deduction_*` subsets).

---

## Repositories *not* cloned but linked

- **FlipFlop (Laban et al.).** No public code release as of the paper's stated plan; protocol is fully described in §3 of `papers/laban2023_flipflop_are_you_sure.pdf` and is straightforward to reimplement (≈100 lines).
- **SycEval (Fanous et al.).** Their pipeline is described in their paper but no canonical repo is publicly indexed; the rebuttal templates are listed inline.
- **BrokenMath.** Data is on HuggingFace (`INSAIT-Institute/BrokenMath`); their full evaluation harness is in the openreview supplementary — not cloned.
- **MONICA, PPT-Bench, etc.** All recent (2025–2026); we have the papers but the codebases are either unreleased or peripheral.

## Quick verification

```bash
ls code/                                    # five directories listed
cat code/SYCON-Bench/debate_setting/data/questions.txt | head
head -1 code/sycophancy-eval/datasets/are_you_sure.jsonl
head -2 code/assert-bench/input_data/input.csv
```

All five clones validated during resource finding.

## Suggested isolated-environment policy for the experiment runner

The main `.venv` is intentionally lightweight (just `pypdf`, `requests`, `arxiv`, `httpx`, `datasets`). Each external repo has its own `requirements.txt` — when actually running their harnesses, create per-repo virtualenvs to avoid pinning conflicts. None of the harnesses need GPU for inference-only use against API models; if running open-source models locally, GPU + vLLM/transformers will be needed.
