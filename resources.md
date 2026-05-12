# Resources Catalogue

## Summary

| Resource type | Count | Total size |
|---------------|-------|------------|
| Papers (PDFs) | 20 | ~44 MB |
| Datasets (referenced) | 8 + sample JSONs | tiny — pulled on demand |
| Cloned repositories | 5 | ~40 MB excluding BIG-bench |

Everything in this workspace targets the research question: *does skeptical/judgmental prompt tone change LLM reasoning quality, and is there a "sweet spot" where it helps?* See `literature_review.md` for the full synthesis.

---

## Papers

20 PDFs in `papers/`. Detailed methodology notes in `literature_review.md`. The table below is sortable by `papers/README.md` priority tier (1 = deep-read; 2 = important context; 3 = supporting).

| # | Tier | Year | Title | File | Why it matters |
|---|------|------|-------|------|----------------|
| 1 | 1 | 2023 | Are You Sure? Challenging LLMs Leads to Performance Drops in the FlipFlop Experiment | `papers/laban2023_flipflop_are_you_sure.pdf` | Defines our two-turn challenge baseline; quantifies the FlipFlop effect (avg −17%). |
| 2 | 1 | 2023 | Towards Understanding Sycophancy in Language Models | `papers/sharma2023_understanding_sycophancy.pdf` | Foundational; explains why RLHF causes sycophancy. |
| 3 | 1 | 2025 | Measuring Sycophancy of Language Models in Multi-turn Dialogues (SYCON-Bench) | `papers/hong2025_sycon_bench_multiturn.pdf` | ToF/NoF metrics; Andrew-prompt mitigation hints at sweet-spot. |
| 4 | 1 | 2025 | From Fact to Judgment: Task Framing on LLM Conviction | `papers/rabbani2025_fact_to_judgment_task_framing.pdf` | Closest experimental analogue; shows model-dependent direction. |
| 5 | 1 | 2025 | SycEval: Evaluating LLM Sycophancy | `papers/fanous2025_syceval.pdf` | Preemptive vs in-context rebuttals; progressive vs regressive sycophancy. |
| 6 | 1 | 2025 | Challenging the Evaluator: LLM Sycophancy Under User Rebuttal | `papers/kim2025_challenging_evaluator_rebuttal.pdf` | Separates social framing / reasoning content / casual tone effects. |
| 7 | 2 | 2023 | Simple synthetic data reduces sycophancy in LLMs | `papers/wei2023_synthetic_data_reduces_sycophancy.pdf` | Mitigation baseline; fine-tuning reduces sycophancy without harming acc. |
| 8 | 2 | 2023 | Can ChatGPT Defend its Belief in Truth? (Debate) | `papers/wang2023_can_chatgpt_defend_belief_debate.pdf` | Original "defend against invalid arguments" setup. |
| 9 | 2 | 2023 | Large Language Models Understand and Can be Enhanced by Emotional Stimuli (EmotionPrompt) | `papers/li2023_emotionprompt_emotional_stimuli.pdf` | Direct precedent for tonal stimuli boosting performance. |
| 10 | 2 | 2026 | Diagnosing and Mitigating Sycophancy and Skepticism in LLM Causal Judgment (CAUSALT3) | `papers/chang2026_causal_sycophancy_skepticism_causalt3.pdf` | Names the Skepticism Trap and Sycophancy Trap. |
| 11 | 2 | 2026 | Good Arguments Against the People Pleasers | `papers/feng2026_good_arguments_people_pleasers.pdf` | Reasoning can rationalise sycophancy — must measure trace, not just length. |
| 12 | 3 | 2025 | BrokenMath: Sycophancy in Theorem Proving | `papers/petrov2025_brokenmath_sycophancy_theorem.pdf` | False-premise olympiad problems. |
| 13 | 3 | 2025 | Sycophancy under Pressure / Pressure-Tune | `papers/zhang2025_sycophancy_under_pressure_pressure_tune.pdf` | Scientific QA adversarial dialogues. |
| 14 | 3 | 2025 | AssertBench: Evaluating Self-Assertion | `papers/lee2025_assertbench.pdf` | 41k FEVEROUS-derived claims. |
| 15 | 3 | 2026 | Beyond Social Pressure: Epistemic Attack (PPT-Bench) | `papers/au2026_beyond_social_pressure_epistemic_attack.pdf` | Four pressure taxonomies. |
| 16 | 3 | 2025 | Reasoning Isn't Enough: Truth-Bias and Sycophancy | `papers/barkett2025_reasoning_isnt_enough_truth_bias.pdf` | Reasoning helps but doesn't solve sycophancy. |
| 17 | 3 | 2025 | MONICA: Real-Time CoT Sycophancy Monitoring | `papers/2025_monica_realtime_calibration.pdf` | Per-step inference-time mitigation. |
| 18 | 3 | 2026 | Feedback Indices for Rebuttals on MCQs | `papers/dunlap2026_feedback_indices_rebuttals.pdf` | Newer/reasoning-effort models are less sycophantic. |
| 19 | 3 | 2024 | Conformity in Large Language Models | `papers/zhu2024_conformity_llms.pdf` | Asch-style benchmark + mitigations. |
| 20 | 3 | 2025 | Large Language Models Often Know When They Are Being Evaluated | `papers/needham2025_llms_know_being_evaluated.pdf` | Eval-awareness — relevant to "social facilitation under observation". |

See `papers/README.md` for descriptions of each PDF.

## Datasets

Documented in `datasets/README.md`. Tiny JSON samples (3 records each) in `datasets/samples/`. Data files are excluded from git via `datasets/.gitignore`.

| Name | Source | Where the dataset is materialised | Used in literature |
|------|--------|----------------------------------|--------------------|
| TruthfulQA-MC | HF `truthful_qa` | downloaded on demand | Laban, Sharma, Kim |
| MMLU | HF `cais/mmlu` | downloaded on demand | Sharma, Laban, Kim |
| GSM8K | HF `openai/gsm8k` | downloaded on demand | Multi-agent debate work |
| BrokenMath | HF `INSAIT-Institute/BrokenMath` | downloaded on demand | Petrov 2025 |
| SYCON debate / unethical prompts | `code/SYCON-Bench/` (cloned) | local files | Hong 2025 |
| Sharma sycophancy-eval (answer / are_you_sure / feedback) | `code/sycophancy-eval/datasets/` (cloned) | local JSONL | Sharma 2023 |
| AssertBench claims | `code/assert-bench/input_data/input.csv` (cloned) | local CSV (~14 MB) | Lee 2025 |
| MedQuad | HF `lavita/MedQuAD` | not pulled yet — verify license before use | Fanous 2025 |

## Cloned repositories

Documented in `code/README.md`. Five repos, ~40 MB excluding BIG-bench:

| Path | Source | Purpose |
|------|--------|---------|
| `code/SYCON-Bench/` | github.com/JiseungHong/SYCON-Bench | Multi-turn sycophancy harness + ToF/NoF metrics + debate prompts |
| `code/sycophancy-eval/` | github.com/meg-tong/sycophancy-eval | Sharma's pre-formatted "Are you sure?" / feedback / biased-question datasets |
| `code/llm-conversational-judgment/` | github.com/LadyPary/llm-conversational-judgment | CJT factual-vs-conversational framing experiment + pre-computed baselines |
| `code/assert-bench/` | github.com/achowd32/assert-bench | Paired-framing self-assertion harness over FEVEROUS claims |
| `code/BIG-bench/` | github.com/google/BIG-bench | Reference; only `logical_fallacy_detection` needed for FlipFlop reproducibility |

## Resource gathering notes

### Search strategy
- Primary: paper-finder service at `localhost:8000` in **diligent** mode, which returned 129 papers including 48 with relevance ≥ 2. Three queries were run; the most productive was `"LLM sycophancy skeptical prompt reasoning"` (cached at `paper_search_results/diligent_full.json`).
- Secondary: targeted WebSearch for specific seminal/recent papers by title (Laban "FlipFlop", Sharma "Sycophancy", Hong "SYCON-Bench", Rabbani "Fact to Judgment", Fanous "SycEval", Kim "Challenging the Evaluator", Chang "CAUSALT3", Feng "People Pleasers", Petrov "BrokenMath", Lee "AssertBench", Zhu "Conformity", Wei "synthetic data", Li "EmotionPrompt", Needham "eval-awareness").
- arXiv API direct lookup was rate-limited, so PDFs were fetched directly from `arxiv.org/pdf/<id>.pdf` once IDs were known.

### Selection criteria
- Papers with **direct mechanistic relevance** to the user's hypothesis (sycophancy / framing / tone effects / reasoning trace) were prioritised over general LLM-evaluation papers.
- Recent (2024–2026) work was emphasised so the experiment runner can compare against state-of-the-art baselines.
- One older "positive arm" paper (EmotionPrompt 2023) and one eval-awareness paper (Needham 2025) were included to cover the *positive* and *social-facilitation* arms of the hypothesis, which the sycophancy literature alone does not address.

### Challenges encountered
- Semantic Scholar API returned HTTP 429 immediately, so we couldn't bulk-resolve Semantic Scholar IDs to arXiv IDs that way. Solved by targeted WebSearch for each title.
- arXiv API was also rate-limited (HTTP 429) after a handful of requests; resolved by using known arXiv IDs directly.
- The paper-finder script's `find_papers.py` truncates the result file when piped through `head`, which made the first parsed JSON invalid. Worked around by re-running without piping.

### Gaps and workarounds
- **No paper directly tests the user's "sweet spot" hypothesis.** Closest is Kim & Khashabi's casual-vs-formal axis (H3), Rabbani's framing axis, and EmotionPrompt's tone axis — none of which combine into a graded preemptive-skepticism ladder. This is the experimental novelty.
- **No paper reports reasoning-token cost as a function of prompt tone.** With explicit-CoT reasoning models now widely available (o4-mini, DeepSeek-R1, etc.) this is now directly measurable; recommend the experiment runner instrument this from day one.
- **MedQuad** isn't fully downloaded; check license before scraping the full 43k-item set. A 500-item subsample is sufficient for our hypothesis tests.

## Recommendations for experiment design

**Primary datasets (objective):** TruthfulQA-MC1 + MMLU subjects (HS-math, prof-medicine, formal-logic, moral-scenarios) + GSM8K test subset. Total ~1500 items.

**Subjective probes:** 100 SYCON debate topics + 100–200 BrokenMath items.

**Baseline methods to compare against:**
- Plain zero-shot — no tone.
- Standard FlipFlop two-turn protocol (Laban's `Are you sure?` challenger).
- EmotionPrompt (Li 2023, "This is very important to my career").
- "Andrew prompt" third-person persona (Hong 2025).

**Tone-intensity ladder (the core manipulation).** Preemptive tone manipulation applied *before* the question, six levels from neutral to hostile — see §8 of `literature_review.md` for the proposed wording ladder.

**Evaluation metrics:**
- Initial / final accuracy and **FlipFlop Δ** (Laban).
- Conditional flip probabilities (Correct→Flip, Wrong→Flip) (Laban).
- **%Sorry / apology rate** — operational marker of where the sweet spot ends (Laban).
- **Reasoning token count** per condition — new metric, addresses the "do LLMs think *longer*" sub-question.
- Progressive vs regressive sycophancy rates (Fanous 2025).

**Code to adapt/reuse:**
- `code/sycophancy-eval/` for the standard two-turn protocol and pre-built prompt formatting.
- `code/SYCON-Bench/debate_setting/` for the ToF/NoF metrics and multi-turn pressure scripts.
- `code/llm-conversational-judgment/` for the framing manipulation prompt builders.
- `code/assert-bench/main.py` as a minimal example of paired-framing query construction.
