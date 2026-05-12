# Downloaded papers

20 PDFs covering sycophancy, multi-turn challenge experiments, prompt-tone manipulations, framing effects, evaluation awareness, and EmotionPrompt. Numbering follows the priority order in `literature_review.md` so the first ~8 entries are the must-reads.

Detailed methodological notes for the deep-read papers (1–6 below) live in `literature_review.md`.

## Priority 1 — deep-read, central to the hypothesis

1. **laban2023_flipflop_are_you_sure.pdf** — Laban, Murakhovs'ka, Xiong, Wu (Salesforce). *Are You Sure? Challenging LLMs Leads to Performance Drops in the FlipFlop Experiment.* [arXiv 2311.08596](https://arxiv.org/abs/2311.08596). 10 LLMs × 7 tasks × 5 challenger utterances (AUS / IDTS / ABS / TEACH / PHD). Avg flip rate 46%, avg accuracy drop 17%. Defines our baseline two-turn protocol.

2. **sharma2023_understanding_sycophancy.pdf** — Sharma et al. (Anthropic). *Towards Understanding Sycophancy in Language Models.* [arXiv 2310.13548](https://arxiv.org/abs/2310.13548). ICLR 2024. Four sycophancy probes plus PM analysis showing RLHF preference data favours sycophantic responses. Companion code+data at `code/sycophancy-eval/`.

3. **hong2025_sycon_bench_multiturn.pdf** — Hong et al. *Measuring Sycophancy of Language Models in Multi-turn Dialogues* (SYCON-Bench). [arXiv 2505.23840](https://arxiv.org/abs/2505.23840). 17 models × 3 scenarios (debate / unethical / false presupposition); introduces ToF and NoF metrics. "Andrew prompt" mitigation = strongest hint of a "sweet spot" effect. Code at `code/SYCON-Bench/`.

4. **rabbani2025_fact_to_judgment_task_framing.pdf** — Rabbani, Bozdag, Hakkani-Tür (UIUC). *From Fact to Judgment: Investigating the Impact of Task Framing on LLM Conviction in Dialogue Systems.* [arXiv 2511.10871](https://arxiv.org/abs/2511.10871). Reframing factual queries as Conversational Judgment Task changes accuracy by 9.24% on average; direction (sycophantic vs over-critical) depends on the model. Code at `code/llm-conversational-judgment/`.

5. **fanous2025_syceval.pdf** — Fanous et al. (Stanford). *SycEval: Evaluating LLM Sycophancy.* [arXiv 2502.08177](https://arxiv.org/abs/2502.08177). AAAI/AIES 2025. Distinguishes preemptive vs in-context rebuttals and progressive vs regressive sycophancy on AMPS-Math and MedQuad. **Critical for us**: preemptive prompts are *more* sycophancy-inducing (61.75% vs 56.52%).

6. **kim2025_challenging_evaluator_rebuttal.pdf** — Kim & Khashabi (JHU). *Challenging the Evaluator: LLM Sycophancy Under User Rebuttal.* [arXiv 2509.16533](https://arxiv.org/abs/2509.16533). Findings of EMNLP 2025. Separates social framing (H1) from rebuttal-content reasoning (H2) from casual tone (H3); all three independently amplify capitulation.

## Priority 2 — important context

7. **wei2023_synthetic_data_reduces_sycophancy.pdf** — Wei, Huang, Lu, Zhou, Le (Google). *Simple synthetic data reduces sycophancy in large language models.* [arXiv 2308.03958](https://arxiv.org/abs/2308.03958).

8. **wang2023_can_chatgpt_defend_belief_debate.pdf** — Wang, Yue, Sun. *Can ChatGPT Defend its Belief in Truth? Evaluating LLM Reasoning via Debate.* [arXiv 2305.13160](https://arxiv.org/abs/2305.13160). The original "challenge with absurdly invalid arguments" debate setup.

9. **li2023_emotionprompt_emotional_stimuli.pdf** — Li et al. *Large Language Models Understand and Can be Enhanced by Emotional Stimuli.* [arXiv 2307.11760](https://arxiv.org/abs/2307.11760). Direct precedent for the *positive* arm of the hypothesis: ~10% boost from tonal stimuli.

10. **chang2026_causal_sycophancy_skepticism_causalt3.pdf** — Chang. *Diagnosing and Mitigating Sycophancy and Skepticism in LLM Causal Judgment* (CAUSALT3). [arXiv 2601.08258](https://arxiv.org/abs/2601.08258). Names the dual failure modes — *Skepticism Trap* and *Sycophancy Trap*.

11. **feng2026_good_arguments_people_pleasers.pdf** — Feng et al. *Good Arguments Against the People Pleasers: How Reasoning Mitigates (Yet Masks) LLM Sycophancy.* [arXiv 2603.16643](https://arxiv.org/abs/2603.16643). Reasoning may rationalise rather than resist sycophancy — must measure reasoning trace quality, not just length.

## Priority 3 — supporting benchmarks and angles

12. **petrov2025_brokenmath_sycophancy_theorem.pdf** — *BrokenMath: Sycophancy in Theorem Proving with LLMs.* [arXiv 2510.04721](https://arxiv.org/abs/2510.04721). 451 olympiad problems perturbed with false premises. HF: `INSAIT-Institute/BrokenMath`.

13. **zhang2025_sycophancy_under_pressure_pressure_tune.pdf** — Zhang et al. *Sycophancy under Pressure: Evaluating and Mitigating Sycophantic Bias via Adversarial Dialogues in Scientific QA.* [arXiv 2508.13743](https://arxiv.org/abs/2508.13743).

14. **lee2025_assertbench.pdf** — Lee & Chowdhary. *AssertBench: A Benchmark for Evaluating Self-Assertion in Large Language Models.* [arXiv 2506.11110](https://arxiv.org/abs/2506.11110). 41,836 FEVEROUS-derived claims with paired true/false user framings. Code at `code/assert-bench/`.

15. **au2026_beyond_social_pressure_epistemic_attack.pdf** — Au & Noronha. *Beyond Social Pressure: Benchmarking Epistemic Attack in Large Language Models* (PPT-Bench). [arXiv 2604.07749](https://arxiv.org/abs/2604.07749). Four pressure types (Epistemic Destabilization / Value Nullification / Authority Inversion / Identity Dissolution) at L0/L1/L2 escalation.

16. **barkett2025_reasoning_isnt_enough_truth_bias.pdf** — Barkett, Long, Thakur. *Reasoning Isn't Enough: Examining Truth-Bias and Sycophancy in LLMs.* [arXiv 2506.21561](https://arxiv.org/abs/2506.21561). ICML 2025.

17. **2025_monica_realtime_calibration.pdf** — *MONICA: Real-Time Monitoring and Calibration of Chain-of-Thought Sycophancy in Large Reasoning Models.* [arXiv 2511.06419](https://arxiv.org/abs/2511.06419). Per-step monitor that calibrates sycophantic drift during reasoning.

18. **dunlap2026_feedback_indices_rebuttals.pdf** — Dunlap, Parent, Widenhorn. *Feedback Indices to Evaluate LLM Responses to Rebuttals for Multiple Choice Type Questions.* [arXiv 2601.03285](https://arxiv.org/abs/2601.03285). Physics MCQ + fictitious rebuttals; newer/higher-reasoning models show less sycophancy.

19. **zhu2024_conformity_llms.pdf** — Zhu et al. *Conformity in Large Language Models.* [arXiv 2410.12428](https://arxiv.org/abs/2410.12428). Asch-style conformity benchmark + "Devil's Advocate" and "Question Distillation" mitigations.

20. **needham2025_llms_know_being_evaluated.pdf** — Needham et al. *Large Language Models Often Know When They Are Being Evaluated.* [arXiv 2505.23836](https://arxiv.org/abs/2505.23836). Frontier models have non-trivial eval-awareness — relevant for the "social facilitation under observation" framing.

## Reading workflow

Papers 1–6 are deep-read summarised in `literature_review.md`. For papers 7+, the abstract + first three pages are sufficient context for the experiment runner. Use the chunker if needed:

```bash
python .claude/skills/paper-finder/scripts/pdf_chunker.py papers/<filename>.pdf --pages-per-chunk 3
```

Chunked outputs for papers 1, 2, 3, 4, 5, 6, 7 (Wei), and 11 (Feng) are already in `papers/pages/`.
