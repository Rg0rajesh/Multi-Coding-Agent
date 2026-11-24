# Multi-Coding Agent: Adaptive, Cost-Aware Multi-Agent Code Generation and Self-Debugging on Open-Weight Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Reproducibility: 100% Verified](https://img.shields.io/badge/Reproducibility-100%25%20Verified-brightgreen.svg)](#one-click-artifact-verification)
[![Paper: IEEE Format](https://img.shields.io/badge/Paper-IEEE%20Format-red.svg)](#citation)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rg0rajesh/Multi-Coding-Agent/blob/main/Multi_Coding_Agent_Replication.ipynb)

Official research artifact repository for the paper:
**"Adaptive, Cost-Aware Multi-Agent Code Generation and Self-Debugging on Open-Weight Language Models"**.

This repository contains the complete end-to-end multi-agent pipeline, model serving interfaces, benchmark datasets, execution sandbox, and water-tight experimental execution logs across **HumanEval (164 tasks)** and **MBPP (974 tasks)**, as well as the **Zero-Test Benchmark (100 tasks)** and **Diagnostic Verifier Reliability Suite (100 paired samples)**.

---

## Key Highlights & Contributions

Prevailing multi-agent code generation architectures incur severe token bloat (often 10,000 to 30,000 tokens per task), assume the universal existence of pre-existing unit test suites, and suffer from high false-rejection rates when compact language models are deployed as self-evaluators.

**Multi-Coding Agent** resolves these challenges via three chained, cost-aware modules:
1. **Module 1 (M1) — Adaptive Loop Controller**: Employs sub-millisecond static AST pre-repair for trivial syntax errors, canonical error normalization, cycle/oscillation detection, and dynamic budget extension ($K(t)$) to eliminate runaway multi-turn repair loops.
2. **Module 2 (M2) — Spec-Driven Autonomous Test Oracle**: Automatically mines docstring assertions and synthesizes contract boundary invariant tests when external test suites are absent (the Zero-Test condition), achieving a 100% compilation success rate.
3. **Module 3 (M3) — Tiered Hybrid Execution Verifier**: Combines deterministic AST syntax checks with isolated sandbox test execution and monotonic candidate rollback, suppressing judge False Negative Rates (FNR) from 70.0% down to 2.0% and lifting Cohen's $\kappa$ to 0.70–0.80.

---

## Main Experimental Results

Evaluated across six open-weight language models spanning 0.5B to 6.7B parameters on a single consumer GPU (Tesla T4, 4-bit Q4_K_M precision):

### Table I: Main Benchmark Results (HumanEval & MBPP)
| Model | Params | HumanEval Pass@1 (%) | HumanEval Solved | HumanEval Avg Tokens | MBPP Pass@1 (%) | MBPP Solved | MBPP Avg Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-Coder 0.5B** | 0.5B | 71.95% | 118 / 164 | 2,727 | 50.10% | 488 / 974 | 4,656 |
| **Qwen2.5-Coder 1.5B** | 1.5B | 80.49% | 132 / 164 | 1,547 | 69.71% | 679 / 974 | 2,772 |
| **DeepSeek-Coder 1.3B** | 1.3B | 80.49% | 132 / 164 | 1,848 | 70.53% | 687 / 974 | 3,590 |
| **Llama 3.2 3B** | 3.0B | 79.27% | 130 / 164 | 1,974 | 70.33% | 685 / 974 | 2,348 |
| **Qwen2.5-Coder 3B** | 3.0B | **92.68%** | 152 / 164 | **1,096** | 77.52% | 755 / 974 | **1,884** |
| **DeepSeek-Coder 6.7B** | 6.7B | 84.76% | 139 / 164 | 1,526 | **78.03%** | 760 / 974 | 2,285 |

*Note: MBPP results consist of strictly clean, single-run execution logs across all 974 benchmark problems with zero duplicate reruns.*

---

## Repository Structure

```text
Multi-Coding Agent/
├── README.md                      <- Comprehensive repository overview and guide
├── LICENSE                        <- MIT Open Source License
├── requirements.txt               <- Python package dependencies
├── setup.py                       <- Standard package installation setup
├── verify_all_results.py          <- One-click zero-dependency validation script
├── reproduce_paper_tables.py      <- Paper table generator (plain text & LaTeX)
│
├── my_pipeline/                   <- Core pipeline modules
│   ├── pipeline.py                <- Main multi-agent coordinator (M1 + M2 + M3)
│   ├── module1_adaptive_threshold.py <- M1: Static repair, error normalization, loop stop
│   ├── module2_test_oracle.py     <- M2: Generative spec-driven test synthesis
│   ├── module3_hybrid_verification.py <- M3: Tiered AST + sandbox + judge verifier
│   ├── debug_specialist.py        <- Specialist prompt templates and error injection
│   ├── sandbox_exec.py            <- Secure isolated process code execution sandbox
│   ├── run_experiment.py          <- Main experimental CLI evaluation driver
│   ├── run_small_models.py        <- Colab / Ollama runner for resource-constrained setups
│   ├── analyze_module2_zerotest.py <- Zero-test evaluation analyzer
│   ├── evaluate_module3_judge_reliability.py <- Paired reliability grader
│   └── generate_reports.py        <- Diagnostic report builder
│
├── shared/                        <- Shared infrastructural components
│   ├── dataset_loader.py          <- Unified loader for HumanEval and MBPP
│   ├── models.py                  <- LLM serving interface (Ollama, vLLM, HF)
│   ├── logger.py                  <- Atomic JSONL execution logger
│   └── trace_logger.py            <- Turn-by-turn prompt & candidate trace logger
│
├── analysis/                      <- Rigorous statistical analysis tools
│   ├── paired_comparison.py       <- McNemar paired tests & Wilson score intervals
│   ├── significance_test.py       <- Cohen's kappa and contingency table tests
│   └── summarize.py               <- Metric aggregation script
│
├── data/                          <- Benchmark problem sets
│   ├── HumanEval.jsonl            <- 164 HumanEval problem definitions
│   ├── mbpp.jsonl                 <- 974 MBPP problem definitions
│   ├── selected_100_task_ids.json <- 100 Zero-Test task IDs
│   ├── HumanEval_ZeroTest_100.jsonl <- 100 Zero-Test tasks (unit tests stripped)
│   └── module3_paired_ground_truth.jsonl <- 100 paired evaluation samples
│
└── results/                       <- Raw experimental execution traces (.jsonl)
    ├── mbpp/                      <- Clean, single-run full-pipeline MBPP logs (974 runs each)
    │   ├── mypipeline_mbpp_module1+module2+module3_qwen2.5-coder-0.5b.jsonl
    │   ├── mypipeline_mbpp_module1+module2+module3_qwen2.5-coder-1.5b.jsonl
    │   ├── mypipeline_mbpp_module1+module2+module3_deepseek-coder-1.3b.jsonl
    │   ├── mypipeline_mbpp_module1+module2+module3_llama3.2-3b.jsonl
    │   ├── mypipeline_mbpp_module1+module2+module3_qwen2.5-coder-3b.jsonl
    │   └── mypipeline_mbpp_module1+module2+module3_deepseek-coder-6.7b.jsonl
    ├── humaneval/                 <- 24 execution logs covering all 4 ablation stages
    ├── zerotest/                  <- 12 logs for M1 and M1+M2 zero-test evaluations
    ├── judge_reliability/         <- Ground-truth pairing & judge cache files
    └── summaries/                 <- Summary CSVs for ablation and significance testing
```

---

## One-Click Artifact Verification

To enable immediate, independent validation by peer reviewers, this repository includes an automated verification script that re-computes all paper metrics directly from the raw execution traces:

```bash
python verify_all_results.py
```

### Verification Output:
```text
================================================================================
1. MBPP BENCHMARK VALIDATION (974 Tasks, Single-Run Deduplication Check)
================================================================================
Model                   Tasks   Dups  Solved    Pass@1    Paper%   AvgTok Status
--------------------------------------------------------------------------------
Qwen2.5-Coder 0.5B        974      0     488    50.10%    50.10%     4656  [PASS]
Qwen2.5-Coder 1.5B        974      0     679    69.71%    69.71%     2772  [PASS]
DeepSeek-Coder 1.3B       974      0     687    70.53%    70.53%     3590  [PASS]
Llama 3.2 3B              974      0     685    70.33%    70.33%     2348  [PASS]
Qwen2.5-Coder 3B          974      0     755    77.52%    77.52%     1884  [PASS]
DeepSeek-Coder 6.7B       974      0     760    78.03%    78.03%     2285  [PASS]

[OK] All 6 MBPP models contain exactly 974 single runs with 0 duplicates.
[OK] Exact 100% agreement with Table I, Table III, Table IV, and Table IX in paper.

>>> ALL VALIDATION CHECKS PASSED PERFECTLY (100% REPRODUCIBLE) <<<
```

---

## Quickstart & Installation

### 1. Environment Setup
```bash
git clone https://github.com/anonymous/Multi-Coding-Agent.git
cd Multi-Coding-Agent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Model Serving
Ensure local open-weight models are served via [Ollama](https://ollama.com/) or vLLM:
```bash
ollama run qwen2.5-coder:3b
ollama run deepseek-coder:6.7b
```

### 3. Running Experiments
To evaluate the full pipeline (+M1+M2+M3) on HumanEval:
```bash
python -m my_pipeline.run_experiment \
  --dataset_name humaneval \
  --model qwen2.5-coder:3b \
  --config +module1+module2+module3
```

To evaluate the full pipeline on MBPP:
```bash
python -m my_pipeline.run_experiment \
  --dataset_name mbpp \
  --model qwen2.5-coder:3b \
  --config +module1+module2+module3
```

To evaluate under Zero-Test conditions (stripping test cases):
```bash
python -m my_pipeline.run_experiment \
  --dataset_name humaneval \
  --model deepseek-coder:1.3b \
  --config +module1+module2 \
  --strip_tests \
  --task_list data/selected_100_task_ids.json
```

---

## Citation

```bibtex
@article{multicodingagent2026,
  title={Adaptive, Cost-Aware Multi-Agent Code Generation and Self-Debugging on Open-Weight Language Models},
  author={Anonymous Authors},
  journal={Under Review},
  year={2026}
}
```

---

## License
This repository is licensed under the [MIT License](LICENSE).

