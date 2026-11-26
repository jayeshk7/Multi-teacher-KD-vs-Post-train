# results of the final runs (without the LoRA adapters and the answer text)

The full answer text of every benchmark (about 400 MB) and the LoRA adapters are not in git. Numbers in the report and
the slides use each run's best checkpoint benchmarked on the RTX 6000 Ada (the run's own benchmark if it ran on the Ada,
else its Ada re-eval).

- `runs/<run>/` (31 runs: the 8 recipes and the 2 first stages KD math and KD code, 3 seeds each, plus one rank-48 KD math
  run `inst_kd_math_r48_s1`)
  - `log.json`: training log (losses, val checks, drift, chosen checkpoint, train hours)
  - `start.json`: which checkpoint the run starts from (sequential recipes start from a first-stage run)
  - `summary_best.json`, `summary_last.json`: benchmark scores of the best / last checkpoint on the GPU the run trained on
    (`gpu` field)
  - `summary_best_ada_reeval.json`, `val_best_ada_reeval.json`: the same best checkpoint benchmarked again on the Ada
    (runs that trained on the A6000)
- `reference/`: instruct on the Ada and on the A6000, the math and the code teacher
- `logs/`: training and re-eval logs
- `per_question.jsonl`: one row per result folder; GSM8K and MATH-500 as strings of 1 (right) / 0 (wrong) in dataset order,
  held-out code as the fraction of tests passed per question, MBPP+ / HumanEval+ as task -> 1 / 0
- error analysis: `../analysis/error_analysis_all.json` (GSM8K / MATH-500 answer style and error kinds per recipe, from
  `error_analysis.py`) and `errors.jsonl.gz` / `errors.md` here (one category per wrong answer on every benchmark, failing
  code run again for its exception, from `../analysis/errors.py`), and `../analysis/stray_python_rescore.json` (held-out with the
  stray "python" line fixed, robustness check)
