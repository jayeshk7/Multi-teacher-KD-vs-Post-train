# info.md: handoff for the next Claude session on this repo

Written 24 Oct 2026 by the Claude session that wrote the code and ran the sequential experiments. Read this first, then
`README.md` (setup and results), `logs_review.md` (findings that hold over 3 seeds) and `LOG.md` (every run in order,
newest at the bottom, with the reasons behind each decision).

## 1. Project

- CS725 (FML) team project: multi-teacher knowledge distillation, joint vs sequential, later vs post-training.
- Teachers: Qwen2.5-Math-7B-Instruct and Qwen2.5-Coder-7B-Instruct. Student: Qwen2.5-0.5B-Instruct. All three share
  one tokenizer.
- Training data (`0_data_gen/gen_data.py`): 7473 GSM8K train prompts (math) and 7473 KodCode-Light-RL-10K prompts
  (code), answered by the domain's teacher (greedy, up to 1600 tokens). Answers that hit the limit are dropped for
  every setting: 7441 math and 7469 code rows kept. The last 200 rows of each domain are the val set, never trained on.
- Ground truth answers (the "gold" field in the data and code): GSM8K's solution without the `<<...>>` calculator notes
  plus "The answer is \boxed{N}."; KodCode's solution as a bare python code block.
- Loss per batch: ALPHA * KL(teacher || student) per token on the teacher's answer + (1 - ALPHA) * CE per token on the
  ground truth answer. ALPHA 0.5 = main setting, 0.0 = ground truth SFT, 1.0 = KD only. TEMPERATURE = 1.
- Benchmarks: GSM8K test (1319), MBPP / MBPP+ (378), HumanEval / HumanEval+ (164) through evalplus 0.3.1. Greedy,
  pass@1, full test sets. GSM8K is right if the last `\boxed{}` (or the last number if nothing is boxed) equals the
  ground truth number. Base student: GSM8K 651 (49.4%), MBPP 183 (48.4%), MBPP+ 156 (41.3%), HE 62 (37.8%), HE+ 56
  (34.1%).
- Midsem checkpoint presentation: Monday 26 Oct 2026, evening.

## 2. Status

Done:
- data generation, baselines (README section 1)
- sequential KD (README section 2): 6 settings (math -> code and code -> math, ALPHA 0.5 / 0.0 / 1.0) x seeds 0, 1, 2
  = 18 runs, all valid. README holds the 3-seed means, `LOG.md` every seed (questions right, val losses), and
  `2_sequential_kd/runs/` every run's result files and logs.

Not done (README `### TODO`):
- joint KD (README section 3): 3 settings x 3 seeds = 9 runs. The scripts are ready (section 4 below) but have never run
  on a GPU, only in a CPU test with tiny random models (LOG entry "24 Oct 2026 ~01:15 UTC").
- teacher SFT (CE on the teacher's answers, `CE_TARGET = "response"`): not implemented.
- temperature ablation (TEMPERATURE = 2.0 vs 1.0): planned after the 26 Oct checkpoint.
- failure analysis, post-training (`4_post_train/` is empty).
- filtered teacher answers ablation: dropped by the team, its TODO line is still in the README.

Known README error: the loss table (`README.md:41`) says `(ALPHA * KL) + (ALPHA * CE)`. The code
(`3_joint_kd/kd_train.py:108`) uses `ALPHA * KL + (1 - ALPHA) * CE`.

## 3. Repo layout

| path | what |
|------|------|
| `0_data_gen/gen_data.py` | teacher answers for the training prompts, uploaded to the hub dataset |
| `1_baseline_scripts/` | `bench_student.py`, `bench_math_teacher.py`, `bench_code_teacher.py` + `bench_utils.py` |
| `2_sequential_kd/` | the 6 sequential scripts, `kd_utils.py`, `bench_utils.py`, `results_table.py` |
| `2_sequential_kd/runs/` | copy of the server run folder: 18 run folders, 18 training logs, `chain.sh`, `verify_delete.py`, chain logs |
| `3_joint_kd/` | `kd_train.py` (ALPHA 0.5), `abl_sft_joint.py` (0.0), `abl_kl_only_joint.py` (1.0), `kd_utils.py`, `bench_utils.py` |
| `LOG.md` | every run and check, in order |
| `logs_review.md` | the findings and the claims that did not survive the seeds |

- `bench_utils.py` and `kd_utils.py` are copies of one file each; keep them identical (`md5sum */bench_utils.py
  */kd_utils.py`).
- The joint ablation scripts differ from `kd_train.py` only in the header, `ALPHA` and `OUT_DIR`; the sequential
  ablations likewise from their `seq_*` script.
- Each run folder in `runs/` has `results_stage1_<domain>/` and `results_stage2_<domain>/` with `summary.json`
  (accuracies and benchmark sizes), `gsm8k.jsonl` (fields `right`, `gold`, `answer`), `mbpp.jsonl`, `humaneval.jsonl`
  and evalplus's `*_eval_results.json`. Seed 0 folders and logs have no suffix (`seq_math_code`, `seq_math_code.log`);
  seeds 1 and 2 have `_seed1` / `_seed2` (logs are named after the script: `seq_math_then_code_seed1.log`).
- `2_sequential_kd/results_table.py` (run from `2_sequential_kd/`) prints the README rows and every seed from `runs/`.

## 4. Running the joint experiments

What `kd_train.py` does:
- one training run on both domains: batches of 32 prompts, each from one domain, so each batch needs only its own
  domain's teacher. Per epoch 226 math + 227 code batches, shuffled together. 2 epochs = 906 optimizer steps, the same
  data and step count as one sequential run (452 math + 454 code steps).
- one AdamW optimizer (LR 1e-5), 3% linear warmup then cosine decay to 0 over all 906 steps, gradient clipping 1.0,
  micro batches of 4 (`MICRO`).
- student in fp32 with Adam on `cuda:0`, both teachers in bf16 on `cuda:1` (30.5 GB). The same memory layout as the
  sequential runs, which peaked at 43-46 GB on the student GPU (48 GB A6000).
- val on both domains at step 0 and after each epoch (steps 453 and 906), printed as `math val at step N: kl X  ce Y`.
- at the end: save the bf16 checkpoint to `OUT_DIR/joint` (with `log.json`), load it again with `from_pretrained`,
  benchmark, write `OUT_DIR/results_joint/`, upload `OUT_DIR` to the hub.
- `SEED` comes from the environment and sets the data order only. `OUT_DIR` is `joint_kd`, `joint_kd_seed1`,
  `joint_kd_seed2` (`abl_sft_joint*`, `abl_kl_only_joint*` for the ablations).

How to run one (from a folder holding the script, `kd_utils.py` and `bench_utils.py`; `data/` is downloaded from the
hub on start):

```
SEED=1 CUDA_VISIBLE_DEVICES=0,1 nohup python -u kd_train.py > kd_train_seed1.log 2>&1 < /dev/null &
```

Checks for the first run:
- the step-0 val must equal the base student's: math KL 0.1706, CE 0.8387; code KL 0.2212, CE 0.3674 (0.2213 seen
  once). Anything else means the data or the model loading changed.
- the first logged steps show whether memory holds (no OOM traceback). Warnings "memory allocation failed with OOM"
  without a traceback are harmless (PyTorch frees its cache and retries).
- time per run is not measured. A sequential run (906 steps + 2 benchmark rounds) took 1 h 40 min - 1 h 55 min on these
  GPUs; a joint run has 1 benchmark round, so expect about 1.5 h, and the 9 runs about 7-8 h on 2 GPU pairs.

Reusing `2_sequential_kd/runs/chain.sh` (runs scripts one after another on one GPU pair, waits for the GPUs to be free)
and `verify_delete.py` (deletes a run's local checkpoint once every file is on the hub with the same sha256 or size)
needs three changes:
1. `chain.sh` has `cd /tmp/jayk_runs/2_sequential_kd` hardcoded, and its first argument is the PID of a run to wait for
   (line 7-10 then checks that run's `results_stage2_*`).
2. `chain.sh` names a run's folder `<OUT_DIR>_seed<N>`, also for seed 0, but seed 0 writes `<OUT_DIR>` without a suffix.
   Seeds 1 and 2 were the only ones chained so far, so this never mattered; for joint seed 0 it would look in the wrong
   folder (the check fails safe: nothing is deleted).
3. `verify_delete.py` looks for `<out_dir>/stage*`; the joint checkpoint is `<out_dir>/joint`.

Deleting the checkpoint also deletes `joint/log.json` locally (it stays on the hub); the val lines are in the stdout log
too.

## 5. The department server

- Mangesh's account; 4 GPUs of 48 GB: 0, 1, 2 are RTX A6000, 3 is an RTX 6000 Ada. The runs used the pairs 0 + 1 and
  2 + 3, student on the first GPU of the pair. Check `nvidia-smi` before starting: other users share the server.
- `~/jayk_fml/env.sh` sets `HF_HOME=/tmp/jayk_hf` (the models are cached there, ~30 GB) and
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and activates `~/FML/.venv` (torch 2.13.0+cu126, transformers 5.20.2, evalplus
  0.3.1). Never delete `~/FML/.venv`, and don't install packages there without asking Mangesh.
- Runs were in `/tmp/jayk_runs/` because the home quota is 27.6 GB. `/tmp` is wiped on a reboot: copy every finished
  run's results and logs into the repo (like `2_sequential_kd/runs/`) and check the copy with md5sum.
- Hugging Face: the data and the checkpoints are in public repos of the account `jajamken`: dataset
  `jajamken/FML_project` (`data/math.jsonl`, `data/code.jsonl`, baseline results) and model repo `jajamken/FML_project`
  (one folder per run). The scripts download from `HUB_DATA` and upload to `HUB_REPO`, both set to that name.
  - Downloading needs no login, so `HUB_DATA` works as it is. The dataset will not change.
  - Uploading needs write access to jajamken's repo, and the server's login in `/tmp/jayk_hf` (jajamken's) is being
    logged out. With `HUB_REPO` unchanged, a run writes its checkpoint and benchmark results locally, then crashes at
    the upload (nonzero exit, so `chain.sh` keeps the checkpoint).
  - So choose before the joint runs: `HUB_REPO = ""` (everything stays local; jajamken uploads the run folders later by
    logging in on the server, so keep the checkpoints in `/tmp` until then), or `HUB_REPO` = a repo of your own account
    (log in with your own token in a separate `HF_HOME`). `verify_delete.py` checks against `jajamken/FML_project`, so
    change that name too, or don't run it when there is no upload.
  - Never print, paste or commit a token.

## 6. Conventions the team has followed

- Log every run in `LOG.md` as it happens: start (UTC time, server, GPUs, settings), end (exit code, tracebacks, hub
  check) and results (questions right per benchmark, val losses). Mark runs that turn out invalid with "INVALID" in the
  heading and keep them.
- Report results as questions right and %, with the difference to the base student in percentage points (pp). One
  question is 0.08 pp on GSM8K, 0.26 pp on MBPP / MBPP+, 0.61 pp on HumanEval / HE+.
- README tables are 3-seed means as `% (pp)`, as in section 2; the joint table in section 3 should follow that format.
- Only present claims that hold in all 3 seeds. The seed gaps reach 53 GSM8K questions, 13 MBPP and 7-8 HumanEval
  questions, and the val losses do not show them.
- Say "ground truth" in the README and the report; the code and older LOG entries say "gold".
- The README `### TODO` lists only project work (experiments, ablations, analyses), not run steps or setup chores.
- Code style: short lowercase comments, settings as globals at the top (no CLI arguments), run-once scripts.
- Benchmark only a saved checkpoint loaded with `from_pretrained`: `.to(torch.bfloat16)` on a live model also rounds
  RoPE's `inv_freq` buffer and changes the answers (LOG entry "23 Oct 2026 20:07 UTC").
- Fixed budget, 2 epochs, no early stopping; only the final model is benchmarked. Val losses are for monitoring.

## 7. What the joint results should be compared against

Joint and sequential see the same data and the same 906 steps, so compare at the same ALPHA (3-seed means, README
section 2):

| sequential, after both stages | GSM8K        | MBPP        | MBPP+       | HE          | HE+         |
|-------------------------------|--------------|-------------|-------------|-------------|-------------|
| math -> code, ALPHA 0.5       | 43.2 (-6.1)  | 49.9 (+1.5) | 42.5 (+1.2) | 35.4 (-2.4) | 32.9 (-1.2) |
| code -> math, ALPHA 0.5       | 40.8 (-8.5)  | 47.1 (-1.3) | 41.0 (-0.3) | 30.7 (-7.1) | 27.2 (-6.9) |
| math -> code, ALPHA 0.0       | 35.4 (-13.9) | 48.9 (+0.5) | 42.8 (+1.5) | 30.9 (-6.9) | 27.6 (-6.5) |
| code -> math, ALPHA 0.0       | 36.1 (-13.3) | 47.2 (-1.2) | 41.4 (+0.2) | 30.7 (-7.1) | 28.7 (-5.5) |
| math -> code, ALPHA 1.0       | 48.3 (-1.1)  | 49.1 (+0.7) | 42.0 (+0.7) | 34.1 (-3.7) | 30.1 (-4.1) |
| code -> math, ALPHA 1.0       | 50.7 (+1.3)  | 48.1 (-0.4) | 42.2 (+0.9) | 33.1 (-4.7) | 29.5 (-4.7) |

Questions the joint runs answer (see `logs_review.md` for the evidence behind each):
1. Does mixing the domains avoid the one-sided forgetting? In sequential training a math stage with ground truth CE
   drops HumanEval to 49-52 (base 62) in all 12 runs, while a code stage costs little GSM8K (finding 5).
2. Does joint training land between the two orders, or above the better one (math -> code at ALPHA 0.5)?
3. Do the format effects stay? Count in each `gsm8k.jsonl`: answers containing "The answer is \boxed{" (ground truth
   style, findings 1 and 3), answers without `\boxed{` (finding 8), answers containing "```python" (finding 9: only
   SFT on both domains made the student answer math with code, 3 / 6 / 58 answers by seed).
4. Does the ground truth CE still overfit in epoch 2 (math val CE rises by about 0.02 in every sequential run with
   ground truth CE, finding 7)?

To add the joint runs to `results_table.py`: the summaries are `<OUT_DIR>/results_joint/summary.json`, and the final
val is the last math and code val line of the log (3 val points per domain instead of 6).
