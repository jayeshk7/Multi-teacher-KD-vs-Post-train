# Run log

Every run, with its settings and results, in the order they ran.

## 23 Oct 2026 - gen_data.py smoke test (molab, RTX Pro 6000 96 GB)

Settings: `0_data_gen/gen_data.py` with `TOTAL_PROMPTS = 8` (only change), `MAX_OUTPUT_LENGTH = 1600`,
`BATCH_SIZE = 128`, `HF_HUB_DATA = "jajamken/FML_project"`. transformers 5.14.1.

| domain | kept | cut | answer tokens (median / p95 / max) | gold tokens (median / p95 / max) | teacher accuracy |
|--------|------|-----|------------------------------------|----------------------------------|------------------|
| math   | 8    | 0   | 274 / 354 / 354                    | 71 / 119 / 119                   | 8/8 = 1.000      |
| code   | 8    | 0   | 89 / 234 / 234                     | 120 / 176 / 176                  | -                |

Time: 1m31s in total (each teacher download ~30 s).

Checked by decoding the first row of each file: prompts have the right system prompt and chat format; the math answer
is step-by-step and ends in `\boxed{72}` + `<|im_end|>`; the math gold has the calculator notes removed and ends in
`\boxed{72}`; the code answer uses the declared function name. The code teacher opens its block with a plain ```` ``` ````
while the gold uses ```` ```python ````.

Verdict: pipeline works end to end. Numbers on 8 prompts are not results.

## 23 Oct 2026 - gen_data.py full run (molab, RTX Pro 6000 96 GB)

Settings: `0_data_gen/gen_data.py` unchanged (`TOTAL_PROMPTS = 7473`, `MAX_OUTPUT_LENGTH = 1600`, `BATCH_SIZE = 128`,
greedy, `repetition_penalty = 1.0`), started with `nohup ... > gen_data.log`.

| domain | kept | cut         | answer tokens (median / p95 / max) | gold tokens (median / p95 / max) | teacher accuracy |
|--------|------|-------------|------------------------------------|----------------------------------|------------------|
| math   | 7441 | 32 (0.43%)  | 288 / 467 / 1264                   | 87 / 178 / 347                   | 0.956            |
| code   | 7469 | 4 (0.05%)   | 162 / 400 / 1089                   | 127 / 279 / 909                  | -                |

Math: accuracy 0.956 on gsm8k train (cut answers counted wrong), vs 95.2 reported on test; the teacher's SFT data includes
gsm8k train, so train can be higher. Cut answers are in `{math,code}_cut.jsonl` for the failure analysis (code cut indices
672, 2695, 3460, 7056). The longest kept answer is 1264 tokens, so 1600 is enough for both domains.

Time: started 05:52 UTC, math done 07:28 (~96 min, ~1.6 min per 128-prompt batch), code done and uploaded 09:11
(~102 min), 3 h 19 min in total. The molab terminal closed during the run; the process survived because of nohup.

Upload checked from the server as jajamken (commit 09:11:14 UTC): hub root has math.jsonl (19.5 MB, same size as on
molab), math_cut.jsonl, code.jsonl (24.8 MB), code_cut.jsonl; kept + cut = 7473 per domain; every response and every
gold ends in `<|im_end|>` (151645).

## 23 Oct 2026 - bench_student.py quick test (department server, RTX A6000 48 GB)

Settings: `1_baseline_scripts/bench_student.py` with `MAX_QUESTIONS = 16` (only change), `MAX_OUTPUT_LENGTH = 1600`,
`BATCH = 128`, `HUB_RESULTS = "jajamken/FML_project"`. Mangesh's venv `~/FML/.venv` (python 3.12, torch 2.13 cu126,
transformers 5.20.2, evalplus 0.3.1), `HF_HOME=/tmp/jayk_hf` (his home quota is too small for the 7B models).

| benchmark      | base          | plus          |
|----------------|---------------|---------------|
| gsm8k (16)     | 3/16 = 0.188  | -             |
| mbpp (16)      | 10/16 = 0.625 | 8/16 = 0.500  |
| humaneval (16) | 12/16 = 0.750 | 12/16 = 0.750 |

Checks: evalplus's own pass@1 over all tasks (0.026 x 378 = 10, 0.021 x 378 = 8, 0.073 x 164 = 12) matches our counts on
the 16 asked tasks (skipped tasks get empty code). All 16 gsm8k answers end in `\boxed{...}` and the extracted number is the
model's own, so the 13 misses are real mistakes, not extraction bugs; none hit the length limit. evalplus expected outputs
took 52 s (mbpp+) and 15 s (humaneval+), now cached in `~/.cache/evalplus`. Upload ok: `results_student/` (6 files) is in the
hub repo as jajamken.

Verdict: benchmark pipeline and upload work. gsm8k 3/16 is too few questions to compare with Qwen's 49.6.

## 23 Oct 2026 - baselines, full run (department server, 3x RTX A6000 48 GB, one model per GPU)

Settings: the three `1_baseline_scripts/bench_*.py` unchanged (`MAX_QUESTIONS = 2000` = every question,
`MAX_OUTPUT_LENGTH = 1600`, `BATCH = 128`, greedy, `repetition_penalty = 1.0`), same env as the quick test, started
07:17 server time with `CUDA_VISIBLE_DEVICES` 0 (math teacher), 1 (code teacher), 2 (student).

### student, Qwen2.5-0.5B-Instruct (finished 07:31, ~14.5 min)

| benchmark | ours                 | Qwen reported |
|-----------|----------------------|---------------|
| GSM8K     | 651/1319 = 49.4      | 49.6          |
| MBPP      | 183/378 = 48.4       | 49.6          |
| MBPP+     | 156/378 = 41.3       | -             |
| HumanEval | 62/164 = 37.8        | 35.4          |
| HE+       | 56/164 = 34.1        | -             |

All three reported numbers are within 2.5 points of ours (Qwen's MBPP is likely the original MBPP test set, not
evalplus's 378, so that one is a loose check). Answer lengths in tokens (median / p95 / max): GSM8K 294 / 506 / 1600, MBPP 122 / 242 /
1600, HumanEval 225 / 479 / 1600. Hit the 1600 limit: GSM8K 16 (all wrong), MBPP 4, HumanEval 2. Upload checked: hub
`results_student/summary.json` has n_gsm8k 1319.

### code teacher, Qwen2.5-Coder-7B-Instruct (finished 07:41, ~24 min incl. ~5 min model download)

| benchmark | ours                 | Qwen reported |
|-----------|----------------------|---------------|
| GSM8K     | 1167/1319 = 88.5     | -             |
| MBPP      | 310/378 = 82.0       | 83.5          |
| MBPP+     | 267/378 = 70.6       | 71.7          |
| HumanEval | 149/164 = 90.9       | 88.4          |
| HE+       | 140/164 = 85.4       | 84.1          |

All four reported numbers are within 2.5 points of ours. Answer lengths in tokens (median / p95 / max): GSM8K 293 / 487
/ 1237, MBPP 126 / 326 / 457, HumanEval 311 / 490 / 747; no answer hit the 1600 limit. Upload checked (6 files).

### math teacher, Qwen2.5-Math-7B-Instruct (finished 08:22, ~66 min incl. ~5 min model download)

| benchmark | ours                 | Qwen reported |
|-----------|----------------------|---------------|
| GSM8K     | 1260/1319 = 95.5     | 95.2          |
| MBPP      | 177/378 = 46.8       | -             |
| MBPP+     | 152/378 = 40.2       | -             |
| HumanEval | 101/164 = 61.6       | -             |
| HE+       | 90/164 = 54.9        | -             |

GSM8K matches the report. Answer lengths in tokens (median / p95 / max): GSM8K 291 / 470 / 1600, MBPP 215 / 472 / 1600,
HumanEval 359 / 772 / 1600. Hit the 1600 limit: GSM8K 7 (1 still right, it boxed the answer before looping), MBPP 7,
HumanEval 5. The GSM8K ones are all repetition loops (`re:]`, `]`, `}` repeated to the limit), the cost of greedy with
`repetition_penalty = 1.0`. Upload checked (6 files).

Why its MBPP is below the student's (46.8 vs 48.4): 113 of its 201 MBPP failures define the function under a different
name than the tests call (it rewrites MBPP's mixed-case names, e.g. `differ_At_One_Bit_Pos` -> `differ_at_one_bit_position`,
`find_Rotations` -> `find_rotations`; the Mbpp/6 code is otherwise correct). Wrong or missing function name among failures:

| model        | MBPP fails | wrong name | HumanEval fails | wrong name |
|--------------|------------|------------|-----------------|------------|
| math teacher | 201        | 113        | 63              | 10         |
| code teacher | 68         | 0          | 15              | 1          |
| student      | 195        | 25         | 102             | 1          |

This is a real failure under evalplus's rules (the prompt's assert shows the name), so the score stands; it is a
note for the failure analysis. Per task on MBPP base, student and math teacher are close to symmetric: 64 tasks only the
student passes, 58 only the math teacher passes, 119 both.

Memory note: the logs show `CUDACachingAllocator ... memory allocation failed with OOM` warnings, but the runs go on.
PyTorch's cache fills up with freed KV-cache blocks, the failed request makes it release the cache and retry, and the retry
succeeds. The math teacher's requests are 1.47-1.56 GB = 128 x 28 heads x ~1600-1700 positions x 128 dims x 2 bytes, i.e.
the KV cache expanded from 4 to 28 heads for attention at one layer. A real OOM would end the process with a traceback.
A warning fires only when a batch grows longer than every earlier batch (request sizes rise and never repeat: 52 for the
math teacher, 9 code teacher, 1 student). The math teacher's batches reach ~1700 positions because `generate` keeps the
whole batch running until its last answer stops, and its looping answers run to the 1600 limit.

## 23 Oct 2026 - shared code refactor, checked against the old scripts (laptop cpu + department server)

Change: the benchmark code (prompts, `MAX_OUTPUT_LENGTH`, `GEN_BATCH`, generate, scoring) moved to `bench_utils.py`, the
same file in `1_baseline_scripts/`, `2_sequential_kd/` and `3_joint_kd/`. The kd losses and val (`collate`, `logits_at`,
`losses`, `val_losses`, `val_all`, `VOCAB_SIZE`, `PAD`) moved to `kd_utils.py`, the same file in `2_sequential_kd/` and
`3_joint_kd/`. The scripts keep their settings and their training / run loops. Check the copies match with
`md5sum */bench_utils.py */kd_utils.py`.

Check 1, laptop cpu: old (backup) and new versions of all 6 scripts run with tiny random Qwen2 models, fake gsm8k / evalplus,
the same small settings on both sides, uploads blocked. Every output file is byte-identical (model weights, `log.json`,
answers, eval results, `summary.json`; 83 files). Two deliberate bugs in the new code were caught (answer mask shifted by
one: 16 files differ; answers given to the wrong question: 12 files differ).

Check 2, server: new `bench_student.py` (copy in `~/jayk_fml/refactor_check/`, `OUT_DIR = "results_student_refactor"`,
no upload) on GPU 2 (RTX A6000, same GPU as the baseline run), 13:50-14:05 UTC (~14 min). Same as `results_student/`
question by question: gsm8k 1319/1319 identical answers (651 right in both), mbpp 378/378 identical code and identical
base / plus status, humaneval 164/164 the same. So the baselines in section 1 stand for the new code, and reruns on the
same GPU type with the same batches give identical answers.

## 23 Oct 2026 - kd temperature added, checked on laptop cpu

Change: `kd_utils.losses(..., temperature=1.0)`. At T != 1 both teacher and student logits are divided by T before
log_softmax and the kl is multiplied by T ^ 2 (hinton et al. 2015); ce stays at T = 1; val kl is always at T = 1.
Each kd script has `TEMPERATURE = 1.0` and passes it to `losses()`. Main runs stay at T = 1, T = 2 is a later ablation.
Literature: minitron (arxiv 2407.14679, appendix A.2) swept 0.1 / 0.5 / 1.0 / 3.0 and found 1.0 best; minillm uses no
logit temperature; gkd uses teacher temperature 0.1 on gsm8k and 1.0 elsewhere; distilbert's code uses 2.0.

Check (same cpu harness as the refactor check, tiny random models, fakes, uploads blocked): old scripts vs current
scripts at T = 1 give byte-identical outputs for both seq scripts, kd_train.py and bench_student.py (the only
difference: the seq scripts no longer save tokenizer files, removed on purpose). T = 2 run of seq_code_then_math.py:
val at step 0 identical to T = 1 (val is at T = 1), train ce at step 1 identical (ce has no T), train kl and the final
weights differ, so T only changes the kl term.

## 23 Oct 2026 - review of 2_sequential_kd/ and sync of kd_train.py, checked on laptop cpu

Change: comment fixes in `seq_code_then_math.py` (scheduler comment, autocast pointer, `lr_scheduler` spelling),
`seq_math_then_code.py` rebuilt from it (differs only in the header line, `DISTILL_ORDER`, `OUT_DIR`), `kd_train.py`
brought to the same names / comments / hub settings (keeps its molab `DEVICE` and `MICRO = 8`), comment and header typo
fixes in `kd_utils.py` and `bench_utils.py`, all copies synced (`md5sum */bench_utils.py */kd_utils.py` matches).

Check (same cpu harness): old scripts vs current scripts give byte-identical outputs for both seq scripts, kd_train.py
and bench_student.py; the only missing files are the tokenizer files, which the kd scripts no longer save (on purpose).

## 23 Oct 2026 - kd smoke tests (department server, student GPU 2 RTX A6000, teachers GPU 3 RTX 6000 Ada)

Settings: a temporary copy of `seq_code_then_math.py` that trains on the 64 longest rows per domain (worst case for
memory: longest answers 1264 math, 1089 code), `EPOCHS = 1` (2 steps per stage), `MAX_QUESTIONS = 16`, no upload,
otherwise unchanged (`MICRO = 4`, `BATCH = 32`, `VAL_SIZE = 200`). Data downloaded from the hub as jajamken.

| run | peak allocated, student gpu | teacher gpu | step time (code / math) | val (both domains) | benchmark (16 q) |
|-----|-----------------------------|-------------|-------------------------|--------------------|------------------|
| 1   | 45.7 GB of 50.9             | 32.9 GB     | 14 s / 9 s              | 50 s               | 54-59 s          |
| 2   | 43.3 GB of 50.9             | 32.9 GB     | 14 s / 8 s              | 50 s               | 50-53 s          |

Run 2 has `kd_utils.losses()` freeing the logits early (`del t, s` and `del t_logp`, cpu outputs byte-identical to
before). Both runs went through both stages, save, benchmark and evalplus without errors. Val after 2 code steps:
code ce 0.367 -> 0.340; after 2 math steps: math ce 0.832 -> 0.620, code ce unchanged (0.340). Stage 2's step-0 val
matches stage 1's last val up to bf16 rounding (the bf16 save and continue works).

GPU training is not bit-reproducible: step-0 val and step-1 losses are identical in both runs, from step 2 on they differ
in the 4th decimal (backward kernels use atomic adds whose order varies), and gsm8k on 16 questions came out 3/16 vs
4/16 at stage 1. Evaluation alone is deterministic (refactor check). So two identical training runs end slightly
different; differences of about 1 point between kd arms are not results.

Memory note for the joint run: about 38 GB of the student-gpu peak grows with MICRO, so `MICRO = 8` with the teachers
(30.5 GB) on the same 96 GB molab gpu would not fit on long batches; use `MICRO = 4` there.

## 23 Oct 2026 - main sequential runs started (department server, GPUs 2 + 3) - INVALID (rope bf16 rounding bug, see the 20:07 UTC entry)

`seq_math_then_code.py` then `seq_code_then_math.py`, chained in one nohup (`CUDA_VISIBLE_DEVICES=2,3`), started
17:07 UTC, so both runs use the same gpus (student on the A6000, both teachers on the Ada). Settings as in the scripts:
ALPHA 0.5, TEMPERATURE 1.0, EPOCHS 2, BATCH 32, MICRO 4, LR 1e-5, VAL_SIZE 200, SEED 0, results uploaded to the
`jajamken/FML_project` model repo after each stage. Results: to be filled in.

### seq_math_then_code.py, stage 1 (math), finished 18:00 UTC - INVALID (rope bf16 rounding bug, see the 20:07 UTC entry)

Training 17:07-17:43 UTC (36 min incl. model loading and 3 val passes), benchmark 17:43-18:00 (17 min).

| benchmark | base student    | after math kd   | change         |
|-----------|-----------------|-----------------|----------------|
| GSM8K     | 651/1319 = 49.4 | 553/1319 = 41.9 | -98 questions  |
| MBPP      | 183/378 = 48.4  | 174/378 = 46.0  | -9             |
| MBPP+     | 156/378 = 41.3  | 147/378 = 38.9  | -9             |
| HumanEval | 62/164 = 37.8   | 52/164 = 31.7   | -10            |
| HE+       | 56/164 = 34.1   | 47/164 = 28.7   | -9             |

Val (200 prompts per domain, per-token KL / CE):

| step        | math KL | math CE | code KL | code CE |
|-------------|---------|---------|---------|---------|
| 0           | 0.171   | 0.839   | 0.221   | 0.367   |
| 226 (ep. 1) | 0.136   | 0.540   | 0.246   | 0.400   |
| 452 (ep. 2) | 0.133   | 0.562   | 0.245   | 0.399   |

Train CE drops from ~0.50 to ~0.35 when epoch 2 starts (the same data again) while math val CE rises 0.540 -> 0.562:
epoch 2 overfits the gold answers. Code val gets ~10% worse in epoch 1 (forgetting), then stays.

Why GSM8K went down: the student mostly copies the short gold-answer style, not the teacher's. Every gold answer in
the training data ends in "The answer is \boxed{...}." and no teacher answer does; the kd student writes that phrase in
931/1319 GSM8K answers (71%), and its answers are much shorter (median 108 tokens vs 294 for the base student; the gold
answers have median 87, the teacher's 288). Accuracy is 41.4% on those gold-style answers and 43.3% on the rest, both
below the base 49.4%, so the shorter style is not the whole story (the epoch-2 overfitting likely adds to it). 236
questions the base student got right are now wrong, 138 the other way. The gold-CE ablation (ALPHA = 1.0, KL only)
tests this directly.

## 23 Oct 2026 - ablation started: KL only, math stage (department server, GPUs 0 + 1, both RTX A6000) - INVALID (rope bf16 rounding bug, see the 20:07 UTC entry)

`2_sequential_kd/abl_kl_only_math.py` = `seq_math_then_code.py` with `ALPHA = 1.0` (no gold CE, CE still computed and
logged), `DISTILL_ORDER = ["math"]`, `OUT_DIR = "abl_kl_only_math"`. Same seed, so the same batches in the same order as
stage 1 of the main run; the only difference is the missing gold CE term. Tests whether the gold CE term causes the
GSM8K drop (the student copying the short gold style). Started 18:51 UTC with `python -u` (live log), in parallel with
the main seq runs on GPUs 2 + 3. The student is on an A6000 like in the main runs; the teachers are on an A6000 here
instead of the Ada, which only changes bf16 rounding in the teacher logits.

### seq_math_then_code.py, stage 2 (code), benchmark finished ~18:52 UTC - INVALID (rope bf16 rounding bug, see the 20:07 UTC entry)

Training 18:00-18:40 UTC (40 min), benchmark ~18:40-18:52.

| benchmark | base student    | after stage 1 (math) | after stage 2 (code) |
|-----------|-----------------|----------------------|----------------------|
| GSM8K     | 651/1319 = 49.4 | 553/1319 = 41.9      | 564/1319 = 42.8      |
| MBPP      | 183/378 = 48.4  | 174/378 = 46.0       | 186/378 = 49.2       |
| MBPP+     | 156/378 = 41.3  | 147/378 = 38.9       | 162/378 = 42.9       |
| HumanEval | 62/164 = 37.8   | 52/164 = 31.7        | 58/164 = 35.4        |
| HE+       | 56/164 = 34.1   | 47/164 = 28.7        | 53/164 = 32.3        |

Val (per-token KL / CE) during stage 2: code KL 0.245 -> 0.170 -> 0.162, code CE 0.399 -> 0.297 -> 0.293 (well below
the base student's 0.221 / 0.367); math KL 0.133 -> 0.145 -> 0.152, math CE 0.562 -> 0.559 -> 0.572 (forgetting, but
math KL stays below the base student's 0.171).

Notes: code kd brings the code benchmarks back to about the base student (MBPP +3 and MBPP+ +6 questions above base,
HumanEval -4 and HE+ -3 below), even though code val KL / CE improve a lot. GSM8K does not drop further on the code stage
(+11 questions vs stage 1); the share of gold-style math answers falls from 931 to 685 of 1319 (40.0% right on those,
45.7% on the rest), so the code stage partly undoes the copied gold style.

## 23 Oct 2026 - ablation queued: gold SFT only (ALPHA = 0.0), math then code (department server, GPUs 0 + 1) - INVALID (rope bf16 rounding bug, see the 20:07 UTC entry)

`2_sequential_kd/abl_sft_math_code.py` = `seq_math_then_code.py` with `ALPHA = 0.0` (no KL, KL still computed and
logged), `OUT_DIR = "abl_sft_math_code"`, same seed, both stages. With the KL-only run this gives ALPHA = 0 / 0.5 / 1.0
for the gold CE weight. Runs from `/tmp/jayk_runs/2_sequential_kd/` (private folder, not under the home quota, which
was at 21.9 of 27.6 GB); every stage is uploaded to the hub. A waiter (`waiter.sh`, nohup on the server) starts it
when the KL-only run ends and GPUs 0 and 1 are free, with `python -u`.

### abl_kl_only_math.py (ALPHA = 1.0, math stage only), finished 19:46 UTC - INVALID (rope bf16 rounding bug, see the 20:07 UTC entry)

Training 18:51-19:32 UTC, benchmark 19:32-19:46. Uploaded to the hub (`abl_kl_only_math/`).

| benchmark | base student    | math kd, ALPHA = 0.5 | math kd, KL only (ALPHA = 1.0) |
|-----------|-----------------|----------------------|--------------------------------|
| GSM8K     | 651/1319 = 49.4 | 553/1319 = 41.9      | 673/1319 = 51.0                |
| MBPP      | 183/378 = 48.4  | 174/378 = 46.0       | 184/378 = 48.7                 |
| MBPP+     | 156/378 = 41.3  | 147/378 = 38.9       | 159/378 = 42.1                 |
| HumanEval | 62/164 = 37.8   | 52/164 = 31.7        | 56/164 = 34.1                  |
| HE+       | 56/164 = 34.1   | 47/164 = 28.7        | 46/164 = 28.0                  |

| val (step 452) | math KL | math CE | code KL | code CE |
|----------------|---------|---------|---------|---------|
| ALPHA = 0.5    | 0.133   | 0.562   | 0.245   | 0.399   |
| ALPHA = 1.0    | 0.118   | 0.886   | 0.245   | 0.387   |

Answer style on GSM8K: 0/1319 answers in the gold style (vs 931 at ALPHA = 0.5), median 293 tokens (base 294, ALPHA
0.5: 108); 15 hit the length limit (base 16). So without the gold CE term the student keeps the long step-by-step
style and GSM8K goes from 41.9 to 51.0 (+120 questions); this supports the gold-style explanation for the drop at ALPHA
0.5. Against the base student it is +22 questions net (158 gained, 136 lost), a small gain given the run-to-run noise.
HumanEval+ is still 10 below base after math-only training (code forgetting), MBPP / MBPP+ are at base level.
Math val CE on the gold answers does not improve at all (0.839 -> 0.886): the student moves toward the teacher, not
toward the gold answers.

### seq_code_then_math.py, stage 1 (code), finished ~19:50 UTC - INVALID (rope bf16 rounding bug, see the 20:07 UTC entry)

Started 18:52 UTC (right after seq_math_then_code.py), training ~37 min, benchmark ~16 min.

| benchmark | base student    | after code kd   | change |
|-----------|-----------------|-----------------|--------|
| GSM8K     | 651/1319 = 49.4 | 606/1319 = 45.9 | -45    |
| MBPP      | 183/378 = 48.4  | 193/378 = 51.1  | +10    |
| MBPP+     | 156/378 = 41.3  | 165/378 = 43.7  | +9     |
| HumanEval | 62/164 = 37.8   | 54/164 = 32.9   | -8     |
| HE+       | 56/164 = 34.1   | 50/164 = 30.5   | -6     |

Val: code KL 0.221 -> 0.170 -> 0.162, code CE 0.367 -> 0.295 -> 0.293 (the same end point as the code stage of the other
order); math KL 0.171 -> 0.179 -> 0.183 (forgetting), math CE 0.839 -> 0.756 -> 0.773 (math gold CE improves while
training only on code; one guess is that the short gold code answers also make the short math gold answers more
likely, not checked).

Notes: code kd helps MBPP / MBPP+ but hurts HumanEval / HE+, and costs 45 GSM8K questions (forgetting). Not analysed
yet: why HumanEval goes down (answer format, function names, or the plain ``` vs ```python code blocks of the teacher).

## 23 Oct 2026 20:07 UTC - rope bf16 rounding bug, all kd runs stopped and restarted

Bug: the kd scripts benchmarked the student after `student.to(torch.bfloat16)` and continued stage 2 after
`.to(torch.float32)`. `.to(dtype)` also casts rope's `inv_freq` buffer, which is not a saved weight: after `.to(bf16)`
it is rounded to bf16 (max relative error 3.65e-3, a rotary phase error up to 0.94 rad at position 1000, checked on the
server with transformers 5.20.2). `from_pretrained(..., dtype=torch.bfloat16)`, used for the baselines, recomputes it
exactly in fp32. So every kd benchmark above ran with rounded rope while the baselines did not, and every stage 2 trained
with rounded rope. Stage 1 training and the saved checkpoints were not affected. Found because a continuation script that
loads the stage 1 checkpoint differed from an uninterrupted run already at step 0 of stage 2.

Fix (seq scripts, both ablations, kd_train.py): after each stage the student is cast to bf16 and saved, then not used
again; the benchmark loads the checkpoint with `from_pretrained(dtype=bf16)` exactly like the baselines, and the next stage
loads it with `from_pretrained(dtype=fp32)`. So every benchmark scores the uploaded artifact and every stage starts from
the saved checkpoint (the same start the post-training methods in section 4 use).

Checks (cpu harness, tiny models): stage 1 training outputs byte-identical to the old code (4/4 files); the old stage 1
benchmark files all differ from the new ones (0/6 identical), so the rounding did change answers; the in-script benchmark
is byte-identical to a separate fresh benchmark of the saved checkpoint (5/5 files, equal scores); stage 2 is
byte-identical to a fresh continuation from the stage 1 checkpoint (10/10 files). All 6 kd scripts run to the end.

Stopped at 20:07 UTC: `seq_code_then_math.py` (in stage 2) and `abl_sft_math_code.py` (in stage 1), plus their chain /
waiter. Cleanup: the invalid outputs and old copies on the server deleted (`~/jayk_fml/2_sequential_kd`, 3.8 GB;
`refactor_check`, `smoke_logs`; the old `/tmp` run files), home quota back to 19.9 of 27.6 GB; old logs kept in
`~/jayk_fml/invalid_run_logs/`; the three invalid folders removed from the hub model repo (one commit, they stay in its
history). The qualitative style finding above (ALPHA 0.5 copies the short gold style, KL only does not) comes from
training and will be rechecked on the new runs; its numbers are invalid.

Rerun, launched 20:14 UTC from `/tmp/jayk_runs/2_sequential_kd/` (private, not under the quota), `python -u`, all full
math+code runs with the fixed scripts and unchanged settings:
- GPUs 2 + 3 (student on the A6000, teachers on the Ada): `seq_math_then_code.py`, then `seq_code_then_math.py`
- GPUs 0 + 1 (both A6000): `abl_kl_only_math_code.py` (ALPHA 1.0), then `abl_sft_math_code.py` (ALPHA 0.0)

## 24 Oct 2026 00:01 UTC - rerun finished: all 4 sequential runs (fixed scripts, department server)

All 4 ran to the end with no tracebacks; all 16 stage / result folders are on the hub (`jajamken/FML_project`). The
allocator printed a few "memory allocation failed with OOM ... free 40 MB" warnings on the student GPU (also in the old
runs): pytorch freed its cached blocks and retried, the retry worked, no effect on the numbers.

Times (UTC, checkpoint saved / benchmark done): `seq_math_code` stage 1 20:51 / 21:08, stage 2 21:50 / 22:05;
`seq_code_math` stage 1 22:48 / 23:00, stage 2 23:36 / 23:49; `abl_kl_only_math_code` stage 1 20:56 / 21:13, stage 2
21:58 / 22:14; `abl_sft_math_code` stage 1 22:54 / 23:07, stage 2 23:52 / 23:59. About 40 min training + 15 min
benchmark per stage.

Checks:
- stage 2 starts where stage 1 ended in all 4 runs: stage 2 step-0 val equals stage 1's final val up to 0.0006 (the
  student is rounded to bf16 when saved), e.g. `seq_math_code` math KL / CE 0.1330 / 0.5615 -> 0.1332 / 0.5615.
- training noise at the loss level: stage 1 training code is the same as in the invalid runs, and their final val
  losses differ by at most 0.0009 (`seq_math_code` 0.1331 / 0.5617 / 0.2451 / 0.3993 old vs 0.1330 / 0.5615 / 0.2453 /
  0.3994 new). This says nothing about benchmark noise; that needs the seed repeats (README TODO).
- the rope bug moved the stage 1 benchmarks by only a few questions (GSM8K 553 -> 560 at ALPHA 0.5, 673 -> 672 KL only;
  this also includes training nondeterminism).

Results, questions right (base student: GSM8K 651/1319, MBPP 183/378, MBPP+ 156/378, HumanEval 62/164, HE+ 56/164):

| run (ALPHA)            | stage  | GSM8K     | MBPP     | MBPP+    | HumanEval | HE+      |
|------------------------|--------|-----------|----------|----------|-----------|----------|
| math -> code (0.5)     | 1 math | 560 (-91) | 174 (-9) | 150 (-6) | 50 (-12)  | 45 (-11) |
| math -> code (0.5)     | 2 code | 561 (-90) | 188 (+5) | 160 (+4) | 56 (-6)   | 51 (-5)  |
| math -> code, SFT (0)  | 1 math | 477 (-174)| 165 (-18)| 139 (-17)| 50 (-12)  | 46 (-10) |
| math -> code, SFT (0)  | 2 code | 492 (-159)| 186 (+3) | 163 (+7) | 50 (-12)  | 46 (-10) |
| math -> code, KL (1.0) | 1 math | 672 (+21) | 191 (+8) | 164 (+8) | 58 (-4)   | 49 (-7)  |
| math -> code, KL (1.0) | 2 code | 646 (-5)  | 193 (+10)| 165 (+9) | 57 (-5)   | 50 (-6)  |
| code -> math (0.5)     | 1 code | 624 (-27) | 191 (+8) | 166 (+10)| 58 (-4)   | 52 (-4)  |
| code -> math (0.5)     | 2 math | 544 (-107)| 180 (-3) | 155 (-1) | 49 (-13)  | 44 (-12) |

Answer style on GSM8K (`~/jayk_fml/gsm_style.py`, `gsm_compare.py`; gold style = ends in "The answer is \boxed{...}"):

| model                    | gold-style answers | median answer tokens | right |
|--------------------------|--------------------|----------------------|-------|
| base student             | 0                  | 294                  | 651   |
| math stage, ALPHA 1.0    | 0                  | 294                  | 672   |
| math stage, ALPHA 0.5    | 933                | 108                  | 560   |
| math stage, ALPHA 0.0    | 1312               | 86                   | 477   |
| code -> math, ALPHA 0.5  | 953                | 107                  | 544   |

Notes:
- gold CE is what lowers GSM8K: the more weight on it, the more the student copies the short gold style and the fewer
  questions it gets right (ALPHA 1.0 / 0.5 / 0.0: 672 / 560 / 477). The style finding of the invalid runs holds. At
  ALPHA 0.5 it is 41.8% right on its gold-style answers and 44.0% on the rest, both below base.
- KL only is the only math stage above the base student on GSM8K (+21 net), small for one seed.
- forgetting: the code stage after math costs no GSM8K at ALPHA 0.5 (560 -> 561; gold-style answers 933 -> 706) but 26
  questions KL only (672 -> 646). The math stage after code costs 11 MBPP and 9 HumanEval questions (191 -> 180,
  58 -> 49). Code kd alone costs 27 GSM8K questions, and 307 of its GSM8K answers have no \boxed{} (base 30), so part of
  that is the student dropping the asked-for answer format.
- order at ALPHA 0.5: math -> code ends ahead of code -> math on all five benchmarks (GSM8K +17, MBPP +8, MBPP+ +5,
  HumanEval +7, HE+ +7 questions), one seed.
- HumanEval is below base after every stage of every run (-4 to -13 questions), while MBPP goes up after every code
  stage. Both use the same evalplus prompt, so the train / eval prompt difference alone does not explain it. Not analysed
  yet.

## 24 Oct 2026 ~00:10 UTC - filtered teacher answers ablation: started, then dropped

Dropped by decision (not planned for the checkpoint); nothing was trained, the unfinished scripts are not in the repo.
What came out of it:
- math: 7144 of the 7441 kept math teacher answers have the right final number, 297 are wrong (matches the 0.956
  accuracy in gen_data.py, 7144 / 7473).
- code: a sandboxed test runner (bwrap: no network, read-only filesystem, home hidden; pytest is not installed, so only
  plain test functions and `pytest.raises`) was tried on the first 300 code rows: teacher answers 228 pass / 66 fail /
  6 untested, kodcode's own solutions 281 / 13 / 6. 11 of those 13 solution failures were not explained, so the runner
  was not trusted yet.
- benchmark quirk found on the way: evalplus 0.3.1's `sanitize`, which our benchmark uses to extract code, drops
  decorated top-level functions (checked: `@functools.lru_cache def fib` is removed). An answer that defines its entry
  point with a decorator is scored as failing. It applies to every model the same way, and it is evalplus's standard
  pipeline.

## 24 Oct 2026 - server cleanup after the sequential runs

The 8 local stage checkpoints were deleted after checking that all 32 of their files are byte-identical on the hub
(sha256 for the weights). The hf cache's models (the student and both teachers) and the kodcode dataset were deleted too
(31.6 GB); scripts re-download them. Kept on the server: the run scripts, logs, `data/`, the `results_*` folders
(answers for later analysis), the gsm8k and project dataset caches, the hf login token, and `~/jayk_fml/` (baseline
results, analysis scripts, invalid run logs). Our files now take ~100 MB; `/` went from 22 GB to 59 GB free.

## 24 Oct 2026 00:24 UTC - code -> math ablations started (department server, all 4 GPUs)

The math -> code order has the gold SFT (ALPHA 0.0) and KL only (ALPHA 1.0) ablations, so code -> math gets them too:
`2_sequential_kd/abl_kl_only_code_math.py` and `abl_sft_code_math.py` = `seq_code_then_math.py` with `ALPHA = 1.0` /
`0.0` and their own `OUT_DIR`; each differs from its math -> code twin only in `DISTILL_ORDER`, `OUT_DIR` and the header.
Same seed and settings. The three models were downloaded again first (4.5 min, 30 GB in `/tmp/jayk_hf`).
- GPUs 0 + 1 (both A6000): `abl_kl_only_code_math.py`, the same pair its math -> code twin used
- GPUs 2 + 3 (student on the A6000, teachers on the Ada): `abl_sft_code_math.py`. At ALPHA 0 the KL has zero weight, so
  the teacher GPU only affects the logged val KL

## 24 Oct 2026 00:45 UTC - seeds 1 and 2 queued for all 6 sequential runs (department server)

Plan (user): sequential seeds first, then everything joint on Saturday evening / night, results by Sunday morning,
then analysis and report. Teacher SFT and the temperature ablation are postponed until after the 26 Oct checkpoint.

Change in all 6 sequential scripts: `SEED = int(os.environ.get("SEED", 0))` and `OUT_DIR = "<name>" + (f"_seed{SEED}" if
SEED else "")`, so seed 0 keeps the first runs' names and a run is `SEED=1 python -u <script>`; the repo file is exactly
what runs. The seed sets the shuffled data order of every epoch (`random.shuffle`); the pretrained start, the val rows
and the benchmarks are the same for every seed.

Two server-side chains (`chain.sh`, nohup) wait for the running code -> math run on their GPU pair to end, then run
seed 1 of three scripts, then seed 2, each on the same GPU pair as its seed 0 run:
- GPUs 0 + 1: `abl_kl_only_math_code.py`, `abl_sft_math_code.py`, `abl_kl_only_code_math.py`
- GPUs 2 + 3: `seq_math_then_code.py`, `seq_code_then_math.py`, `abl_sft_code_math.py`
Before each run the chain waits until both GPUs are free. After a run exits 0, `verify_delete.py` deletes its local
checkpoints only if every file is on the hub with the same sha256 (weights) or size; results and logs stay. Logs:
`<script>_seed<N>.log`, `chain_01.log`, `chain_23.log`. 12 runs of ~2 h on 2 pairs: done around 14:30 UTC (20:00 IST); with the measured 1 h 40 min - 1 h 55 min per run, more likely ~13:30 UTC.

## 24 Oct 2026 ~01:15 UTC - joint scripts prepared for tonight (not run yet, user to review)

`3_joint_kd/kd_train.py`: the same `SEED` / `OUT_DIR` change as the sequential scripts (`joint_kd`, `joint_kd_seed1`, ...),
and `DEVICE` set to the server layout (student on cuda:0, both teachers on cuda:1, like the sequential runs; molab
would be all cuda:0). New `3_joint_kd/abl_kl_only_joint.py` (ALPHA 1.0) and `abl_sft_joint.py` (ALPHA 0.0) = `kd_train.py`
with ALPHA, OUT_DIR and the header changed.

Check (cpu harness, tiny random models, fake datasets / evalplus, hub calls blocked): all 6 test runs exit 0 (joint
SEED 0 and 1, both joint ablations, seq_math_then_code SEED 0 and 2); joint SEED 0 is byte-identical to the earlier
verified fix-check run (10/10 files) and seq_math_then_code SEED 0 too (20/20), so the change does nothing at seed 0;
SEED 1 / 2 change the outputs (3/10 and 4/20 files identical, the configs) and write to the `_seed1` / `_seed2` folders.

## 24 Oct 2026 02:16 UTC - code -> math ablations finished (department server)

`abl_sft_code_math` ended 02:04 UTC, `abl_kl_only_code_math` 02:16 UTC (1 h 40 min / 1 h 52 min), no tracebacks. All 8
checkpoint files of each are on the hub with the same sha256 / size, so the chain deleted the local copies; results and
logs stay. Stage 2 starts where stage 1 ended (step-0 val within 0.0011, e.g. SFT math CE 0.8736 -> 0.8725). The seed
chains started right after: `seq_math_then_code.py` SEED 1 on GPUs 2 + 3 at 02:04, `abl_kl_only_math_code.py` SEED 1 on
GPUs 0 + 1 at 02:16.

Results, questions right (base student: GSM8K 651/1319, MBPP 183/378, MBPP+ 156/378, HumanEval 62/164, HE+ 56/164;
the ALPHA 0.5 rows are from the 00:01 UTC entry):

| run (ALPHA)            | stage  | GSM8K     | MBPP     | MBPP+    | HumanEval | HE+      |
|------------------------|--------|-----------|----------|----------|-----------|----------|
| code -> math (0.5)     | 1 code | 624 (-27) | 191 (+8) | 166 (+10)| 58 (-4)   | 52 (-4)  |
| code -> math (0.5)     | 2 math | 544 (-107)| 180 (-3) | 155 (-1) | 49 (-13)  | 44 (-12) |
| code -> math, SFT (0)  | 1 code | 634 (-17) | 182 (-1) | 163 (+7) | 56 (-6)   | 52 (-4)  |
| code -> math, SFT (0)  | 2 math | 471 (-180)| 183 (0)  | 163 (+7) | 50 (-12)  | 47 (-9)  |
| code -> math, KL (1.0) | 1 code | 601 (-50) | 184 (+1) | 161 (+5) | 62 (0)    | 55 (-1)  |
| code -> math, KL (1.0) | 2 math | 666 (+15) | 184 (+1) | 162 (+6) | 52 (-10)  | 45 (-11) |

GSM8K answer style (ground truth style = ends in "The answer is \boxed{...}"; no boxed = no `\boxed{}` anywhere, the
scorer then takes the last number):

| model                    | ground truth style | no boxed | median answer tokens | right |
|--------------------------|--------------------|----------|----------------------|-------|
| base student             | 0                  | 30       | 294                  | 651   |
| code stage, ALPHA 1.0    | 0                  | 176      | 284                  | 601   |
| code stage, ALPHA 0.5    | 0                  | 307      | 271                  | 624   |
| code stage, ALPHA 0.0    | 0                  | 428      | 249                  | 634   |
| code -> math, ALPHA 1.0  | 0                  | 12       | 294                  | 666   |
| code -> math, ALPHA 0.0  | 1314               | 4        | 84                   | 471   |

Notes:
- the ground truth style result holds in this order too: the ground truth SFT math stage gives 1314 ground truth style
  answers and 471 right (math -> code: 1312 and 477), KL only 0 and 666 (math -> code: 0 and 672). Both orders, KL only
  is the only math stage that ends above the base student on GSM8K.
- SFT overfits in epoch 2: math val CE 0.5346 after epoch 1 -> 0.5538 after epoch 2 in its math stage. Only the epoch 2
  student is benchmarked.
- the code stage's GSM8K answers lose `\boxed{}` more the more weight the ground truth CE has (176 / 307 / 428 at ALPHA
  1.0 / 0.5 / 0.0; kodcode's ground truth answers are code only). GSM8K right does not follow that order (601 / 624 /
  634); the gaps are 33 questions or less, one seed.
- the KL only code stage is the only stage of any run that keeps HumanEval at base (62); every math stage after it drops
  HumanEval (-6 to -10 questions here) while MBPP stays within 1 question for SFT and KL only.
- order, final stage: KL only ends better on the domain trained last (code -> math GSM8K 666 vs 646; math -> code MBPP
  193 vs 184, HumanEval 57 vs 52). With ground truth CE (ALPHA 0.5 and 0.0), math -> code ends ahead on GSM8K too (561 vs
  544, 492 vs 471), because the math stage itself lowers GSM8K and the later code stage partly undoes its style.

## 24 Oct 2026 - seed runs, results as they finish (department server)

One row per finished stage, questions right; seed 0 is the run from the earlier entries. Every row: exit 0, no
tracebacks, all 8 checkpoint files on the hub (sha256 / size), local checkpoints deleted by the chain.

| run (ALPHA)            | seed | finished (UTC) | stage  | GSM8K | MBPP | MBPP+ | HumanEval | HE+ | val KL / CE math | val KL / CE code |
|------------------------|------|----------------|--------|-------|------|-------|-----------|-----|------------------|------------------|
| math -> code (0.5)     | 0    | 22:05 (23 Oct) | 1 math | 560   | 174  | 150   | 50        | 45  | 0.1330 / 0.5615  | 0.2453 / 0.3994  |
| math -> code (0.5)     | 0    |                | 2 code | 561   | 188  | 160   | 56        | 51  | 0.1521 / 0.5711  | 0.1621 / 0.2925  |
| math -> code (0.5)     | 1    | 03:48          | 1 math | 566   | 170  | 147   | 50        | 47  | 0.1336 / 0.5579  | 0.2457 / 0.3996  |
| math -> code (0.5)     | 1    |                | 2 code | 571   | 188  | 160   | 60        | 57  | 0.1524 / 0.5722  | 0.1640 / 0.2939  |
| math -> code (0.5)     | 2    | 08:56          | 1 math | 549   | 169  | 146   | 51        | 47  | 0.1342 / 0.5557  | 0.2428 / 0.3970  |
| math -> code (0.5)     | 2    |                | 2 code | 578   | 190  | 162   | 58        | 54  | 0.1528 / 0.5719  | 0.1642 / 0.2937  |
| math -> code (1.0)     | 0    | 22:14 (23 Oct) | 1 math | 672   | 191  | 164   | 58        | 49  | 0.1177 / 0.8850  | 0.2456 / 0.3870  |
| math -> code (1.0)     | 0    |                | 2 code | 646   | 193  | 165   | 57        | 50  | 0.1396 / 0.8694  | 0.1510 / 0.3664  |
| math -> code (1.0)     | 1    | 04:11          | 1 math | 688   | 186  | 159   | 57        | 47  | 0.1181 / 0.8793  | 0.2465 / 0.3849  |
| math -> code (1.0)     | 1    |                | 2 code | 637   | 183  | 156   | 55        | 49  | 0.1392 / 0.8499  | 0.1531 / 0.3677  |
| math -> code (1.0)     | 2    | 10:01          | 1 math | 686   | 186  | 160   | 59        | 49  | 0.1180 / 0.8905  | 0.2443 / 0.3856  |
| math -> code (1.0)     | 2    |                | 2 code | 627   | 181  | 155   | 56        | 49  | 0.1399 / 0.8651  | 0.1535 / 0.3674  |
| code -> math (0.5)     | 0    | 23:49 (23 Oct) | 1 code | 624   | 191  | 166   | 58        | 52  | 0.1831 / 0.7737  | 0.1619 / 0.2930  |
| code -> math (0.5)     | 0    |                | 2 math | 544   | 180  | 155   | 49        | 44  | 0.1337 / 0.5546  | 0.1823 / 0.3077  |
| code -> math (0.5)     | 1    | 05:31          | 1 code | 626   | 184  | 157   | 57        | 53  | 0.1834 / 0.7832  | 0.1615 / 0.2921  |
| code -> math (0.5)     | 1    |                | 2 math | 561   | 175  | 154   | 52        | 46  | 0.1346 / 0.5582  | 0.1838 / 0.3090  |
| code -> math (0.5)     | 2    | 10:57          | 1 code | 611   | 178  | 153   | 57        | 53  | 0.1848 / 0.7856  | 0.1629 / 0.2934  |
| code -> math (0.5)     | 2    |                | 2 math | 511   | 179  | 156   | 50        | 44  | 0.1341 / 0.5579  | 0.1829 / 0.3071  |
| math -> code (0.0)     | 0    | 23:59 (23 Oct) | 1 math | 477   | 165  | 139   | 50        | 46  | 0.3173 / 0.5619  | 0.2612 / 0.4137  |
| math -> code (0.0)     | 0    |                | 2 code | 492   | 186  | 163   | 50        | 46  | 0.2776 / 0.5979  | 0.4055 / 0.2999  |
| math -> code (0.0)     | 1    | 05:57          | 1 math | 477   | 171  | 145   | 49        | 46  | 0.3169 / 0.5584  | 0.2605 / 0.4124  |
| math -> code (0.0)     | 1    |                | 2 code | 470   | 189  | 163   | 52        | 44  | 0.2791 / 0.5963  | 0.4226 / 0.3018  |
| math -> code (0.0)     | 2    | 11:52          | 1 math | 491   | 168  | 140   | 51        | 45  | 0.3111 / 0.5594  | 0.2570 / 0.4100  |
| math -> code (0.0)     | 2    |                | 2 code | 439   | 180  | 159   | 50        | 46  | 0.2768 / 0.6033  | 0.4224 / 0.3015  |
| code -> math (0.0)     | 0    | 02:04          | 1 code | 634   | 182  | 163   | 56        | 52  | 0.1961 / 0.8736  | 0.3828 / 0.3004  |
| code -> math (0.0)     | 0    |                | 2 math | 471   | 183  | 163   | 50        | 47  | 0.3443 / 0.5538  | 0.3347 / 0.3128  |
| code -> math (0.0)     | 1    | 07:12          | 1 code | 629   | 183  | 159   | 58        | 54  | 0.1962 / 0.8738  | 0.3849 / 0.3005  |
| code -> math (0.0)     | 1    |                | 2 math | 481   | 179  | 154   | 51        | 47  | 0.3401 / 0.5590  | 0.3320 / 0.3098  |
| code -> math (0.0)     | 2    | 12:37          | 1 code | 628   | 186  | 165   | 59        | 52  | 0.1978 / 0.8792  | 0.3904 / 0.3004  |
| code -> math (0.0)     | 2    |                | 2 math | 475   | 173  | 153   | 50        | 47  | 0.3441 / 0.5564  | 0.3365 / 0.3139  |
| code -> math (1.0)     | 0    | 02:16          | 1 code | 601   | 184  | 161   | 62        | 55  | 0.1814 / 0.7509  | 0.1517 / 0.3672  |
| code -> math (1.0)     | 0    |                | 2 math | 666   | 184  | 162   | 52        | 45  | 0.1188 / 0.8739  | 0.1707 / 0.3735  |
| code -> math (1.0)     | 1    | 08:02          | 1 code | 614   | 184  | 158   | 56        | 50  | 0.1814 / 0.7556  | 0.1505 / 0.3665  |
| code -> math (1.0)     | 1    |                | 2 math | 671   | 182  | 158   | 52        | 47  | 0.1186 / 0.8821  | 0.1715 / 0.3750  |
| code -> math (1.0)     | 2    | 13:48          | 1 code | 610   | 189  | 164   | 56        | 49  | 0.1825 / 0.7622  | 0.1529 / 0.3678  |
| code -> math (1.0)     | 2    |                | 2 math | 669   | 179  | 158   | 59        | 53  | 0.1187 / 0.8786  | 0.1730 / 0.3761  |

All 18 sequential runs (6 settings x seeds 0, 1, 2) done at 13:48 UTC, every one exit 0, no tracebacks, checkpoints on
the hub. Both chains ended, all 4 GPUs free. KD only seed 2 after code: HumanEval 56 -> 59, the only math stage after a
code stage that does not drop HumanEval (seeds 0 / 1: 62 -> 52, 56 -> 52), so "every math stage after code drops
HumanEval" holds in 8 of 9 runs. KD only GSM8K is above base after all 6 math stages (666 to 688).

All 6 seed 1 runs done at 08:02 UTC. Seed 2 started 07:12 (GPUs 2 + 3) and 08:02 (GPUs 0 + 1).

code -> math (0.5) seed 2 ends at GSM8K 511, 33 / 50 questions below seeds 0 / 1 (544, 561): the largest seed gap so
far (3.8 pp), above the 22 seen up to seed 1. The val losses do not show it (math KL / CE 0.1341 / 0.5579 vs 0.1337 /
0.5546 and 0.1346 / 0.5582). Answer style per seed of this stage (ground truth style = contains "The answer is
\boxed{"):

| seed | right | ground truth style: right / n | other: right / n | no \boxed |
|------|-------|-------------------------------|------------------|-----------|
| 0    | 544   | 378 / 953 (39.7%)             | 166 / 366 (45.4%)| 11        |
| 1    | 561   | 384 / 946 (40.6%)             | 177 / 373 (47.4%)| 9         |
| 2    | 511   | 347 / 922 (37.6%)             | 164 / 397 (41.3%)| 9         |

Same style mix, lower accuracy in both styles, so it is not a style shift. Its code stage before was also the lowest
on GSM8K (611 vs 624, 626) with the most answers without `\boxed{}` (437 vs 307, 329).

math -> code (0.0) seed 2: the code stage drops GSM8K 491 -> 439 (-52), vs +15 and -7 in seeds 0 / 1. 58 of its GSM8K
answers are a ```` ```python ```` block with no final number (67 without `\boxed{}`), all scored wrong. Those 67
questions were 26 right after stage 1; the other 1252 questions fall 465 -> 439, so half the drop is the format, half
lower accuracy. GSM8K answers with a python block, after the last stage:

| run                        | seed 0 | seed 1 | seed 2 |
|----------------------------|--------|--------|--------|
| math -> code, ALPHA 0.0    | 3      | 6      | 58     |
| math -> code, ALPHA 0.5    | 0      | 0      | 0      |
| math -> code, ALPHA 1.0    | 0      | 0      | 0      |

Every code -> math run (both stages, ALPHA 0.0 / 0.5 / 1.0, seeds 0-2) has 0, except 1 after the ground truth SFT code
stage from base in seed 2. So the student answers math with code almost only after a ground truth SFT code stage that
follows a ground truth SFT math stage, and how often varies a lot with the seed.

Benchmark sizes differ, so one question is 0.08 pp on GSM8K (1319), 0.26 pp on MBPP / MBPP+ (378) and 0.61 pp on
HumanEval / HE+ (164); gaps are given in both. `~/jayk_fml/results_table.py` (server, run in the runs folder) prints
every finished run as questions right, % and pp vs base, per seed and as the mean over the finished seeds.

Seed noise so far (seed 0 vs 1, same stage): GSM8K up to 22 questions apart (1.7 pp), MBPP 10 (2.6 pp), MBPP+ 9
(2.4 pp), HumanEval 6 (3.7 pp, KL only code stage 62 vs 56), HE+ 6 (3.7 pp). Val loss of a trained loss term on the trained domain: within 0.004. The rest (the other domain, or the term with
zero weight: CE at ALPHA 1.0, KL at ALPHA 0.0) varies up to 0.02. So a few questions between settings is within noise:
at ALPHA 0.5 HE+ ends at 51 (seed 0) and 57 (seed 1), on both sides of base (56); at ALPHA 1.0 the code stage's MBPP
gain is +2 (seed 0) and -3 (seed 1); at ALPHA 0.0 the code stage's GSM8K change is +15 (seed 0) and -7 (seed 1). What
holds in both seeds: the math stage's GSM8K at ALPHA 1.0 / 0.5 / 0.0 (seed 0: 672 / 560 / 477; seed 1: 688 / 566 / 477),
and at ALPHA 0.5 math -> code ends ahead of code -> math on all five benchmarks (seed 0: GSM8K +17, MBPP +8, MBPP+ +5,
HumanEval +7, HE+ +7; seed 1: +10, +13, +6, +8, +11).
Did not hold in seed 1 (seed 0 claims from the earlier entries): the KL only code stage keeping HumanEval at base (62
-> 56 in seed 1); the SFT code stage recovering GSM8K (+15 -> -7); with SFT, math -> code ending ahead of code -> math on
GSM8K (492 vs 471 in seed 0, 470 vs 481 in seed 1). Digest for review: `logs_review.md`.

## 24 Oct 2026 14:00 UTC - all 3 seeds in: results copied, README means, logs_review.md

Copied every run's files from the server (`/tmp/jayk_runs/2_sequential_kd`, lost on a reboot) to
`2_sequential_kd/runs/`: the 18 run folders (results_stage1_* / results_stage2_*: summary.json, per-question
gsm8k.jsonl, mbpp.jsonl, humaneval.jsonl, evalplus eval_results), the 18 training logs, the chain logs, `chain.sh` and
`verify_delete.py`. 238 files, 67 MB, md5 identical to the server. Not copied: `data/` (the teacher answers are on the
hub, `jajamken/FML_project`) and the scripts (md5 identical to `2_sequential_kd/`). Checkpoints are only on the hub.

`2_sequential_kd/results_table.py` reads `runs/` and prints the README rows (3-seed mean %, pp vs base, val losses
after each stage) and every seed as questions right, % (pp). README section 2 now holds the 3-seed means (was seed 0);
the seed TODO is ticked for sequential.

3-seed checks (on `runs/`, for `logs_review.md`):
- largest gap between the 3 seeds of one setting: GSM8K 53 questions (4.0 pp, SFT math -> code stage 2: 492 / 470 /
  439), MBPP 13, MBPP+ 13, HumanEval 7, HE+ 8
- ground truth style GSM8K answers after the math stage from base: 0 / 0 / 0 (ALPHA 1.0), 933 / 905 / 916 (0.5), 1312 /
  1312 / 1313 (0.0); at 0.5 they are 38.0-41.8% right
- GSM8K answers without `\boxed{}` after the code stage from base: 176 / 179 / 217 (1.0), 307 / 329 / 437 (0.5), 428 /
  400 / 493 (0.0), base 30; after a math stage at most 24, except SFT seed 2 (67)
- math val CE rises from epoch 1 to 2 in all 12 runs with ground truth CE (+0.016 to +0.023); math val KL under KD
  only falls in all 6; code val CE moves by at most 0.005
- HumanEval after the code stage that follows a math stage: ALPHA 0.5 50 -> 56, 50 -> 60, 51 -> 58; SFT +0, +3, -1;
  KD only -1, -2, -3

`logs_review.md` rewritten for 3 seeds: 9 findings, all of which hold in 3 of 3 seeds (finding 5 only for math stages
with ground truth CE, finding 6 only at ALPHA 0.5), and 5 claims that did not survive.
