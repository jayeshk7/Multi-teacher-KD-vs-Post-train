# LOG.md review: what matters for the presentation and the report

Digest of LOG.md, 24 Oct 2026 14:00 UTC, all 18 sequential runs (6 settings x seeds 0, 1, 2). Results are questions
right and % (pp = percentage points vs the base student). One question is 0.08 pp on GSM8K (1319), 0.26 pp on MBPP /
MBPP+ (378), 0.61 pp on HumanEval / HE+ (164). The README tables are the 3-seed means; every run's result files are in
`2_sequential_kd/runs/`, and `2_sequential_kd/results_table.py` prints the means and every seed from them.

Base student: GSM8K 651 (49.4%), MBPP 183 (48.4%), MBPP+ 156 (41.3%), HumanEval 62 (37.8%), HE+ 56 (34.1%).

## 1. Setup facts you should be able to answer questions on

1. Data: the math teacher answered all 7473 GSM8K train prompts, the code teacher 7473 KodCode prompts (greedy, up to
   1600 tokens). 32 math and 4 code answers never finished (repetition loops); those rows are dropped for every
   setting, ground truth SFT included, so all settings train on the same 7441 / 7469 rows.
2. Of the 7441 kept math answers, 297 (4.0%) end in a wrong number. They are kept, so KD distills them; the ground truth
   CE term still uses the right solution.
3. Ground truth answers: GSM8K's solution with the `<<48/2=24>>` calculator notes removed plus "The answer is
   \boxed{N}."; KodCode's solution as a bare ```` ```python ```` block, no text (`0_data_gen/gen_data.py:46-49`).
4. Training budget: fixed, 2 epochs per stage, 452 math + 454 code optimizer steps (joint: the same 906 in one run),
   cosine LR to 0. No early stopping: a fixed budget keeps the settings comparable, and the val losses do not track
   benchmark accuracy (findings 3, 7 and section 4).
5. Validation: the last 200 rows of each domain, never trained on, measured at step 0 and after every epoch, monitoring
   only. KL math = KL(math teacher || student) per token on the math teacher's answers; CE math = the student's CE per
   token on the ground truth GSM8K solutions; the same on code. Both are teacher forced.
6. Benchmarks: the full test sets, greedy, pass@1. GSM8K is right if the last `\boxed{}` (or the last number if nothing
   is boxed) equals the ground truth number. MBPP / HumanEval: evalplus runs the code; + = also passes evalplus's
   extra tests.
7. Seeds change only the data order. The pretrained start, the val rows and the benchmarks are the same for all seeds.

## 2. Why the numbers can be trusted

1. Our baselines match Qwen's reported numbers within 2.5 pp (student GSM8K 49.4 vs 49.6, HumanEval 37.8 vs 35.4; code
   teacher MBPP+ 70.6 vs 71.7; math teacher GSM8K 95.5 vs 95.2).
2. The RoPE bug (23 Oct 20:07 UTC entry): the first runs benchmarked the student after `student.to(bf16)`, which also
   rounds RoPE's `inv_freq` buffer (phase error up to 0.94 rad at position 1000), and stage 2 trained with it. Found
   because a continuation from the saved stage 1 checkpoint differed from the uninterrupted run at step 0. Fixed: every
   benchmark and every next stage now loads the saved checkpoint. All runs were redone; entries marked INVALID are
   history only.
3. Stage 2 starts from stage 1: its step-0 val equals stage 1's final val within 0.0011 in every run.
4. All 18 runs exited 0 with no tracebacks, every checkpoint is on the hub (sha256 checked), and the local copy of the
   result files is checksum-identical to the server's (238 files).
5. GPU training is not bit-reproducible (4th decimal after step 2), so two runs of the same setting differ a little.
   The seeds measure how much (section 4).

## 3. Findings (3 seeds)

1. **The ground truth CE lowers GSM8K, in proportion to its weight.** Math stage, both orders, all seeds:

   | math stage                | ALPHA 1.0 (KD only)            | ALPHA 0.5                      | ALPHA 0.0 (ground truth SFT)   |
   |---------------------------|--------------------------------|--------------------------------|--------------------------------|
   | from base (math -> code)  | 672, 688, 686 (mean +2.4 pp)   | 560, 566, 549 (mean -7.0 pp)   | 477, 477, 491 (mean -12.8 pp)  |
   | after code (code -> math) | 666, 671, 669 (mean +1.3 pp)   | 544, 561, 511 (mean -8.5 pp)   | 471, 481, 475 (mean -13.3 pp)  |

   The mechanism: the student copies the short ground truth style ("The answer is \boxed{N}", median 84-108 tokens vs
   the base's 294, seed 0). Ground truth style GSM8K answers (of 1319), math stage from base: 0 / 0 / 0 at ALPHA 1.0, 933 / 905 /
   916 at 0.5, 1312 / 1312 / 1313 at 0.0. At ALPHA 0.5 the ground truth style answers are 38.0-41.8% right in every
   seed, far below the base's 49.4%.
2. **KD only is the only setting above base on GSM8K**: after all 6 of its math stages, 666 to 688 questions (+1.1 to
   +2.8 pp). A real but small gain, given the teacher is at 95.5%.
3. **The val losses explain the GSM8K ranking** (math stage from base, 3-seed means):

   | setting          | KL math (to teacher) | CE math (to ground truth) | GSM8K |
   |------------------|----------------------|---------------------------|-------|
   | base             | 0.1706               | 0.8387                    | 651   |
   | KD only          | 0.1179               | 0.8849                    | 682   |
   | ALPHA 0.5        | 0.1336               | 0.5584                    | 558   |
   | ground truth SFT | 0.3151               | 0.5599                    | 482   |

   GSM8K falls when the CE to the ground truth falls (copying its style); at equal CE, staying close to the teacher
   (ALPHA 0.5 vs SFT) keeps 77 more questions.
4. **Code distillation does not measurably improve MBPP.** After every code stage MBPP is 178 to 193 (-5 to +10
   questions vs base; best mean +1.5 pp), within the seed noise of 13 questions, although the code teacher scores
   82.0% MBPP. Val KL code does fall (0.2212 -> 0.15 under KD only), so the student does match the teacher on KodCode.
5. **HumanEval never ends above base** (the best stage equals it once: 62). The big drops come from math stages with
   ground truth CE: HumanEval 49-52 (-10 to -13 questions, -6.1 to -7.9 pp) in all 12 such runs, both orders. KD only
   math stage: -3 to -5 from base; after code 52, 52, 59. MBPP drops less after a math stage with ground truth CE
   from base (-2.4 to -4.8 pp). Only the ALPHA 0.5 code stage brings HumanEval back (50 -> 56, 50 -> 60, 51 -> 58); the SFT code stage (+0, +3,
   -1) and the KD only code stage (-1, -2, -3) do not. Not analysed yet.
6. **Order, ALPHA 0.5: math -> code ends ahead of code -> math on all five benchmarks in all 3 seeds** (seed 0: GSM8K
   +17, MBPP +8, MBPP+ +5, HumanEval +7, HE+ +7 questions; seed 1: +10, +13, +6, +8, +11; seed 2: +67, +11, +6, +8, +10;
   means +2.4, +2.8, +1.5, +4.7, +5.7 pp). Ending on math costs HumanEval (finding 5) and GSM8K (finding 1); the code
   stage after math costs no GSM8K (560 -> 561, 566 -> 571, 549 -> 578). KD only: ending on math gives more GSM8K (+20,
   +34, +42 questions); the code side is within noise. Ground truth SFT: no consistent order on GSM8K.
7. **The ground truth CE overfits in epoch 2, the KL does not.** Math val CE rises from epoch 1 to 2 in all 12 runs with
   ground truth CE (+0.016 to +0.023, e.g. SFT 0.5389 -> 0.5619). Math val KL under KD only falls in all 6 (e.g. 0.1218
   -> 0.1177). Code val CE barely moves (within 0.005 either way). Only the epoch 2 model is benchmarked.
8. **A code stage from base removes `\boxed{}` from GSM8K answers, more with more ground truth CE**: 176 / 179 / 217
   answers without it at ALPHA 1.0, 307 / 329 / 437 at 0.5, 428 / 400 / 493 at 0.0 (base 30). After a math stage the
   habit holds (at most 24, except the 67 of finding 9). The scorer then takes the last number, so GSM8K drops less than this suggests.
9. **Ground truth SFT on both domains makes the student answer math with code.** After SFT math -> SFT code, 3 / 6 / 58
   GSM8K answers are a bare ```` ```python ```` block with no final number, all scored wrong. 0 in all 12 runs with the
   KL term, 1 after the SFT code stage from base. Fits the data: the code ground truth is only a code block (setup 3),
   the teacher's code answers mix text and code. In seed 2 this is half of the stage's 52-question GSM8K drop; the
   other half is lower accuracy on the remaining questions (465 -> 439).

## 4. Seed noise: which claims survive

Largest gap between the 3 seeds of one setting and stage: GSM8K 53 questions (4.0 pp), MBPP 13 (3.4 pp), MBPP+ 13
(3.4 pp), HumanEval 7 (4.3 pp), HE+ 8 (4.9 pp). The val losses do not show these gaps: code -> math ALPHA 0.5 seed 2
ends at GSM8K 511 vs 544 / 561 with the same val losses (math KL / CE 0.1341 / 0.5579 vs 0.1337 / 0.5546 and 0.1346 /
0.5582), the same answer style mix and lower accuracy in both styles.

Survive all 3 seeds: findings 1, 2, 3, 5 (math stages with ground truth CE), 6 (ALPHA 0.5), 7, 8, 9.

Did not survive, do not present:
- "the KD only code stage keeps HumanEval at base": 62, 56, 56.
- "the code stage after SFT recovers GSM8K": +15, -7, -52.
- "KD only improves MBPP in the code stage": +2, -3, -5.
- "with ground truth SFT, math -> code ends ahead on GSM8K": 492 vs 471, 470 vs 481, 439 vs 475.
- "every math stage after a code stage drops HumanEval": 8 of 9 runs (KD only seed 2: 56 -> 59).

## 5. Benchmark quirks (for questions on the baselines)

1. The math teacher's MBPP (46.8%) is below the student's (48.4%): 113 of its 201 failures rename the function the tests
   call (e.g. `find_Rotations` -> `find_rotations`). A real failure under evalplus's rules, so the score stands.
2. evalplus 0.3.1's `sanitize` drops decorated top-level functions (`@lru_cache def fib`), scored as failing. Same for
   every model.
3. Greedy decoding with no repetition penalty loops sometimes: 16 base student GSM8K answers hit the 1600 limit, all wrong.
4. The "memory allocation failed with OOM" warnings in the logs are harmless: PyTorch frees its cache and retries.

## 6. Open questions for the analysis

1. Why does HumanEval drop after math stages more than MBPP, and why does only the ALPHA 0.5 code stage bring it back?
2. Why does a strong code teacher (82.0% MBPP) give no measurable MBPP gain?
3. Why is the KD only GSM8K gain small (+1 to +3 pp) next to a 95.5% teacher?
4. Would the epoch 1 model be better for the settings with ground truth CE (finding 7)? Not measurable now, epoch 1 is
   not saved.
5. Why does the KL term stop the code-format leak of finding 9, and why does the seed change it from 3 to 58 answers?
6. Does distilling the 297 wrong math answers matter? (the dropped filtered-answers ablation)
