### Reading list

0. Video lectures

- [x] [CS336 Post-training lecture](https://www.youtube.com/watch?v=2oH6PWPrYFo)
- [ ] [UCLA RL for LLMs](https://www.youtube.com/playlist?list=PLir0BWtR5vRp5dqaouyMU-oTSzaU5LK9r)
- [ ] [Post-training course](https://www.youtube.com/playlist?list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y)

1. Knowledge Distillation

- [x] [Distilling the Knowledge in a Neural Network]( https://arxiv.org/abs/1503.02531)
- [x] [Sequence-Level Knowledge Distillation]( https://arxiv.org/abs/1606.07947)
- [ ] [On the Efficacy of Knowledge Distillation]( https://arxiv.org/abs/1910.01348)
- [ ] [MiniLLM]( https://arxiv.org/abs/2306.08543)
- [ ] [Knowledge Fusion of Large Language Models]( https://arxiv.org/abs/2401.10491)
- [ ] [On-Policy Distillation of Language Models]( https://arxiv.org/abs/2306.13649)

2. RL

- [ ] []()
- [ ] []()
- [ ] []()
- [ ] []()

3. Post training

- [ ] [SFT Memorizes, RL Generalizes]( https://arxiv.org/abs/2501.17161)
- [ ] [RL's Razor]( https://arxiv.org/abs/2509.04259)
- [ ] [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?]( https://arxiv.org/abs/2504.13837)
- [ ] [Spurious Rewards: Rethinking Training Signals in RLVR]( https://arxiv.org/abs/2506.10947)

### Results

Benchmarks are greedy decoding and pass@1
Validation is on the last 200 rows of each domain's training data (GSM8K train, KodCode).

Training losses for KD experiments:

| setting              | loss                               | how                                      |
|----------------------|------------------------------------|------------------------------------------|
| KD + ground truth CE | (ALPHA * KL) + (1 - ALPHA) * CE    | `ALPHA = 0.5`                            |
| ground truth SFT     | CE on the ground truth             | `ALPHA = 0.0`                            |
| KD only              | KL b/w the teacher and the student | `ALPHA = 1.0`                            |
| teacher SFT          | CE on the teacher's answers        | `CE_TARGET = "response"` (not added yet) |

#### 0. Teacher answers (`0_data_gen/gen_data.py`)

cut = answers that hit MAX_OUTPUT_LENGTH and were dropped. so we put a reasonable MAX token limit for the LLM response.

cut answers are completely dropped from the training set. the student does not see them at all.

| domain             | teacher                   | prompts | kept  | cut   |
|--------------------|---------------------------|---------|-------|-------|
| math (GSM8K train) | Qwen2.5-Math-7B-Instruct  | 7473    | 7441  | 32    |
| code (KodCode)     | Qwen2.5-Coder-7B-Instruct | 7473    | 7469  | 4     |

#### 1. Baselines (`1_baseline_scripts/`)

These are the results on the benchmark before any training or fine-tuning. This acts as the baseline score which we are trying to improve.
If an answer was cut, then that is counted as wrong. GSM8K test has 1319, MBPP+ 378 and HumanEval+ 164. Nothing is dropped there.

| model                                   | GSM8K | MBPP  | MBPP+ | HE    | HE+   |
|-----------------------------------------|-------|-------|-------|-------|-------|
| math teacher, Qwen2.5-Math-7B-Instruct  | 95.5  | 46.8  | 40.2  | 61.6  | 54.9  |
| code teacher, Qwen2.5-Coder-7B-Instruct | 88.5  | 82.0  | 70.6  | 90.9  | 85.4  |
| student, Qwen2.5-0.5B-Instruct          | 49.4  | 48.4  | 41.3  | 37.8  | 34.1  |

Reported by Qwen, to sanity-check ours:

| model                     | GSM8K | MBPP  | MBPP+ | HE    | HE+   |
|---------------------------|-------|-------|-------|-------|-------|
| Qwen2.5-Math-7B-Instruct  | 95.2  | -     | -     | -     | -     |
| Qwen2.5-Coder-7B-Instruct | -     | 83.5  | 71.7  | 88.4  | 84.1  |
| Qwen2.5-0.5B-Instruct     | 49.6  | 49.6  | -     | 35.4  | -     |

#### 2. Sequential KD (`2_sequential_kd/`)

The training setup was we train our student in 2 ways:
1. distill using math model first (stage 1) -> then distill using code model (stage 2)
2. distill using code model first (stage 1) -> then distill using math model (stage 2)

For each setup we have 3 different types of losses as mentioned before:
1. KL on teacher + CE on ground truth
2. KL on teacher only
3. CE on ground truth only

This helps us isolate which factor is helping the student the most.

We benchmark our student after each stage. We report validation loss on both domains, so we can find if any domain shows forgetting. However, we do not use early stopping. Our model trains for 2 epochs in every case. Validation loss is purely for monitoring how the model is performing after every step.

The tables below report the following metrics:

1. GSM8K shows the accuracy on GSM8K test set. 
2. MBPP shows the accuracy on MBPP benchmark. 
3. MBPP+ shows the accuracy on MBPP+ benchmark. 
4. HE shows the accuracy on HE benchmark. 
5. HE+ shows the accuracy on HE+ benchmark. 

6. KL math shows the validation loss . 
7. CE math shows the validation loss . 
8. KL code shows the validation loss . 
9. CE code shows the validation loss . 

---

Every number is the mean over seeds 0, 1 and 2 (the seed only changes the training data order). Benchmarks are in %,
with the difference to the student with no training in percentage points (pp) in brackets. Val losses are after the
stage's last epoch. Questions right per seed are in LOG.md, the result files of every run are in
`2_sequential_kd/runs/`, and `2_sequential_kd/results_table.py` prints both tables from them.

1. MATH -> CODE 

| setting              | stage                   | GSM8K        | MBPP        | MBPP+       | HE          | HE+         | KL math | CE math | KL code | CE code |
|----------------------|------------------------ |--------------|-------------|-------------|-------------|-------------|---------|---------|---------|---------|
| student, no training | -                       | 49.4         | 48.4        | 41.3        | 37.8        | 34.1        | 0.1706  | 0.8387  | 0.2212  | 0.3674  |
| KD + ground truth CE | stage 1 distilling math | 42.3 (-7.0)  | 45.2 (-3.2) | 39.1 (-2.2) | 30.7 (-7.1) | 28.3 (-5.9) | 0.1336  | 0.5584  | 0.2446  | 0.3987  |
| KD + ground truth CE | stage 2 distilling code | 43.2 (-6.1)  | 49.9 (+1.5) | 42.5 (+1.2) | 35.4 (-2.4) | 32.9 (-1.2) | 0.1524  | 0.5717  | 0.1634  | 0.2934  |
| ground truth SFT     | stage 1 distilling math | 36.5 (-12.8) | 44.4 (-4.0) | 37.4 (-3.9) | 30.5 (-7.3) | 27.8 (-6.3) | 0.3151  | 0.5599  | 0.2596  | 0.4120  |
| ground truth SFT     | stage 2 distilling code | 35.4 (-13.9) | 48.9 (+0.5) | 42.8 (+1.5) | 30.9 (-6.9) | 27.6 (-6.5) | 0.2778  | 0.5992  | 0.4168  | 0.3011  |
| KD only              | stage 1 distilling math | 51.7 (+2.4)  | 49.6 (+1.2) | 42.6 (+1.3) | 35.4 (-2.4) | 29.5 (-4.7) | 0.1179  | 0.8849  | 0.2455  | 0.3858  |
| KD only              | stage 2 distilling code | 48.3 (-1.1)  | 49.1 (+0.7) | 42.0 (+0.7) | 34.1 (-3.7) | 30.1 (-4.1) | 0.1396  | 0.8615  | 0.1525  | 0.3672  |
| teacher SFT          | stage 1 distilling math |              |             |             |             |             |         |         |         |         |
| teacher SFT          | stage 2 distilling code |              |             |             |             |             |         |         |         |         |

2. CODE -> MATH

| setting              | stage                   | GSM8K        | MBPP        | MBPP+       | HE          | HE+         | KL math | CE math | KL code | CE code |
|----------------------|------------------------ |--------------|-------------|-------------|-------------|-------------|---------|---------|---------|---------|
| student, no training | -                       | 49.4         | 48.4        | 41.3        | 37.8        | 34.1        | 0.1706  | 0.8387  | 0.2212  | 0.3674  |
| KD + ground truth CE | stage 1 distilling code | 47.0 (-2.3)  | 48.8 (+0.4) | 42.0 (+0.7) | 35.0 (-2.8) | 32.1 (-2.0) | 0.1838  | 0.7808  | 0.1621  | 0.2928  |
| KD + ground truth CE | stage 2 distilling math | 40.8 (-8.5)  | 47.1 (-1.3) | 41.0 (-0.3) | 30.7 (-7.1) | 27.2 (-6.9) | 0.1341  | 0.5569  | 0.1830  | 0.3079  |
| ground truth SFT     | stage 1 distilling code | 47.8 (-1.6)  | 48.6 (+0.2) | 42.9 (+1.7) | 35.2 (-2.6) | 32.1 (-2.0) | 0.1967  | 0.8755  | 0.3860  | 0.3004  |
| ground truth SFT     | stage 2 distilling math | 36.1 (-13.3) | 47.2 (-1.2) | 41.4 (+0.2) | 30.7 (-7.1) | 28.7 (-5.5) | 0.3428  | 0.5564  | 0.3344  | 0.3122  |
| KD only              | stage 1 distilling code | 46.1 (-3.2)  | 49.1 (+0.7) | 42.6 (+1.3) | 35.4 (-2.4) | 31.3 (-2.8) | 0.1818  | 0.7562  | 0.1517  | 0.3672  |
| KD only              | stage 2 distilling math | 50.7 (+1.3)  | 48.1 (-0.4) | 42.2 (+0.9) | 33.1 (-4.7) | 29.5 (-4.7) | 0.1187  | 0.8782  | 0.1717  | 0.3749  |
| teacher SFT          | stage 1 distilling code |              |             |             |             |             |         |         |         |         |
| teacher SFT          | stage 2 distilling math |              |             |             |             |             |         |         |         |         |

#### 3. Joint KD (`3_joint_kd/kd_train.py`, ablations `abl_sft_joint.py` and `abl_kl_only_joint.py`)

Distilling both teachers simultaneously, val loss measured on both domains.

| setting              | GSM8K | MBPP  | MBPP+ | HE    | HE+   | KL math | CE math | KL code | CE code | time (h) |
|----------------------|-------|-------|-------|-------|-------|---------|---------|---------|---------|----------|
| KD + ground truth CE |       |       |       |       |       |         |         |         |         |          |
| ground truth SFT     |       |       |       |       |       |         |         |         |         |          |
| KD only              |       |       |       |       |       |         |         |         |         |          |
| teacher SFT          |       |       |       |       |       |         |         |         |         |          |


### TODO

Ablations
- ground truth CE: ALPHA = 1.0 (KL only) and 0.0 (ground truth SFT only) vs 0.5
  - [x] sequential, math -> code
  - [x] sequential, code -> math
  - [ ] joint
- [ ] temperature: TEMPERATURE = 2.0 vs 1.0 on one setting (after the 26 Oct checkpoint)
- [ ] teacher SFT: CE on the teacher's answers instead of the ground truth 
- [ ] filtered teacher answers: train only on math answers with the right final number (4.4% are wrong now) and code
      answers that pass KodCode's tests, vs the unfiltered data
- seeds: rerun every setting with SEED = 1 and 2, report the mean of the 3 seeds
  - [x] sequential
  - [ ] joint

Analysis
- [ ] failure analysis: cut answers, wrong answers, greedy loops, the math teacher's renamed MBPP functions

Post-training (section 4)
- [ ] SFT on the teacher's answers, GRPO, on-policy distillation
