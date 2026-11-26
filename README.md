# Multi-teacher distillation vs distill-then-post-train

How should a small model learn math and code from two specialist teachers? We teach **Qwen2.5-0.5B-Instruct** from
**Qwen2.5-Math-7B-Instruct** and **Qwen2.5-Coder-7B-Instruct** and compare two routes:

- **Route 1, multi-teacher KD:** distill both teachers in one run, each prompt from the teacher of its own domain.
- **Route 2, KD then GRPO:** distill one domain, then train the other with GRPO, which needs only answer checks or
  unit tests, no teacher.

We also ask whether a GRPO second stage forgets less than a KD second stage, and whether the training text or the loss
matters more (cross-entropy on ground truth solutions, cross-entropy on teacher answers, KD on teacher answers). Every
method is trained with LoRA and 3 seeds.

CS725 course project, IIT Bombay: Jayesh Kandpal, Mangesh Poojan, Shivang Goswami, Sancheet Basarikatti.
Report: [`docs/report.pdf`](docs/report.pdf). Slides: [`docs/slides.pdf`](docs/slides.pdf).

## Results

Change against the instruct model in percentage points (pp), 3-seed means, all benchmarked on an RTX 6000 Ada.

![GSM8K and held-out KodCode for every recipe](figures/res_overview.png)

Route 2 wins on the domain it trains with GRPO. Multi GRPO, with no teacher, matches the better route 2 order in each
domain.

![KD vs GRPO as the code stage after KD math](figures/res_forget.png)

Starting from the same KD math checkpoint, a GRPO code stage loses as much math as a KD code stage but gains more code.

![GT-SFT vs T-SFT vs KD](figures/res_text_loss.png)

The training text matters more than the loss: ground truth solutions cost 14 to 15 pp of math, while KD scores about
the same as SFT on the teacher's answers.

![multi KD vs multi GRPO on all five benchmarks](figures/res_transfer_bars.png)

The gains stay on the two tests from the training distribution. More than half of each held-out code gain is a format
change (the models stop writing extra code after the function, which our tester runs and which can fail every test);
with that code deleted, multi GRPO's gain is +8.5 pp instead of +19.3. Details are in the report.

## How to reproduce

Python 3.12, torch 2.13.0, transformers 5.20.2, trl 1.17.1, peft 0.21.2, datasets 5.1.0, math-verify 0.9.0, evalplus
0.3.1, pytest; 48 GB GPUs. The teacher answers are downloaded from the public Hugging Face dataset
`jajamken/FML_project` (made by `0_data_gen/gen_data.py`).

```
0_data_gen/gen_data.py     teacher answers for the training prompts
1_pre_checkpoint/          before the mid-project checkpoint: full fine-tuning (sequential KD results in runs/)
4_post_train/              first GRPO test run on code
5_recipes/                 the final experiments: train_sft_kd.py, train_grpo.py, eval_model.py, common.py, reward.py
  analysis/                error analysis, held-out re-scoring, LoRA vs full fine-tuning, charts
  check/                   GPU-type checks
  results/                 logs and scores of every run
```

```bash
cd 5_recipes
bash run.sh multi_sft 0          # multi GT-SFT and multi T-SFT, seeds 0 1 2 (SEEDS=0 for one seed)
bash run.sh multi_kd 0,1         # student on GPU 0, both teachers on GPU 1
bash run.sh kd_math_chain 0      # KD math, then KD math -> KD code (also kd_code_chain)
bash run.sh rl_code 0            # KD math -> GRPO code (also rl_math, multi_rl; refs = instruct and teachers)
MODEL=<run>/best OUT=<folder> python -u eval_model.py    # benchmark one checkpoint again

cd ../1_pre_checkpoint           # October runs
ORDER=math,code ALPHA=0.5 SEED=0 python -u seq_kd.py
ALPHA=0.5 SEED=0 python -u multi_kd.py
```

Each run writes `log.json`, the best adapter in `best/` and the test results in `results_best/`.

- All reported numbers come from an RTX 6000 Ada. Greedy answers in bf16 depend on the GPU type, so on other GPUs
  expect differences of about the GPU noise (up to 1.7 pp on GSM8K).
- The trained LoRA adapters are not published, so the runs have to be trained again (about 80 GPU-hours of training for
  all 31 runs, plus benchmarking).
- The analysis scripts read the saved answer text (about 400 MB, not in git). Run the benchmarks first, then point
  `DATA` in each script (or the folder it runs from) to the results.
- The October scripts and `gen_data.py` upload to our Hugging Face repo. Set `HUB_REPO`, `HUB_RESULTS` and
  `HF_HUB_DATA` to `""` to keep everything local.
- The October multi-KD results (the `joint_*` folders in the Hugging Face model repo `jajamken/FML_project`) were
  produced by a modified copy of `multi_kd.py` that is not in this repo: it adds the teacher-SFT setting and MATH-500,
  and ran 904 steps instead of 906. `multi_kd.py` runs the same recipe.

## References and resources

Models: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
[Qwen2.5-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct),
[Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct).
Data: [GSM8K](https://huggingface.co/datasets/openai/gsm8k),
[KodCode-Light-RL-10K](https://huggingface.co/datasets/KodCode/KodCode-Light-RL-10K),
[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), [EvalPlus](https://github.com/evalplus/evalplus)
(MBPP+, HumanEval+).
Libraries: [TRL](https://github.com/huggingface/trl), [PEFT](https://github.com/huggingface/peft),
[Math-Verify](https://github.com/huggingface/Math-Verify).

Papers cited in the report:

- Hinton, Vinyals, Dean. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). 2015
- Kim, Rush. [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947). 2016
- Shao et al. [DeepSeekMath](https://arxiv.org/abs/2402.03300) (GRPO). 2024
- Shenfeld, Pari, Agrawal. [RL's Razor: Why Online Reinforcement Learning Forgets Less](https://arxiv.org/abs/2509.04259). 2025
- Hu et al. [LoRA](https://arxiv.org/abs/2106.09685). 2022
- Qwen. [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115). 2025
- Yang et al. [Qwen2.5-Math Technical Report](https://arxiv.org/abs/2409.12122). 2024
- Hui et al. [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186). 2024
- Xu et al. [KodCode](https://arxiv.org/abs/2503.02951). 2025
- Cobbe et al. [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) (GSM8K). 2021
- Hendrycks et al. [Measuring Mathematical Problem Solving with the MATH Dataset](https://arxiv.org/abs/2103.03874). 2021
- Lightman et al. [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (MATH-500). 2024
- Austin et al. [Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732) (MBPP). 2021
- Chen et al. [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (HumanEval). 2021
- Liu et al. [Is Your Code Generated by ChatGPT Really Correct?](https://arxiv.org/abs/2305.01210) (EvalPlus). 2023

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

2. Post training

- [ ] [SFT Memorizes, RL Generalizes]( https://arxiv.org/abs/2501.17161)
- [ ] [RL's Razor]( https://arxiv.org/abs/2509.04259)
- [ ] [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?]( https://arxiv.org/abs/2504.13837)
- [ ] [Spurious Rewards: Rethinking Training Signals in RLVR]( https://arxiv.org/abs/2506.10947)
