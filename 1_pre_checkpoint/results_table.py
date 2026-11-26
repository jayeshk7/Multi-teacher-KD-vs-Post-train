# tables of the sequential runs in runs/ (copied from the server): mean over seeds 0, 1, 2 as README rows (% with pp vs
# the base student, val losses after each stage), then every seed as questions right, % (pp). run from 1_pre_checkpoint/
import glob
import json
import re

# base student, questions right / benchmark size (bench.py)
BASE = {"gsm8k": 651 / 1319, "mbpp": 183 / 378, "mbpp_plus": 156 / 378, "humaneval": 62 / 164, "humaneval_plus": 56 / 164}
BENCH = [("gsm8k", "n_gsm8k"), ("mbpp", "n_mbpp"), ("mbpp_plus", "n_mbpp"), ("humaneval", "n_humaneval"),
         ("humaneval_plus", "n_humaneval")]
# out dir, script name (names the seed 1 / 2 logs), README setting
RUNS = [("seq_math_code", "seq_math_then_code", "KD + ground truth CE"),
        ("abl_sft_math_code", "abl_sft_math_code", "ground truth SFT"),
        ("abl_kl_only_math_code", "abl_kl_only_math_code", "KD only"),
        ("seq_code_math", "seq_code_then_math", "KD + ground truth CE"),
        ("abl_sft_code_math", "abl_sft_code_math", "ground truth SFT"),
        ("abl_kl_only_code_math", "abl_kl_only_code_math", "KD only")]
SEEDS = (0, 1, 2)


def final_val(log, stage):
    # 3 val points per stage (step 0, epoch 1, epoch 2), math then code. returns kl / ce math, kl / ce code after epoch 2
    vals = re.findall(r"(math|code) val at step \d+: kl ([\d.]+)  ce ([\d.]+)", open(log).read())
    end = vals[6 * stage - 2: 6 * stage]
    return [float(end[0][1]), float(end[0][2]), float(end[1][1]), float(end[1][2])]


means, per_seed = [], []
for out, script, setting in RUNS:
    for stage in (1, 2):
        sums, vals = [], []
        for seed in SEEDS:
            path = glob.glob(f"runs/{out}{f'_seed{seed}' if seed else ''}/results_stage{stage}_*/summary.json")[0]
            dom = path.split("_")[-1].split("/")[0]
            s = json.load(open(path))
            sums.append(s)
            vals.append(final_val(f"runs/{script}_seed{seed}.log" if seed else f"runs/{out}.log", stage))
            cells = [f"{round(s[k] * s[n])}, {100 * s[k]:.1f} ({100 * (s[k] - BASE[k]):+.1f})" for k, n in BENCH]
            per_seed.append(f"| {setting} | {out} | {seed} | {stage} {dom} | " + " | ".join(cells) + " |")
        cells = []
        for k, _ in BENCH:
            m = sum(s[k] for s in sums) / len(sums)
            cells.append(f"{100 * m:.1f} ({100 * (m - BASE[k]):+.1f})")
        cells += [f"{sum(v[i] for v in vals) / len(vals):.4f}" for i in range(4)]
        means.append(f"| {setting:20} | stage {stage} distilling {dom} | " + " | ".join(cells) + " |")

print("mean over seeds 0, 1, 2 (README rows): % (pp vs base), val KL / CE math, KL / CE code\n")
print("\n".join(means))
print("\nper seed: questions right, % (pp vs base)\n")
print("| setting | run | seed | stage | GSM8K | MBPP | MBPP+ | HE | HE+ |\n|---|---|---|---|---|---|---|---|---|")
print("\n".join(per_seed))
