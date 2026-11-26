# charts for the slides and the readme, from the result files in git.
# run from 5_recipes/analysis after lora_vs_full.py: python deck2_charts.py <out_dir>
import json

import numpy as np
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = sys.argv[1]
BG, INK, INK2, GRID = "#FAFAF7", "#1A1A1A", "#555555", "#E4E4E0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 20, "axes.facecolor": BG, "figure.facecolor": BG,
                     "savefig.facecolor": BG, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": False, "legend.frameon": False, "axes.unicode_minus": True})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def pp(x):
    return f"{x:+.1f}".replace("-", "−")


# 1. pre-checkpoint, after the whole run, 3-seed means, pp vs instruct (the readme's october tables):
# sequential kd in both orders (after both stages), and multi (joint) kd
BENCH3 = ["GSM8K", "MBPP+", "HumanEval+"]
SEQ = {"math \u2192 code": [(-1.1, 0.7, -4.1), (-6.1, 1.2, -1.2), (-13.9, 1.5, -6.5)],
       "code \u2192 math": [(1.3, 0.9, -4.7), (-8.5, -0.3, -6.9), (-13.3, 0.2, -5.5)]}
MULTI = [(0.8, 0.4, -4.1), (-7.5, 1.7, -2.6), (-13.3, 1.7, -3.7), (1.5, 0.5, -5.3)]


def bars(ax, xs, vals, col, w, fs):
    for x, v in zip(xs, vals):
        ax.bar(x, v, width=w, color=col)
        ax.text(x, v + (0.4 if v >= 0 else -0.4), pp(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=fs)


def panels(labels):
    fig, axs = plt.subplots(1, 3, figsize=(16.64, 5.6), sharey=True)
    fig.subplots_adjust(wspace=0.12)
    for ax, title in zip(axs, BENCH3):
        ax.axhline(0, color=INK2, lw=1.2)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=18)
        ax.tick_params(axis="x", length=0)
        ax.set_title(title, fontsize=22, pad=14)
        ax.set_ylim(-16.5, 4)
        ax.set_yticks([-15, -10, -5, 0])
    axs[0].set_ylabel("change vs instruct (pp)")
    return fig, axs


fig, axs = panels(["KD only", "KD +\nGT CE", "GT-SFT"])
for j, ax in enumerate(axs):
    for k, (name, rows) in enumerate(SEQ.items()):
        bars(ax, [i + (k - 0.5) * 0.47 for i in range(3)], [r[j] for r in rows], [BLUE, ORANGE][k], 0.45, 15)
axs[1].legend(handles=[Patch(color=c, label=n) for n, c in zip(SEQ, [BLUE, ORANGE])], loc="lower center", fontsize=19)
save(fig, "mid_seq")

fig, axs = panels(["KD only", "KD +\nGT CE", "GT-SFT", "teacher\nSFT"])
for j, ax in enumerate(axs):
    bars(ax, range(4), [r[j] for r in MULTI], BLUE, 0.6, 19)
save(fig, "mid_multi")

# 2. lora vs full fine-tuning, matched recipes, last checkpoints: paired bars per recipe (3-seed means, pp vs instruct)
d = json.load(open("lora_vs_full.json"))["benchmarks_pp"]
names = list(d)[::-1]
GREY = "#8C8C85"
fig, axs = plt.subplots(1, 3, figsize=(16.64, 6.6), sharey=True)
fig.subplots_adjust(wspace=0.08)
h = 0.38
for ax, (k, title) in zip(axs, [("gsm8k", "GSM8K"), ("mbpp_plus", "MBPP+"), ("humaneval_plus", "HumanEval+")]):
    ax.axvline(0, color=INK2, lw=1.2)
    for y, n in enumerate(names):
        for off, v, col in [(h / 2, d[n]["full_ft_mean"][k], GREY), (-h / 2, d[n]["lora_last_mean"][k], BLUE)]:
            ax.barh(y + off, v, height=h * 0.94, color=col)
            ax.text(v + (0.3 if v >= 0 else -0.3), y + off, pp(v), ha="left" if v >= 0 else "right", va="center", fontsize=14)
    ax.set_xlim(-18, 6)
    ax.set_xticks([-15, -10, -5, 0, 5])
    ax.set_title(title, fontsize=22, pad=14)
    ax.set_xlabel("change vs instruct (pp)", fontsize=17)
    ax.tick_params(axis="y", length=0)
axs[0].set_yticks(range(len(names)))
axs[0].set_yticklabels(names, fontsize=17, color=INK)
fig.legend(handles=[Patch(color=GREY, label="full fine-tuning (October, all weights)"),
                    Patch(color=BLUE, label="LoRA (November, rank 16)")],
           loc="lower center", ncol=2, fontsize=18, bbox_to_anchor=(0.55, -0.1))
save(fig, "lora_bars")

# 3. convergence: kd math and kd code from instruct, lora vs full fine-tuning. validation kl to the teacher (solid,
# 3-seed mean) and training kl (dashed, 3-seed mean, moving average over about 30 steps); same val rows, same measure
import re

PRE = "../../1_pre_checkpoint/runs"
POST = "../results/runs"


def movavg(xs, ys, w):
    k = np.ones(w) / w
    return np.convolve(xs, k, "valid"), np.convolve(ys, k, "valid")


fig, axs = plt.subplots(1, 2, figsize=(16.64, 5.4), sharey=True)
fig.subplots_adjust(wspace=0.08)
for ax, (dom, run, pre, title) in zip(axs, [("math", "kd_math", "abl_kl_only_math_code", "KD on math: KL to the math teacher"),
                                            ("code", "kd_code", "abl_kl_only_code_math", "KD on code: KL to the code teacher")]):
    logs = [json.load(open(f"{POST}/inst_{run}_s{sd}/log.json")) for sd in range(3)]
    vs = [x["step"] for x in logs[0]["val"]]
    vv = np.mean([[x[f"val_loss_{dom}"] for x in l["val"]] for l in logs], axis=0)
    tr = np.mean([[x["loss"] for x in l["train"]] for l in logs], axis=0)
    tx, ty = movavg(np.arange(1, len(tr) + 1), tr, 30)
    ax.plot(tx, ty, color=BLUE, lw=1.6, ls="--")
    ax.plot(vs, vv, color=BLUE, lw=2.6, marker="o", markersize=8)
    pv, pt = [], []
    for sfx in ["", "_seed1", "_seed2"]:
        first = open(f"{PRE}/{pre}{sfx}.log").read().split("{'model'")[0]
        pv.append([(int(s), float(k)) for s, k in re.findall(rf"{dom} val at step (\d+): kl ([\d.]+)", first)])
        pt.append([(int(s), float(k)) for s, k in re.findall(rf"{dom} step (\d+)/\d+\s+kl ([\d.]+)", first)])
    px, py = movavg(np.array([s for s, _ in pt[0]]), np.mean([[k for _, k in p] for p in pt], axis=0), 3)
    ax.plot(px, py, color=GREY, lw=1.6, ls="--")
    ax.plot([s for s, _ in pv[0]], np.mean([[k for _, k in p] for p in pv], axis=0), color=GREY, lw=2.6, marker="s", markersize=8)
    ax.axvline(vs[len(vs) // 2], color=INK2, lw=1, ls=":")
    ax.text(vs[len(vs) // 2] + 6, 0.235, "epoch 2", fontsize=16, color=INK2)
    ax.set_title(title, fontsize=21, pad=12)
    ax.set_xlabel("training step", fontsize=18)
    ax.set_ylim(0, 0.25)
    ax.set_xlim(0, 460)
axs[0].set_ylabel("KL per token")
axs[1].legend(handles=[Line2D([], [], color=BLUE, lw=2.6, marker="o", label="LoRA, validation"),
                       Line2D([], [], color=GREY, lw=2.6, marker="s", label="full FT, validation"),
                       Line2D([], [], color=BLUE, lw=1.6, ls="--", label="LoRA, training"),
                       Line2D([], [], color=GREY, lw=1.6, ls="--", label="full FT, training")],
              loc="lower left", ncol=2, fontsize=15, columnspacing=1.2)
save(fig, "converge")

# 4. results (post-checkpoint): best checkpoints benchmarked on the rtx 6000 ada (own eval, else the ada re-eval),
# change vs instruct on the ada in pp, 3-seed means. question benchmarks from exact counts, held-out = mean test pass rate
RES = "../results"


def ada(run):
    s = json.load(open(f"{RES}/runs/{run}/summary_best.json"))
    return s if "Ada" in s.get("gpu", "") else json.load(open(f"{RES}/runs/{run}/summary_best_ada_reeval.json"))


INS = json.load(open(f"{RES}/reference/eval_instruct/summary.json"))
TEA = {"math": json.load(open(f"{RES}/reference/eval_teacher_math/summary.json")),
       "code": json.load(open(f"{RES}/reference/eval_teacher_code/summary.json"))}
NQ = {"gsm8k": 1319, "math500": 500, "mbpp_plus": 378, "humaneval_plus": 164}
KEYS = ["gsm8k", "math500", "heldout_code", "mbpp_plus", "humaneval_plus"]


def ch(s, ref, k):
    return 100 * (round(s[k] * NQ[k]) - round(ref[k] * NQ[k])) / NQ[k] if k in NQ else 100 * (s[k] - ref[k])


REC = [("multi GT-SFT", "multi_gt_sft", "sft"), ("multi T-SFT", "multi_t_sft", "sft"), ("multi KD", "multi_kd", "kd"),
       ("KD math", "kd_math", "kd"), ("KD code", "kd_code", "kd"), ("KD math → KD code", "seq_kd_math_then_kd_code", "kd"),
       ("KD code → KD math", "seq_kd_code_then_kd_math", "kd"), ("KD math → GRPO code", "seq_kd_math_then_rl_code", "rl"),
       ("KD code → GRPO math", "seq_kd_code_then_rl_math", "rl"), ("multi GRPO", "multi_rl", "rl")]
FAM = {"sft": (GREY, "SFT (GT-SFT, T-SFT)"), "kd": (BLUE, "KD only"), "rl": (ORANGE, "with GRPO")}
SUM = {lab: [ada(f"inst_{key}_s{sd}") for sd in range(3)] for lab, key, _ in REC}
CH = {lab: {k: [ch(s, INS, k) for s in SUM[lab]] for k in KEYS} for lab in SUM}
MEAN = {lab: {k: float(np.mean(v)) for k, v in d.items()} for lab, d in CH.items()}
COLR = {lab: FAM[f][0] for lab, _, f in REC}
for lab in MEAN:
    print(f"{lab:22s}", "  ".join(f"{k} {MEAN[lab][k]:+.1f} {[round(x, 1) for x in CH[lab][k]]}" for k in KEYS))
fam_legend = [Patch(color=c, label=l) for c, l in FAM.values()]


def hbars(ax, names, vals, cols, fs=15, h=0.66):
    for y, (n, v, c) in enumerate(zip(names, vals, cols)):
        ax.barh(y, v, height=h, color=c)
        ax.text(v + (0.3 if v >= 0 else -0.3), y, pp(v), ha="left" if v >= 0 else "right", va="center", fontsize=fs)


def vbars(ax, xs, vals, cols, w, fs, lim):
    for x, v, c in zip(xs, vals, cols):
        ax.bar(x, v, width=w, color=c)
        ax.text(x, v + (0.012 if v >= 0 else -0.012) * (lim[1] - lim[0]), pp(v), ha="center",
                va="bottom" if v >= 0 else "top", fontsize=fs)


# 4a. overview: gsm8k and held-out code for every recipe
names = [l for l, _, _ in REC][::-1]
fig, axs = plt.subplots(1, 2, figsize=(16.64, 5.3), sharey=True)
fig.subplots_adjust(wspace=0.06)
for ax, (k, title, lim) in zip(axs, [("gsm8k", "GSM8K", (-19, 9)), ("heldout_code", "held-out KodCode", (-4, 24))]):
    ax.axvline(0, color=INK2, lw=1.2)
    hbars(ax, names, [MEAN[n][k] for n in names], [COLR[n] for n in names])
    ax.set_xlim(*lim)
    ax.set_title(title, fontsize=22, pad=12)
    ax.set_xlabel("change vs instruct (pp)", fontsize=17)
    ax.tick_params(axis="y", length=0)
axs[0].set_yticks(range(len(names)))
axs[0].set_yticklabels(names, fontsize=18, color=INK)
# no colour legend: the row labels already name each recipe's method
save(fig, "res_overview")

# 4b. route 1 vs route 2 (and multi grpo): gsm8k and held-out code
RT = ["multi KD", "KD math → GRPO code", "KD code → GRPO math", "multi GRPO"]
RTL = ["multi KD\n(route 1)", "KD math →\nGRPO code", "KD code →\nGRPO math", "multi GRPO\n(no teacher)"]
fig, axs = plt.subplots(1, 2, figsize=(16.64, 4.5))
fig.subplots_adjust(wspace=0.16)
for ax, (k, title, lim) in zip(axs, [("gsm8k", "GSM8K", (-4, 7)), ("heldout_code", "held-out KodCode", (0, 23))]):
    ax.axhline(0, color=INK2, lw=1.2)
    vbars(ax, range(4), [MEAN[n][k] for n in RT], [COLR[n] for n in RT], 0.62, 19, lim)
    ax.set_ylim(*lim)
    ax.set_xticks(range(4))
    ax.set_xticklabels(RTL, fontsize=17)
    ax.tick_params(axis="x", length=0)
    ax.set_title(title, fontsize=22, pad=12)
axs[0].set_ylabel("change vs instruct (pp)")
save(fig, "res_routes")

# 4c. forgetting: the code stage from the same kd math checkpoint (same seed), with kd or with grpo; change during the
# code stage = second stage minus its own start
FG = {}
for name, key in [("KD code stage", "seq_kd_math_then_kd_code"), ("GRPO code stage", "seq_kd_math_then_rl_code")]:
    FG[name] = {k: [ch(ada(f"inst_{key}_s{sd}"), ada(f"inst_kd_math_s{sd}"), k) for sd in range(3)] for k in KEYS}
    print("forget", name, {k: [round(x, 1) for x in v] for k, v in FG[name].items()})
for name, key in [("KD math stage", "seq_kd_code_then_kd_math"), ("GRPO math stage", "seq_kd_code_then_rl_math")]:
    print("forget (code first)", name, {k: [round(ch(ada(f"inst_{key}_s{sd}"), ada(f"inst_kd_code_s{sd}"), k), 1) for sd in range(3)] for k in KEYS})
fig, axs = plt.subplots(1, 3, figsize=(16.64, 4.7), sharey=True)
fig.subplots_adjust(wspace=0.1)
lim = (-6, 22)
for ax, (k, title) in zip(axs, [("gsm8k", "GSM8K"), ("math500", "MATH-500"), ("heldout_code", "held-out KodCode")]):
    ax.axhline(0, color=INK2, lw=1.2)
    vbars(ax, range(2), [float(np.mean(FG[n][k])) for n in FG], [BLUE, ORANGE], 0.6, 20, lim)
    ax.set_ylim(*lim)
    ax.set_xticks(range(2))
    ax.set_xticklabels(["KD", "GRPO"], fontsize=19)
    ax.tick_params(axis="x", length=0)
    ax.set_title(title, fontsize=22, pad=12)
axs[0].set_ylabel("change in the code stage (pp)")
save(fig, "res_forget")

# 4d. text vs loss: gt-sft vs t-sft (same loss, different text) vs kd (same text as t-sft, different loss)
TL = ["multi GT-SFT", "multi T-SFT", "multi KD"]
fig, axs = plt.subplots(1, 3, figsize=(16.64, 4.7), sharey=True)
fig.subplots_adjust(wspace=0.1)
lim = (-18, 22)
for ax, (k, title) in zip(axs, [("gsm8k", "GSM8K"), ("math500", "MATH-500"), ("heldout_code", "held-out KodCode")]):
    ax.axhline(0, color=INK2, lw=1.2)
    vbars(ax, range(3), [MEAN[n][k] for n in TL], ["#6E6E68", "#B4B4AD", BLUE], 0.62, 19, lim)
    ax.set_ylim(*lim)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["GT-SFT", "T-SFT", "KD"], fontsize=19)
    ax.tick_params(axis="x", length=0)
    ax.set_title(title, fontsize=22, pad=12)
axs[0].set_ylabel("change vs instruct (pp)")
save(fig, "res_text_loss")

# 4e. transfer: every recipe's mean change per benchmark, one dot per recipe
TR = [("heldout_code", "held-out KodCode"), ("gsm8k", "GSM8K"), ("mbpp_plus", "MBPP+"), ("humaneval_plus", "HumanEval+"),
      ("math500", "MATH-500")]
fig, ax = plt.subplots(figsize=(16.64, 4.9))
ax.axvline(0, color=INK2, lw=1.2)
for i, (k, _) in enumerate(TR):
    y = len(TR) - 1 - i
    ax.axhline(y, color=GRID, lw=1, zorder=0)
    for lab, _, f in REC:
        ax.plot(MEAN[lab][k], y, "o", color=FAM[f][0], markersize=15, markeredgecolor=BG, markeredgewidth=1.5, zorder=3, alpha=0.9)
    if k in ("gsm8k", "math500"):
        v = MEAN["multi GT-SFT"][k]
        ax.annotate("GT-SFT", (v, y), xytext=(0, 16), textcoords="offset points", ha="center", fontsize=16, color=INK2)
ax.set_yticks(range(len(TR)))
ax.set_yticklabels([t for _, t in TR][::-1], fontsize=19, color=INK)
ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.6, len(TR) - 0.3)
ax.set_xlim(-17, 22)
ax.set_xlabel("change vs instruct (pp), one dot per recipe (3-seed mean)", fontsize=17)
ax.spines["left"].set_visible(False)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, markersize=13, label=l) for c, l in FAM.values()],
          loc="lower right", fontsize=17)
save(fig, "res_transfer")

# 4f. students vs teachers: absolute scores on the ada; math benchmarks against the math teacher, code against the code teacher
TB = [("gsm8k", "GSM8K", "math"), ("math500", "MATH-500", "math"), ("heldout_code", "held-out code", "code"),
      ("mbpp_plus", "MBPP+", "code"), ("humaneval_plus", "HumanEval+", "code")]
fig, ax = plt.subplots(figsize=(16.64, 4.9))
w = 0.2
for i, (k, _, t) in enumerate(TB):
    vals = [100 * INS[k], 100 * np.mean([s[k] for s in SUM["multi KD"]]), 100 * np.mean([s[k] for s in SUM["multi GRPO"]]), 100 * TEA[t][k]]
    for j, (v, c) in enumerate(zip(vals, ["#C9C9C2", BLUE, ORANGE, BG])):
        x = i + (j - 1.5) * w
        ax.bar(x, v, width=w * 0.92, color=c, edgecolor=INK if j == 3 else c, linewidth=2 if j == 3 else 0)
        ax.text(x, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=15)
    gap = [100 * (np.mean([s[k] for s in SUM[n]]) - INS[k]) / (TEA[t][k] - INS[k]) for n in ["multi KD", "multi GRPO"]]
    print(f"gap closed {k:15s} multi KD {gap[0]:5.1f}%  multi GRPO {gap[1]:5.1f}%  teacher {100 * TEA[t][k]:.1f}")
ax.set_xticks(range(len(TB)))
ax.set_xticklabels([b for _, b, _ in TB], fontsize=19)
ax.tick_params(axis="x", length=0)
ax.set_ylim(0, 105)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("score (%)")
fig.legend(handles=[Patch(color="#C9C9C2", label="instruct (0.5B)"), Patch(color=BLUE, label="multi KD"),
                    Patch(color=ORANGE, label="multi GRPO"), Patch(facecolor=BG, edgecolor=INK, linewidth=2, label="7B teacher of that domain")],
           loc="lower center", ncol=4, fontsize=18, bbox_to_anchor=(0.53, -0.1))
save(fig, "res_teachers")

# 4g. training cost: gpu-hours = train_h x gpus from log.json, a sequential recipe adds its first stage (same seed);
# validation and benchmarks not counted. teacher answers: gen_data.py, 3 h 19 min on one rtx pro 6000
PARTS = {"multi GT-SFT": ["multi_gt_sft"], "multi T-SFT": ["multi_t_sft"], "multi KD": ["multi_kd"], "KD math": ["kd_math"],
         "KD code": ["kd_code"], "KD math → KD code": ["kd_math", "seq_kd_math_then_kd_code"],
         "KD code → KD math": ["kd_code", "seq_kd_code_then_kd_math"], "KD math → GRPO code": ["kd_math", "seq_kd_math_then_rl_code"],
         "KD code → GRPO math": ["kd_code", "seq_kd_code_then_rl_math"], "multi GRPO": ["multi_rl"]}


def gpu_h(run):
    l = json.load(open(f"{RES}/runs/{run}/log.json"))
    return l["train_h"] * len(l["settings"]["gpus"])


COST = {lab: float(np.mean([sum(gpu_h(f"inst_{q}_s{sd}") for q in PARTS[lab]) for sd in range(3)])) for lab in PARTS}
print("cost", {k: round(v, 2) for k, v in COST.items()})
TEACHER_H = 3 + 19 / 60
fig, ax = plt.subplots(figsize=(16.64, 6.2))
for y, n in enumerate(names):
    ax.barh(y + 1, COST[n], height=0.66, color=COLR[n])
    ax.text(COST[n] + 0.15, y + 1, f"{COST[n]:.1f} h", va="center", fontsize=15)
ax.barh(0, TEACHER_H, height=0.66, color=BG, edgecolor=GREY, hatch="//", linewidth=1.5)
ax.text(TEACHER_H + 0.15, 0, f"{TEACHER_H:.1f} h, once, used by every KD and T-SFT recipe", va="center", fontsize=15)
ax.set_yticks(range(len(names) + 1))
ax.set_yticklabels(["teacher answers"] + names, fontsize=18, color=INK)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 16)
ax.set_xlabel("training GPU-hours (3-seed mean)", fontsize=17)
ax.legend(handles=fam_legend, loc="upper right", fontsize=17)
save(fig, "res_cost")

# 4h. transfer, simpler than 4e: multi kd vs multi grpo on all five benchmarks
TR2 = [("heldout_code", "held-out\nKodCode"), ("gsm8k", "GSM8K"), ("mbpp_plus", "MBPP+"), ("humaneval_plus", "HumanEval+"),
       ("math500", "MATH-500")]
fig, ax = plt.subplots(figsize=(16.64, 5.0))
lim = (-7, 25)
ax.axhline(0, color=INK2, lw=1.2)
ax.axvline(1.5, color=GRID, lw=2, ls="--")
for j, (n, c) in enumerate([("multi KD", BLUE), ("multi GRPO", ORANGE)]):
    vbars(ax, [i + (j - 0.5) * 0.38 for i in range(len(TR2))], [MEAN[n][k] for k, _ in TR2], [c] * len(TR2), 0.36, 19, lim)
ax.text(0.5, 23.5, "same distribution as training", ha="center", fontsize=18, color=INK2)
ax.text(3, 23.5, "other benchmarks", ha="center", fontsize=18, color=INK2)
ax.set_ylim(*lim)
ax.set_xticks(range(len(TR2)))
ax.set_xticklabels([t for _, t in TR2], fontsize=19)
ax.tick_params(axis="x", length=0)
ax.set_ylabel("change vs instruct (pp)")
ax.legend(handles=[Patch(color=BLUE, label="multi KD"), Patch(color=ORANGE, label="multi GRPO")], loc="center right",
          bbox_to_anchor=(1.0, 0.62), fontsize=18)
save(fig, "res_transfer_bars")

# 5. pre-checkpoint sequential kd, kd only, after each stage: pp vs the base student on the a6000, from the run
# summaries in 1_pre_checkpoint/runs (same numbers as results_table.py)
SK = "../../1_pre_checkpoint/runs"
BASE_N = {"gsm8k": (651, 1319), "mbpp_plus": (156, 378), "humaneval_plus": (56, 164)}
NKEY = {"gsm8k": "n_gsm8k", "mbpp_plus": "n_mbpp", "humaneval_plus": "n_humaneval"}


def stage_pp(out, stage):
    import glob
    rows = []
    for sd in range(3):
        s = json.load(open(glob.glob(f"{SK}/{out}{f'_seed{sd}' if sd else ''}/results_stage{stage}_*/summary.json")[0]))
        rows.append({k: 100 * (round(s[k] * s[NKEY[k]]) - b) / n for k, (b, n) in BASE_N.items()})
    return rows


ST = {}
for name, out in [("math → code", "abl_kl_only_math_code"), ("code → math", "abl_kl_only_code_math"),
                  ("math → code, KD + GT CE", "seq_math_code"), ("code → math, KD + GT CE", "seq_code_math"),
                  ("math → code, GT-SFT", "abl_sft_math_code"), ("code → math, GT-SFT", "abl_sft_code_math")]:
    ST[name] = [stage_pp(out, 1), stage_pp(out, 2)]
    for st in (0, 1):
        print("seq stage", name, st + 1, {k: (round(float(np.mean([r[k] for r in ST[name][st]])), 1),
                                            [round(r[k], 1) for r in ST[name][st]]) for k in BASE_N})
B3 = [("gsm8k", "GSM8K"), ("mbpp_plus", "MBPP+"), ("humaneval_plus", "HumanEval+")]
fig, axs = plt.subplots(1, 2, figsize=(16.64, 5.2), sharey=True)
fig.subplots_adjust(wspace=0.08)
lim = (-6.5, 4.5)
for ax, (name, col, title, s1, s2) in zip(axs, [("math → code", BLUE, "math first, then code", "math", "code"),
                                                 ("code → math", ORANGE, "code first, then math", "code", "math")]):
    ax.axhline(0, color=INK2, lw=1.2)
    # each panel's legend in its own colours: light = after stage 1, solid = after stage 2
    ax.legend(handles=[Patch(facecolor=col, alpha=0.35, edgecolor=col, linewidth=2, label=f"after stage 1 ({s1})"),
                       Patch(color=col, label=f"after stage 2 ({s2})")], loc="lower left", fontsize=18)
    for st in (0, 1):
        xs = [i + (st - 0.5) * 0.4 for i in range(3)]
        for x, (k, _) in zip(xs, B3):
            v = float(np.mean([r[k] for r in ST[name][st]]))
            ax.bar(x, v, width=0.38, color=col, alpha=0.35 if st == 0 else 1.0, edgecolor=col, linewidth=2)
            ax.text(x, v + (0.15 if v >= 0 else -0.15), pp(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=19)
    ax.set_ylim(*lim)
    ax.set_xticks(range(3))
    ax.set_xticklabels([t for _, t in B3], fontsize=20)
    ax.tick_params(axis="x", length=0)
    ax.set_title(title, fontsize=22, pad=12)
axs[0].set_ylabel("change vs instruct (pp)")
save(fig, "mid_seq_stages")
