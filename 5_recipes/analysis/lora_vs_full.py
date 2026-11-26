# lora (post-checkpoint) vs full fine-tuning (pre-checkpoint) on the matched recipes, last checkpoints.
# same data rows, steps, batch, schedule, loss and val set; differs in lora r16 lr 1e-4 vs full ft lr 1e-5.
# pp vs instruct on the gpu the run was benchmarked on (pre: a6000, post: the run's own gpu). run from 5_recipes/analysis
import json

PRE = "../../1_pre_checkpoint/runs"
POST = "../results/runs"
REF = "../results/reference"
OUT = "lora_vs_full.json"
B = {"gsm8k": 1319, "mbpp_plus": 378, "humaneval_plus": 164}
INS = {g: json.load(open(f"{REF}/{f}/summary.json")) for g, f in [("ada", "eval_instruct"), ("a6000", "eval_instruct_a6000")]}


def pp(s, ref):
    return {k: 100 * (round(s[k] * n) - round(ref[k] * n)) / n for k, n in B.items()}


def pre_run(folder, stage):
    return [pp(json.load(open(f"{PRE}/{folder}{sfx}/{stage}/summary.json")), INS["a6000"]) for sfx in ["", "_seed1", "_seed2"]]


def post_run(run, which="last"):
    out = []
    for sd in range(3):
        s = json.load(open(f"{POST}/inst_{run}_s{sd}/summary_{which}.json"))
        out.append(pp(s, INS["a6000" if "A6000" in s.get("gpu", "Ada") else "ada"]))
    return out


# joint pre-checkpoint runs: 3-seed means only (readme), no per-seed files here
JOINT = {"multi KD": (0.8, 0.4, -4.1), "multi T-SFT": (1.5, 0.5, -5.3), "multi GT-SFT": (-13.3, 1.7, -3.7)}
PAIRS = [("KD math", pre_run("abl_kl_only_math_code", "results_stage1_math"), "kd_math"),
         ("KD code", pre_run("abl_kl_only_code_math", "results_stage1_code"), "kd_code"),
         ("KD math → KD code", pre_run("abl_kl_only_math_code", "results_stage2_code"), "seq_kd_math_then_kd_code"),
         ("KD code → KD math", pre_run("abl_kl_only_code_math", "results_stage2_math"), "seq_kd_code_then_kd_math"),
         ("multi KD", None, "multi_kd"), ("multi T-SFT", None, "multi_t_sft"), ("multi GT-SFT", None, "multi_gt_sft")]

res = {}
for name, pre, run in PAIRS:
    pre_mean = {k: sum(r[k] for r in pre) / 3 for k in B} if pre else dict(zip(B, JOINT[name]))
    post_last, post_best = post_run(run, "last"), post_run(run, "best")
    res[name] = {"full_ft_seeds": pre, "full_ft_mean": pre_mean, "lora_last_seeds": post_last,
                 "lora_last_mean": {k: sum(r[k] for r in post_last) / 3 for k in B},
                 "lora_best_mean": {k: sum(r[k] for r in post_best) / 3 for k in B}}
    f = lambda d: " ".join(f"{d[k]:+6.1f}" for k in B)
    print(f"{name:20s} full {f(pre_mean)} | lora last {f(res[name]['lora_last_mean'])} | lora best {f(res[name]['lora_best_mean'])}")

# val KL to the teacher at the end of training (same 200 val rows per domain, same measure; instruct 0.1706 / 0.2212)
kl = {"full ft, math stage from instruct": [0.1177, 0.1181, 0.1180], "full ft, code stage from instruct": [0.1517, 0.1505, 0.1529],
      "full ft, math KL after the code stage from instruct": [0.1814, 0.1814, 0.1825],
      "full ft, code KL after the math stage from instruct": [0.2456, 0.2465, 0.2443]}
for run, key in [("kd_math", "val_loss_math"), ("kd_code", "val_loss_code")]:
    kl[f"lora {run} last"] = [json.load(open(f"{POST}/inst_{run}_s{sd}/log.json"))["val"][-1][key] for sd in range(3)]
# the other domain's KL after a lora first stage = step 0 of the second stage that starts from it (best checkpoint)
kl["lora, math KL after KD code (best)"] = [json.load(open(f"{POST}/inst_seq_kd_code_then_kd_math_s{sd}/log.json"))["val"][0]["val_loss_math"] for sd in range(3)]
kl["lora, code KL after KD math (best)"] = [json.load(open(f"{POST}/inst_seq_kd_math_then_kd_code_s{sd}/log.json"))["val"][0]["val_loss_code"] for sd in range(3)]
for k, v in kl.items():
    print(f"{k:55s} {' '.join(f'{x:.4f}' for x in v)}")
json.dump({"benchmarks_pp": res, "val_kl": kl}, open(OUT, "w"), indent=1)
