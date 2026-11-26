# how repeatable are the evals? four checks
# 1. same weights, same gpu type, evaluated twice: runs whose best step = last step (results_best vs results_last)
# 2. same weights, other gpu type: instruct on the ada vs on the a6000
# 3. step 0 val (the start model) of every run, grouped by gpu type and start
# 4. the re-eval path (reeval.sh) on a run benchmarked on the ada: same answers expected
# run: python3 check/repeat.py
import glob
import json


def lines(path):
    return [json.loads(l) for l in open(path)]


def compare(a, b):
    for name, key, field in [("gsm8k", "answer", "right"), ("math500", "answer", "right"),
                             ("heldout_code", "answer", "reward")]:
        x, y = lines(f"{a}/{name}.jsonl"), lines(f"{b}/{name}.jsonl")
        if key not in x[0]:
            key = next(k for k in x[0] if k in ("completion", "text", "output"))
        same_text = sum(p[key] == q[key] for p, q in zip(x, y))
        score_diff_same_text = sum(p[key] == q[key] and p[field] != q[field] for p, q in zip(x, y))
        score_diff_other_text = sum(p[key] != q[key] and p[field] != q[field] for p, q in zip(x, y))
        print(f"  {name:13s} same text {same_text}/{len(x)}; score differs with same text {score_diff_same_text}, "
              f"with different text {score_diff_other_text}")


print("1. same weights, same gpu, evaluated twice")
for f in sorted(glob.glob("inst_*/log.json")):
    run = f.split("/")[0]
    log = json.load(open(f))
    last = log["val"][-1]["step"]
    if log.get("best_step") == last and glob.glob(f"{run}/results_last/summary.json"):
        print(f" {run} (best = last = step {last}, {log['settings']['gpus'][0]})")
        compare(f"{run}/results_best", f"{run}/results_last")

print("2. instruct, ada vs a6000")
compare("eval_instruct/results", "eval_instruct_a6000/results")

print("3. step 0 val")
for f in sorted(glob.glob("inst_*/log.json")):
    log = json.load(open(f))
    s, v = log["settings"], log["val"][0]
    gpu = s.get("gpus", ["?"])[0].replace("NVIDIA ", "")
    print(f" {f.split('/')[0]:36s} {gpu:22s} start {s['start'][-24:]:24s} math {v['val_math']:.3f} "
          f"code {v['val_code']:.6f}")

print("4. re-eval path (reeval.sh -> eval_model.py) vs the run's own benchmark, same weights, both on the ada")
compare("inst_kd_math_s0/results_best", "eval_ada/inst_kd_math_s0_best/results")
for name in ["humaneval", "mbpp"]:
    x = lines(f"inst_kd_math_s0/results_best/{name}.jsonl")
    y = lines(f"eval_ada/inst_kd_math_s0_best/results/{name}.jsonl")
    print(f"  {name:13s} same text {sum(p['solution'] == q['solution'] for p, q in zip(x, y))}/{len(x)}")
