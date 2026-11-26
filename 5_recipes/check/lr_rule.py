# rl lr choice, the rule fixed before the runs: highest val at step 60 among the lrs whose cut share over
# steps 41-60 is <= 10% and whose entropy stays above half its step-1 value; if the best is within 1 SE of 1e-4, 1e-4.
# "stays above" is read two ways (every step 1-60, and the mean of steps 41-60); both are printed
import json
from math import sqrt
N = 200  # val rows per domain
runs = {lr: json.load(open(f"lr_{lr}/log.json")) for lr in ["5e-5", "1e-4", "2e-4"]}
res = {}
for lr, log in runs.items():
    steps = [s for s in log["train"] if "entropy" in s]
    ent = [float(s["entropy"]) for s in steps]
    cut = [float(s["completions/clipped_ratio"]) for s in steps]
    last = [v for v in log["val"] if v["step"] == log["settings"]["steps"]][0]
    m, c = last["val_math"], last["val_code"]
    se = 0.5 * sqrt(m * (1 - m) / N + c * (1 - c) / N)  # code: m(1-m) bounds the variance of a score in [0, 1]
    half = ent[0] / 2
    res[lr] = dict(val=last["val"], math=m, code=c, se=se, cut41=sum(cut[40:60]) / len(cut[40:60]), ent1=ent[0],
                   ent_min=min(ent), ent41=sum(ent[40:60]) / len(ent[40:60]), below=sum(e <= half for e in ent),
                   ok_every=sum(cut[40:60]) / len(cut[40:60]) <= 0.10 and min(ent) > half,
                   ok_mean=sum(cut[40:60]) / len(cut[40:60]) <= 0.10 and sum(ent[40:60]) / len(ent[40:60]) > half,
                   vals=[(v["step"], round(v["val_math"], 3), round(v["val_code"], 3), round(v["val"], 4),
                          round(v["drift_math"], 3), round(v["drift_code"], 3)) for v in log["val"]],
                   train_h=log["train_h"], val_h=log["val_h"], n=len(steps))
for lr, r in res.items():
    print(f"lr {lr}: {r['n']} steps, val@60 {r['val']:.4f} (math {r['math']:.3f}, code {r['code']:.3f}, SE {r['se']:.4f}), "
          f"cut 41-60 {r['cut41']:.3f}, entropy step1 {r['ent1']:.3f} min {r['ent_min']:.3f} mean41-60 {r['ent41']:.3f}, "
          f"steps <= half {r['below']}, ok(every step) {r['ok_every']}, ok(mean 41-60) {r['ok_mean']}, "
          f"train {r['train_h']:.2f} h, val {r['val_h']:.2f} h")
    print("   val checks (step, math, code, val, drift math, drift code):", r["vals"])
for reading in ["ok_every", "ok_mean"]:
    ok = [lr for lr in res if res[lr][reading]]
    if not ok:
        print(f"{reading}: no lr passes, the rule does not decide"); continue
    best = max(ok, key=lambda lr: res[lr]["val"])
    ref = res["1e-4"]
    keep = best != "1e-4" and "1e-4" in ok and res[best]["val"] - ref["val"] <= ref["se"]
    print(f"{reading}: passing {ok}, best {best}; {'within 1 SE of 1e-4 -> 1e-4' if keep else 'chosen: ' + best}")
