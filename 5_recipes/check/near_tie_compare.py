# near-tie check, part 2 (cpu): the two gpus' replays of instruct's gsm8k (check/near_tie.py). for each answer, the first
# token where the a6000 and the ada differ (the split), the top-2 logits there on each gpu, against the steps where both
# agree. bf16 ulp of x = 2 ^ (floor(log2 |x|) - 7): the spacing of bf16 numbers near x (8 significant bits)
# run from 5_recipes/: python3 check/near_tie_compare.py   (writes check/near_tie/splits.jsonl)
import json
import math

import torch
from transformers import AutoTokenizer

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
A = torch.load("check/near_tie/a6000.pt")
B = torch.load("check/near_tie/ada.pt")
tok = AutoTokenizer.from_pretrained(TOKENIZER)
right = {g: [json.loads(l)["right"] for l in open(f)] for g, f in
         [("a6000", "eval_instruct_a6000/results/gsm8k.jsonl"), ("ada", "eval_instruct/results/gsm8k.jsonl")]}


def ulp(x):
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7)


def logit_of(r, k, t):
    # the logit of token t at step k, if it is in that step's top 2
    hit = (r["top_i"][k] == t).nonzero()
    return r["top_v"][k, hit[0, 0]].item() if len(hit) else float("nan")


def pct(x, q):
    return torch.quantile(torch.tensor(x, dtype=torch.float64), q).item() if x else float("nan")


print(f"replay = saved answers: a6000 {A['same_as_saved']}/1319, ada {B['same_as_saved']}/1319")
splits, agree_gap, agree_gap_ulp, shift, shift_ulp = [], [], [], [], []
near = {"a6000": 0, "ada": 0}   # agreed steps with top-1 - top-2 <= 1 ulp, per gpu
same_text_other_ids = []
for q, (a, b) in enumerate(zip(A["rows"], B["rows"])):
    n = min(len(a["ids"]), len(b["ids"]))
    d = (a["ids"][:n] != b["ids"][:n]).nonzero()
    k = d[0, 0].item() if len(d) else None
    # steps where both gpus had the same text so far and picked the same token
    for name, r in (("a6000", a), ("ada", b)):
        g = (r["top_v"][:k, 0] - r["top_v"][:k, 1]).tolist()
        agree_gap += g
        gu = [x / ulp(v) for x, v in zip(g, r["top_v"][:k, 0].tolist())]
        agree_gap_ulp += gu
        near[name] += sum(x <= 1 for x in gu)
    if k is not None and A["answers"][q] == B["answers"][q]:
        same_text_other_ids.append(q)
    same_top = (a["top_i"][:k, 0] == b["top_i"][:k, 0])
    s = (a["top_v"][:k, 0] - b["top_v"][:k, 0])[same_top].abs().tolist()
    shift += s
    shift_ulp += [x / ulp(v) for x, v in zip(s, a["top_v"][:k, 0][same_top].tolist())]
    if k is None:
        continue
    ta, tb = a["ids"][k].item(), b["ids"][k].item()
    la_a, la_b = logit_of(a, k, ta), logit_of(a, k, tb)   # on the a6000: its token, the ada's token
    lb_a, lb_b = logit_of(b, k, ta), logit_of(b, k, tb)   # on the ada
    u = ulp(a["top_v"][k, 0].item())
    splits.append({"q": q, "k": k, "len_a6000": len(a["ids"]), "len_ada": len(b["ids"]),
                   "tok_a6000": tok.decode([ta]), "tok_ada": tok.decode([tb]),
                   "a6000_margin": la_a - la_b, "ada_margin": lb_b - lb_a, "ulp": u,
                   "gap_a6000": (a["top_v"][k, 0] - a["top_v"][k, 1]).item(),
                   "gap_ada": (b["top_v"][k, 0] - b["top_v"][k, 1]).item(),
                   "right_a6000": right["a6000"][q], "right_ada": right["ada"][q]})

with open("check/near_tie/splits.jsonl", "w") as f:
    for s in splits:
        f.write(json.dumps(s) + "\n")

n = len(splits)
both_top2 = [s for s in splits if not math.isnan(s["a6000_margin"]) and not math.isnan(s["ada_margin"])]
print(f"answers: {1319 - n} identical, {n} split")
print(f"at the split, each gpu's runner-up is the other gpu's token: {len(both_top2)}/{n}")
for g in ["a6000", "ada"]:
    gap = [s[f"gap_{g}"] for s in splits]
    gu = [s[f"gap_{g}"] / s["ulp"] for s in splits]
    print(f"  {g}: top-1 - top-2 at the split: median {pct(gap, 0.5):.4f} logits ({pct(gu, 0.5):.2f} ulp), "
          f"p90 {pct(gap, 0.9):.4f}; exactly 0 (tie) {sum(x == 0 for x in gap)}, <= 1 ulp {sum(x <= 1 for x in gu)}")
print(f"steps where both agree: {len(agree_gap) // 2} per gpu; top-1 - top-2 median {pct(agree_gap, 0.5):.3f} logits "
      f"({pct(agree_gap_ulp, 0.5):.1f} ulp), p1 {pct(agree_gap, 0.01):.4f}; <= 1 ulp: "
      f"{sum(x <= 1 for x in agree_gap_ulp)} of {len(agree_gap_ulp)} ({100 * sum(x <= 1 for x in agree_gap_ulp) / len(agree_gap_ulp):.2f}%)")
print(f"cross-gpu change of the top-1 logit at agreed steps: median {pct(shift, 0.5):.4f} ({pct(shift_ulp, 0.5):.2f} ulp), "
      f"p90 {pct(shift, 0.9):.4f} ({pct(shift_ulp, 0.9):.2f} ulp), p99 {pct(shift, 0.99):.4f}, max {max(shift):.4f}; "
      f"exactly equal {sum(x == 0 for x in shift)} of {len(shift)}")
for g in ["a6000", "ada"]:
    at = sum(s[f"gap_{g}"] / s["ulp"] <= 1 for s in splits)
    print(f"  {g}: near-ties (<= 1 ulp) met while the texts were still the same: {near[g] + at}, of which the split {at} "
          f"({100 * at / (near[g] + at):.1f}%); per answer {(near[g] + at) / 1319:.2f}")
print(f"same text, different token ids: {same_text_other_ids}")
ks = [s["k"] for s in splits]
rel = [s["k"] / max(s["len_a6000"], s["len_ada"]) for s in splits]
print(f"split position: median token {pct(ks, 0.5):.0f} (p10 {pct(ks, 0.1):.0f}, p90 {pct(ks, 0.9):.0f}); "
      f"as a fraction of the answer: median {pct(rel, 0.5):.2f}")
flip = sum(s["right_a6000"] != s["right_ada"] for s in splits)
print(f"split answers whose right / wrong differs: {flip}/{n} (only a6000 right {sum(s['right_a6000'] and not s['right_ada'] for s in splits)}, "
      f"only ada right {sum(s['right_ada'] and not s['right_a6000'] for s in splits)})")
