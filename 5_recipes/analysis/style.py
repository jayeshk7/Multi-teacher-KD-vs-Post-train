# q1: does training copy the style of the training answers instead of teaching skill? for every model: answer length,
# format and how much is written out; its accuracy change vs the instruct model and the change inside the same style;
# and whether the questions it lost are the same in all 3 seeds.
# writes style.md. needs run folders with the answer text (not in git): run from analysis/ inside that folder

import collections
import glob
import json
import os
import re

from transformers import AutoTokenizer

RUNS = ".."
REF = "eval_instruct"
TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
# model -> its results folders (one per seed); the reference has one
MODELS = collections.defaultdict(list)
for p in sorted(glob.glob(f"{RUNS}/*_s[0-9]/results_best")):
    MODELS[re.sub(r"_s\d$", "", p.split("/")[-2])].append(p)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)


def ntok(text):
    # answer length in tokens
    return len(tokenizer(text).input_ids)


def read(folder, bench):
    # per question: right (1/0) and style features, in a fixed question order
    rows = []
    if bench == "gsm8k":
        for r in map(json.loads, open(f"{folder}/gsm8k.jsonl")):
            a = r["answer"].strip()
            rows.append({"right": float(r["right"]), "tokens": ntok(a),
                         "gt_style": bool(re.search(r"The answer is \\boxed\{[^{}]*\}\.?$", a)),
                         "equations": len(re.findall(r"\d\s*=\s*\$?-?\d", a)), "boxed": "\\boxed" in a})
    elif bench == "math500":
        for r in map(json.loads, open(f"{folder}/math500.jsonl")):
            rows.append({"right": float(r["right"]), "tokens": ntok(r["answer"]), "boxed": "\\boxed" in r["answer"]})
    elif bench in ("mbpp", "humaneval"):
        ev = json.load(open(f"{folder}/{bench}_eval_results.json"))["eval"]
        raw = {r["task_id"]: r["raw"] for r in map(json.loads, open(f"{folder}/{bench}.jsonl"))}
        for t, v in sorted(ev.items()):
            code, _, after = raw[t].partition("```")
            rows.append({"right": float(v[0]["base_status"] == v[0]["plus_status"] == "pass"), "tokens": ntok(raw[t]),
                         "text_after_code": bool(after.strip()),
                         "test_code": bool(re.search(r"^\s*(assert |print\()", code, re.M))})
    else:
        for r in map(json.loads, open(f"{folder}/heldout_code.jsonl")):
            a = r["answer"]
            outside = re.sub(r"```.*?```", "", a, flags=re.S)
            rows.append({"right": float(r["reward"] == 1), "tokens": ntok(a), "text_outside_code": bool(outside.strip()),
                         "test_code": bool(re.search(r"^\s*(assert |print\(|def test_)", a, re.M))})
    return rows


FILES = {"gsm8k": "gsm8k.jsonl", "math500": "math500.jsonl", "mbpp": "mbpp_eval_results.json",
         "humaneval": "humaneval_eval_results.json", "heldout_code": "heldout_code.jsonl"}
BENCHES = {"gsm8k": ("gt_style", "GSM8K"), "math500": ("boxed", "MATH-500"), "mbpp": ("text_after_code", "MBPP+"),
           "humaneval": ("text_after_code", "HE+"), "heldout_code": ("text_outside_code", "held-out code")}


def bins(rows, cuts, flag):
    # style bin of each answer: length tercile (cuts from the instruct model) x the benchmark's format flag
    return [(sum(r["tokens"] > c for c in cuts), r[flag]) for r in rows]


MIN_BIN = 30   # answers a style bin needs in both models (per seed for the model) to be compared


def decompose(ref_rows, runs, flag):
    # accuracy change vs instruct, and the part of it that happens inside the same style: model minus instruct accuracy
    # per style bin (length tercile x the format flag), weighted by instruct's bin shares. only bins both models use
    # often enough count; coverage = share of instruct's answers in those bins. low coverage = the styles barely overlap
    lens = sorted(r["tokens"] for r in ref_rows)
    cuts = [lens[len(lens) // 3], lens[2 * len(lens) // 3]]
    ref_acc = collections.defaultdict(list)
    for k, r in zip(bins(ref_rows, cuts, flag), ref_rows):
        ref_acc[k].append(r["right"])
    acc = collections.defaultdict(list)
    for rows in runs:
        for k, r in zip(bins(rows, cuts, flag), rows):
            acc[k].append(r["right"])
    kept = [k for k in ref_acc if len(ref_acc[k]) >= MIN_BIN and len(acc[k]) >= MIN_BIN * len(runs)]
    weight = sum(len(ref_acc[k]) for k in kept)
    mean = lambda x: sum(x) / len(x)
    total = 100 * (mean([r["right"] for rows in runs for r in rows]) - mean([r["right"] for r in ref_rows]))
    if not kept:
        return total, None, 0.0
    within = 100 * sum(len(ref_acc[k]) * (mean(acc[k]) - mean(ref_acc[k])) for k in kept) / weight
    return total, within, weight / len(ref_rows)


def flips(ref_rows, runs):
    # questions the instruct model gets right that the model loses in all 3 seeds, vs what chance would give if each
    # seed lost questions independently at its observed rate
    right = [i for i, r in enumerate(ref_rows) if r["right"]]
    lost = [sum(1 for rows in runs if not rows[i]["right"]) for i in right]
    rate = sum(lost) / (len(right) * len(runs))
    return sum(n == len(runs) for n in lost), len(right) * rate ** len(runs)


if __name__ == "__main__":
    ref_folder = f"{RUNS}/{REF}/results"
    out = ["# style vs skill (q1)\n"]
    for bench, (flag, name) in BENCHES.items():
        # a benchmark is skipped where its answers were not saved (the october runs have no math-500 / held-out code)
        if not os.path.exists(f"{ref_folder}/{FILES[bench]}"):
            continue
        ref_rows = read(ref_folder, bench)
        feats = [k for k in ref_rows[0] if k not in ("right", "tokens")]
        out += [f"## {name}\n", "| model | right | median tokens | " + " | ".join(f"% {k}" if k != "equations" else
                "equations written" for k in feats) + " |", "|---" * (len(feats) + 3) + "|"]
        styled = {"instruct": [ref_rows]}
        styled.update({m: [read(f, bench) for f in folders if os.path.exists(f"{f}/{FILES[bench]}")]
                       for m, folders in MODELS.items()})
        styled = {m: runs for m, runs in styled.items() if runs}
        for m, runs in styled.items():
            allrows = [r for rows in runs for r in rows]
            cells = [f"{100 * sum(r['right'] for r in allrows) / len(allrows):.1f}",
                     f"{sorted(r['tokens'] for r in allrows)[len(allrows) // 2]}"]
            for k in feats:
                v = sum(r[k] for r in allrows) / len(allrows)
                cells.append(f"{v:.2f}" if k == "equations" else f"{100 * v:.1f}")
            out.append(f"| {m} | " + " | ".join(cells) + " |")
        out += ["", f"accuracy change vs instruct (pp), and the change inside the same style (length tercile x {flag}, "
                f"bins both models use >= {MIN_BIN} times); coverage = instruct's answers in those bins; questions instruct "
                "gets right that the model loses in all seeds vs chance\n",
                "| model | change | within same style | coverage | lost in all seeds | chance |", "|---|---|---|---|---|---|"]
        for m in [m for m in styled if m != "instruct"]:
            runs = styled[m]
            total, within, coverage = decompose(ref_rows, runs, flag)
            seen, chance = flips(ref_rows, runs)
            inside = f"{within:+.1f}" if within is not None and coverage >= 0.5 else "styles barely overlap"
            out.append(f"| {m} | {total:+.1f} | {inside} | {100 * coverage:.0f}% | {seen} | {chance:.1f} |")
        out.append("")
    with open("style.md", "w") as f:
        f.write("\n".join(out))
    print("\n".join(out))
