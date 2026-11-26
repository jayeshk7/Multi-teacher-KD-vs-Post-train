# error analysis of the saved test answers (gsm8k.jsonl, math500.jsonl), all on the rtx 6000 ada.
# per recipe: answer style, arithmetic slips, skipped steps, where a wrong answer leaves the reference solution,
# math-500 by level and answer type. run once: python error_analysis.py (writes OUT, prints a summary)

import json
import re
import warnings
from collections import Counter
from statistics import mean

warnings.filterwarnings("ignore", category=SyntaxWarning)   # eval of "2(3)" style text

DATA = "/home/jayeshk7/Desktop/IITB_coursework/Sem3/FML/project/stage2_gpu1_mirror"   # run folders with answer text (not in git)
OUT = "error_analysis_all.json"

# recipe label -> run folder name. every run's best checkpoint, benchmarked on the ada
NAMES = [("multi GT-SFT", "multi_gt_sft"), ("multi T-SFT", "multi_t_sft"), ("multi KD", "multi_kd"), ("KD math", "kd_math"),
         ("KD code", "kd_code"), ("KD math → KD code", "seq_kd_math_then_kd_code"),
         ("KD code → KD math", "seq_kd_code_then_kd_math"), ("KD math → GRPO code", "seq_kd_math_then_rl_code"),
         ("KD code → GRPO math", "seq_kd_code_then_rl_math"), ("multi GRPO", "multi_rl")]


def ada_folder(run):
    # the run's own results if it ran on the ada, else its ada re-eval
    own = f"inst_{run}/results_best"
    gpu = json.load(open(f"{DATA}/{own}/summary.json")).get("gpu", "")
    return own if "Ada" in gpu else f"eval_ada/inst_{run}_best/results"


RECIPES = {"instruct": ["eval_instruct/results"]}
RECIPES.update({label: [ada_folder(f"{key}_s{sd}") for sd in range(3)] for label, key in NAMES})


def load(path):
    return [json.loads(line) for line in open(f"{DATA}/{path}")]


def normalize(text):
    # latex and plain arithmetic to one plain form: 1,200 -> 1200, \times -> *, \frac{a}{b} -> ((a)/(b)), units dropped
    t = text.replace("$", " ")
    for _ in range(3):
        t = re.sub(r"(\d)(?:,|\{,\}|\\,)(\d{3})", r"\1\2", t)
    t = re.sub(r"\\(text|mathrm|textbf|mbox)\{[^{}]*\}", " ", t)
    for _ in range(3):
        t = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", t)
    t = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", t)
    t = t.replace("\\approx", "=").replace("\\left", "").replace("\\right", "")
    t = re.sub(r"\\(times|cdot)|×|⋅", "*", t)
    t = re.sub(r"\\div|÷", "/", t)
    t = re.sub(r"\\[()\[\]]|\\[,;! ]|\\quad|\\qquad", " ", t)
    t = re.sub(r"(\d)\s*[xX]\s*(?=[\d(])", r"\1*", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*\\?%", r"(\1/100)", t)
    t = t.replace("^", "**")
    return t


ARITH_TAIL = re.compile(r"[\d.\s+\-*/()]+$")
ARITH_HEAD = re.compile(r"^[\d.\s+\-*/()]+")
NUM_HEAD = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")
HAS_OP = re.compile(r"[\d)]\s*(\*\*|[+\-*/])\s*[\d(.]")


def value(expr):
    # value of a plain arithmetic expression, None if it can't be computed
    if re.search(r"\*\*\s*\d{3,}", expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}))
    except Exception:
        return None


def calcs(text):
    # every written calculation "a op b = c": (left value, right value, decimals written, text).
    # chains a = b = c are checked pair by pair, the right side may itself be arithmetic (2*(365-180) = 2*185 = 370)
    out = []
    for line in normalize(text).split("\n"):
        line = re.sub(r"^\s*(\d+\.|[-*])\s+", " ", line)   # list markers "1. ", "- "
        parts = line.split("=")
        for left, right in zip(parts, parts[1:]):
            m = ARITH_TAIL.search(left)
            expr = m.group().strip() if m else ""
            if not expr or not HAS_OP.search(expr) or expr[0] in ")*/+":
                continue   # no calculation, or its first operand is not a number ("80 lorries * 1/4")
            lv = value(expr)
            head = ARITH_HEAD.match(right)
            if lv is None or (head and HAS_OP.search(head.group())):
                continue   # the right side is itself arithmetic: checked at the next "="
            r = NUM_HEAD.match(right)
            if not r:
                continue
            if re.match(r"[a-zA-Z(]", right[r.end():]):
                continue   # algebra: "= 2S", "= 2(A - 4)"
            rtext, rv = r.group(1), float(r.group(1))
            places = max((len(x.split(".")[1]) for x in re.findall(r"\d+\.\d+", rtext)), default=0)
            out.append((lv, rv, places, f"{m.group().strip()} = {rtext}"))
    return out


def calc_ok(lv, rv, places, text=""):
    # right if equal up to the decimals the answer wrote (rounded or cut: 10/3 = 3.33, 32/3 = 10) or within 0.5 %
    # (4500/2.81 = 1600). with a percent in it, a factor of 100 is notation (60/150*100 = 40%), not a slip
    for v in [lv, lv * 100, lv / 100] if "/100" in text else [lv]:
        if abs(v - rv) <= 10 ** -places + 1e-9 or abs(v - rv) <= 0.005 * abs(v):
            return True
    return False


def numbers(text):
    # every number in the text, rounded
    out = set()
    for n in re.findall(r"-?\d+(?:\.\d+)?", normalize(text)):
        try:
            out.add(round(float(n), 4))
        except ValueError:
            pass
    return out


def ref_steps(solution):
    # gsm8k test solutions mark each step as <<expression=value>>
    return [float(v) for v in re.findall(r"<<[^=>]*=\s*(-?[\d.]+)\s*>>", solution) if v.strip(".")]


def looping(text):
    # very long, or one line repeated 4+ times
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 10]
    return len(text) > 4500 or (bool(lines) and Counter(lines).most_common(1)[0][1] >= 4)


def size(c):
    # calculation size: a product / quotient of two numbers >= 10, a sum or chain of 3+ numbers, or small
    e = c[3].split("=")[0]
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", e)]
    if re.search(r"[*/]", e) and len([v for v in nums if v >= 10]) >= 2:
        return "large mul / div"
    return "3+ numbers" if len(nums) >= 3 else "small"


def trigrams(text):
    # 3-word phrases, to measure how much of the question the answer restates
    w = re.findall(r"[a-z]+", text.lower())
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def final_number(text):
    # same rule as the gsm8k score: last \boxed{}, else the last number
    found = re.findall(r"\\boxed\{([^{}]*)\}", text) or re.findall(r"-?\d[\d,]*\.?\d*", text)
    try:
        return float(re.sub(r"[^\d.\-]", "", found[-1]))
    except (IndexError, ValueError):
        return None


def repeats(text):
    # a loop: a long line 4+ times, or the last 600 characters are mostly repeated lines
    tail = text[-600:].split("\n")
    return looping(text) or len(set(tail)) < 0.5 * len(tail)


def wrong_kind(ans, row_calcs):
    # one label per wrong gsm8k answer, checked in this order
    if not re.search(r"\d", ans):
        return "no number"
    if looping(ans):
        return "loop or very long"
    if any(not calc_ok(*c) for c in row_calcs):
        return "arithmetic slip"
    return "calculations right, setup wrong"


def gsm8k_stats(rows, questions):
    n = len(rows)
    per = []
    sizes = Counter()
    for row, q in zip(rows, questions):
        c = calcs(row["answer"])
        for x in c:
            sizes[(size(x), calc_ok(*x))] += 1
        ref = ref_steps(q["solution"])
        qnums = numbers(q["question"])
        needed = qnums & numbers(q["solution"])   # question numbers the reference solution uses
        fv = final_number(row["answer"])
        new_ref = [v for v in ref if round(v, 4) not in qnums]
        found = numbers(row["answer"])
        hit = [round(v, 4) in found for v in new_ref]
        first_miss = hit.index(False) if False in hit else None
        per.append({"right": row["right"], "chars": len(row["answer"]),
                    "lines": sum(1 for l in row["answer"].split("\n") if l.strip()),
                    "boxed": "\\boxed{" in row["answer"], "n_calc": len(c), "n_ref": len(ref),
                    "slip": any(not calc_ok(*x) for x in c),
                    "kind": None if row["right"] else wrong_kind(row["answer"], c),
                    "ref_recall": mean(hit) if hit else None,
                    "miss_first": (first_miss == 0) if new_ref else None,
                    "skips_needed": bool(needed - found) if needed else None,
                    "restated": len(trigrams(q["question"]) & trigrams(row["answer"])) / max(1, len(trigrams(q["question"]))),
                    "stops_early": (not row["right"]) and fv is not None and any(abs(fv - v) < 1e-6 for v in ref[:-1])})
    wrong = [p for p in per if not p["right"]]
    kinds = Counter(p["kind"] for p in wrong)
    by_steps = {}
    for lo, hi, name in [(1, 2, "1-2"), (3, 3, "3"), (4, 4, "4"), (5, 5, "5"), (6, 99, "6+")]:
        sel = [p for p in per if lo <= p["n_ref"] <= hi]
        by_steps[name] = {"n": len(sel), "acc": mean(p["right"] for p in sel)}
    return {
        "acc": mean(p["right"] for p in per),
        "chars": mean(p["chars"] for p in per), "lines": mean(p["lines"] for p in per),
        "boxed": mean(p["boxed"] for p in per),
        "calcs_per_answer": mean(p["n_calc"] for p in per),
        "fewer_calcs_than_ref": mean(p["n_calc"] < p["n_ref"] for p in per),
        "answers_with_slip": mean(p["slip"] for p in per),
        "wrong": len(wrong),
        "wrong_kinds_pct_of_all": {k: 100 * v / n for k, v in kinds.items()},
        "wrong_kinds_pct_of_wrong": {k: 100 * v / len(wrong) for k, v in kinds.items()},
        "wrong_ref_recall": mean(p["ref_recall"] for p in wrong if p["ref_recall"] is not None),
        "right_ref_recall": mean(p["ref_recall"] for p in per if p["right"] and p["ref_recall"] is not None),
        "wrong_off_at_step1": mean(p["miss_first"] for p in wrong if p["miss_first"] is not None),
        "acc_by_ref_steps": by_steps,
        "skips_needed_number": mean(p["skips_needed"] for p in per if p["skips_needed"] is not None),
        "wrong_skips_needed_number": mean(p["skips_needed"] for p in wrong if p["skips_needed"] is not None),
        "question_restated": mean(p["restated"] for p in per),
        "wrong_stops_at_earlier_step": mean(p["stops_early"] for p in wrong),
        "slip_rate_by_size": {k: sizes[(k, False)] / max(1, sizes[(k, False)] + sizes[(k, True)])
                              for k in ["small", "large mul / div", "3+ numbers"]},
        "calcs_by_size": {k: sizes[(k, False)] + sizes[(k, True)] for k in ["small", "large mul / div", "3+ numbers"]},
    }, per


def answer_type(gold):
    # math-500 answer type, to split accuracy by it
    g = gold.strip()
    if re.fullmatch(r"-?\d+", g.replace(",", "")):
        return "integer"
    if re.fullmatch(r"-?\d*\.\d+", g):
        return "decimal"
    if re.search(r"\\d?frac|^-?\d+/\d+$", g) and not re.search(r"[a-zA-Z]{2,}", g.replace("frac", "").replace("dfrac", "")):
        return "fraction"
    if "sqrt" in g or "pi" in g:
        return "root / pi"
    return "expression / other"


def math500_stats(rows):
    out = {"acc": mean(r["right"] for r in rows), "chars": mean(len(r["answer"]) for r in rows),
           "boxed": mean("\\boxed{" in r["answer"] for r in rows), "cut": mean(not r["ended"] for r in rows),
           "latex": mean(bool(re.search(r"\\\(|\\\[|\$", r["answer"])) for r in rows),
           "cut_and_looping": mean((not r["ended"]) and repeats(r["answer"]) for r in rows)}
    out["by_level"] = {str(lv): mean(r["right"] for r in rows if r["level"] == lv) for lv in range(1, 6)}
    types = Counter(answer_type(r["gold"]) for r in rows)
    out["by_type"] = {t: {"n": types[t], "acc": mean(r["right"] for r in rows if answer_type(r["gold"]) == t)}
                      for t in types}
    return out


def avg(dicts):
    # mean over seeds of every number in the (nested) stats dicts
    first = dicts[0]
    if isinstance(first, dict):
        return {k: avg([d[k] for d in dicts if k in d]) for k in first}
    return mean(dicts)


questions = load("analysis/gsm8k_test_q.jsonl")
train = load("analysis/gsm8k_train_sol.jsonl")
results, per_answer = {}, {}
for name, dirs in RECIPES.items():
    g_seeds, m_seeds = [], []
    for d in dirs:
        g, per = gsm8k_stats(load(f"{d}/gsm8k.jsonl"), questions)
        g_seeds.append(g)
        m_seeds.append(math500_stats(load(f"{d}/math500.jsonl")))
        per_answer[d] = per
    results[name] = {"gsm8k": avg(g_seeds), "math500": avg(m_seeds), "gsm8k_seeds": g_seeds, "math500_seeds": m_seeds}

# the training targets of gt-sft: gsm8k train solutions without the <<...>> notes, plus the boxed line
gold_text = [re.sub(r"<<.*?>>", "", s["solution"].split("####")[0]).strip() for s in train]
results["GT-SFT training targets"] = {"chars": mean(len(t) + 25 for t in gold_text),
                                      "calcs_per_answer": mean(len(calcs(t)) for t in gold_text)}

# answers lost by every gt-sft seed that instruct and every t-sft seed got right (the examples slide)
ins = load(f"{RECIPES['instruct'][0]}/gsm8k.jsonl")
gts = [load(f"{d}/gsm8k.jsonl") for d in RECIPES["multi GT-SFT"]]
tss = [load(f"{d}/gsm8k.jsonl") for d in RECIPES["multi T-SFT"]]
robust = [i for i in range(len(ins)) if ins[i]["right"] and all(t[i]["right"] for t in tss)
          and not any(g[i]["right"] for g in gts)]
results["robust_lost_gsm8k"] = {"n": len(robust),
                                "kinds_seed0": Counter(per_answer[RECIPES["multi GT-SFT"][0]][i]["kind"] for i in robust)}

# per recipe, gsm8k and math-500 questions vs instruct: lost / gained per seed, and lost / gained by all 3 seeds
for bench in ["gsm8k", "math500"]:
    base = [r["right"] for r in load(f"{RECIPES['instruct'][0]}/{bench}.jsonl")]
    for name, dirs in list(RECIPES.items())[1:]:
        seeds = [[r["right"] for r in load(f"{d}/{bench}.jsonl")] for d in dirs]
        results[name][f"{bench}_vs_instruct"] = {
            "lost": [sum(b and not x for b, x in zip(base, sd)) for sd in seeds],
            "gained": [sum(x and not b for b, x in zip(base, sd)) for sd in seeds],
            "lost_all_seeds": sum(b and not any(sd[i] for sd in seeds) for i, b in enumerate(base)),
            "gained_all_seeds": sum((not b) and all(sd[i] for sd in seeds) for i, b in enumerate(base))}

json.dump({"results": results, "robust_lost_ids": robust}, open(OUT, "w"), indent=1)

for name, r in results.items():
    if "gsm8k" in r:
        g = r["gsm8k"]
        print(f"{name:22s} acc {100 * g['acc']:5.1f} chars {g['chars']:6.0f} calcs {g['calcs_per_answer']:.2f} "
              f"setup {g['wrong_kinds_pct_of_all'].get('calculations right, setup wrong', 0):5.1f} "
              f"slip {g['wrong_kinds_pct_of_all'].get('arithmetic slip', 0):4.1f} loop {g['wrong_kinds_pct_of_all'].get('loop or very long', 0):4.1f} "
              f"boxed {100 * g['boxed']:5.1f} restated {100 * g['question_restated']:4.1f} "
              f"bigslip {100 * g['slip_rate_by_size']['large mul / div']:4.1f}")
