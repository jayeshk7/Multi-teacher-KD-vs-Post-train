# error analysis: every wrong benchmark answer gets one category. failing code is run again on its first failing input
# (mbpp+, humaneval+) or its tests (held-out kodcode, same sandbox as reward.py) to get the exception type.
# writes errors.jsonl (one row per answer) and errors.md (counts per run and per recipe).
# needs the run folders with the answer text (not in git): run from analysis/ inside that folder

import ast
import collections
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from transformers import AutoTokenizer

sys.path.insert(0, "..")
from reward import FILE_BLOCKS, MEMORY_KB, TEST_HEADER, extract_code, fraction_passed  # noqa: E402

def ada_folder(run):
    # the ada benchmark of the run's best checkpoint: its own eval if it ran on the ada, else the ada re-eval
    own = f"../{run}/results_best"
    gpu = json.load(open(f"{own}/summary.json")).get("gpu", "")
    return own if "Ada" in gpu else f"../eval_ada/{run}_best/results"


RESULTS = {"instruct": "../eval_instruct/results"}
RESULTS.update({r[len("inst_"):]: ada_folder(r) for r in sorted(os.path.basename(p) for p in glob.glob("../inst_*_s[0-9]"))})
TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
TEST_MAX = 1600     # answers this long were cut
TIMEOUT = 10        # as reward.py
WORKERS = min(32, os.cpu_count())

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
problems = {"mbpp": get_mbpp_plus(), "humaneval": get_human_eval_plus()}

NUM = r"-?\d[\d,]*\.?\d*"
SLIP = r"(\d[\d,]*\.?\d*)\s*([+\-*/×x÷])\s*(\d[\d,]*\.?\d*)\s*=\s*\$?(\d[\d,]*\.?\d*)"


def to_float(s):
    return float(re.sub(r"[^\d.\-]", "", s))


def safe_float(s):
    # None for things like "4500-2" that are not one number
    try:
        return to_float(s)
    except ValueError:
        return None


def looping(answer):
    # the same non-trivial line 4 or more times
    lines = [line.strip() for line in answer.splitlines() if len(line.strip()) > 10]
    return bool(lines) and collections.Counter(lines).most_common(1)[0][1] >= 4


def arithmetic_slip(answer):
    # some "a op b = c" in the answer is wrong. only catches single operations, so it is a lower bound
    for a, op, b, c in re.findall(SLIP, answer):
        try:
            a, b, c = to_float(a), to_float(b), to_float(c)
            val = {"+": a + b, "-": a - b, "*": a * b, "×": a * b, "x": a * b, "/": a / b, "÷": a / b}[op]
        except (ValueError, ZeroDivisionError):
            continue
        if all(abs(round(val, k) - c) > 1e-6 for k in range(4)):
            return True
    return False


def gsm8k_category(answer, gold):
    if len(tokenizer(answer).input_ids) >= TEST_MAX - 1:
        return "cut, looping" if looping(answer) else "cut, long"
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", answer)
    if not boxed and not re.findall(NUM, answer):
        return "no final answer"
    last_line = answer.strip().splitlines()[-1] if answer.strip() else ""
    candidates = boxed + re.findall(NUM, last_line)
    if any(abs(v - float(gold)) < 1e-6 for v in map(safe_float, candidates) if v is not None):
        return "right value, not the one scored"
    return "arithmetic slip" if arithmetic_slip(answer) else "wrong reasoning"


def math500_category(row):
    if not row["ended"]:
        return "cut, looping" if looping(row["answer"]) else "cut, long"
    if "\\boxed" not in row["answer"]:
        return "no boxed answer"
    return "wrong answer"


def sandbox(folder, command):
    # same limits as reward.py: memory and file-size caps, small env, temp HOME and TMPDIR in the answer's folder
    env = {"PATH": os.environ["PATH"], "HOME": folder, "TMPDIR": folder, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    try:
        return subprocess.run(["sh", "-c", f"ulimit -v {MEMORY_KB}; ulimit -f {FILE_BLOCKS}; exec {command}"], cwd=folder,
                              env=env, capture_output=True, timeout=TIMEOUT).stdout.decode(errors="replace")
    except subprocess.TimeoutExpired:
        return "RESULT timeout"


def evalplus_exception(solution, entry_point, failing):
    # runs the solution on its first failing input: an exception name, "timeout" or "wrong output"
    # its own example assert failing at import is wrong output. a renamed function is its own category
    driver = ("import sys\nsys.setrecursionlimit(10000)\n"
              "try:\n    from solution import *\nexcept AssertionError:\n    print('RESULT wrong output'); sys.exit()\n"
              "except BaseException as e:\n    print('RESULT import ' + type(e).__name__); sys.exit()\n"
              f"if '{entry_point}' not in globals():\n    print('RESULT wrong function name'); sys.exit()\n"
              "tests = eval(open('input.txt').read())\n"
              "if not tests:\n    print('RESULT no failing input saved'); sys.exit()\n"
              f"try:\n    {entry_point}(*tests[0])\n    print('RESULT wrong output')\n"
              "except BaseException as e:\n    print('RESULT ' + type(e).__name__)\n")
    # evalplus saves the failing inputs as a string in some versions and as a list in others
    failing = failing if isinstance(failing, str) else repr(failing)
    with tempfile.TemporaryDirectory() as folder:
        for name, text in [("solution.py", solution), ("driver.py", driver), ("input.txt", failing)]:
            with open(f"{folder}/{name}", "w") as f:
                f.write(text)
        out = sandbox(folder, f"{sys.executable} driver.py")
    found = re.findall(r"RESULT (.*)", out)
    return found[-1].strip() if found else "crash"


def evalplus_category(bench, row, status):
    if not row["solution"].strip():
        return "no code"
    try:
        ast.parse(row["solution"])
    except SyntaxError:
        return "syntax error"
    if "```" not in row["raw"] and len(tokenizer(row["raw"]).input_ids) >= TEST_MAX - 1:
        return "cut"
    if status["base_status"] == "pass":
        return "edge case (fails only the extra tests)"
    if status["base_status"] == "timeout":
        return "timeout"
    entry = problems[bench][row["task_id"]]["entry_point"]
    return evalplus_exception(row["solution"], entry, status["base_fail_tests"])


def heldout_category(row, test, function_name):
    # (category, fraction of tests passed when run again or None, first error line)
    if not row["ended"]:
        return "cut", None, None
    code = extract_code(row["answer"], function_name)
    if code is None:
        return ("wrong function name" if re.search(r"```", row["answer"]) else "no code"), None, None
    try:
        ast.parse(code)
    except SyntaxError:
        return "syntax error", None, None
    with tempfile.TemporaryDirectory() as folder:
        with open(f"{folder}/solution.py", "w") as f:
            f.write(code)
        with open(f"{folder}/test_solution.py", "w") as f:
            f.write(TEST_HEADER + test)
        out = sandbox(folder, f"{sys.executable} -m pytest -q --tb=short --noconftest -p no:cacheprovider "
                              f"--basetemp={folder}/tmp test_solution.py")
    if out == "RESULT timeout":
        return "timeout", None, None
    lines = out.strip().splitlines()
    rerun = fraction_passed(lines[-1]) if lines else 0.0
    if rerun == 1.0:
        return "passes on rerun", rerun, None
    kinds = collections.Counter(re.findall(r"^E\s+(\w+(?:Error|Exception|Exit|Interrupt))", out, re.M))
    other = [k for k, _ in kinds.most_common() if k != "AssertionError"]
    kind = other[0] if other else "wrong output"
    detail = re.search(rf"^E\s+({kind}.*)$", out, re.M) if other else None
    return kind + (", some tests pass" if row["reward"] > 0 else ""), rerun, detail.group(1)[:120] if detail else None


def analyse(label, folder):
    rows = []
    for i, r in enumerate(json.loads(line) for line in open(f"{folder}/gsm8k.jsonl")):
        gt_style = bool(re.search(r"The answer is \\boxed\{[^{}]*\}\.?\s*$", r["answer"].strip()))
        tokens = len(tokenizer(r["answer"]).input_ids)
        rows.append({"bench": "gsm8k", "id": i, "right": r["right"], "gt_style": gt_style, "tokens": tokens,
                     "category": None if r["right"] else gsm8k_category(r["answer"], r["gold"])})

    for bench in ["mbpp", "humaneval"]:
        status = json.load(open(f"{folder}/{bench}_eval_results.json"))["eval"]
        answers = {json.loads(line)["task_id"]: json.loads(line) for line in open(f"{folder}/{bench}.jsonl")}
        tasks = list(problems[bench])
        wrong = [t for t in tasks if not status[t][0]["base_status"] == status[t][0]["plus_status"] == "pass"]
        with ThreadPoolExecutor(WORKERS) as pool:
            cats = dict(zip(wrong, pool.map(lambda t: evalplus_category(bench, answers[t], status[t][0]), wrong)))
        rows += [{"bench": bench + "+", "id": t, "right": t not in cats, "category": cats.get(t)} for t in tasks]

    if os.path.exists(f"{folder}/math500.jsonl"):
        for r in (json.loads(line) for line in open(f"{folder}/math500.jsonl")):
            rows.append({"bench": "math500", "id": r["unique_id"], "right": r["right"], "subject": r["subject"],
                         "level": r["level"], "category": None if r["right"] else math500_category(r)})

    if os.path.exists(f"{folder}/heldout_code.jsonl"):
        from common import HELD_OUT, kodcode  # noqa: E402
        kod = kodcode().select(HELD_OUT)
        held = [json.loads(line) for line in open(f"{folder}/heldout_code.jsonl")]
        wrong = [i for i, r in enumerate(held) if r["reward"] < 1]
        with ThreadPoolExecutor(WORKERS) as pool:
            cats = dict(zip(wrong, pool.map(lambda i: heldout_category(held[i], kod[i]["test"],
                                                                       kod[i]["test_info"][0]["function_name"]), wrong)))
        rows += [{"bench": "heldout_code", "id": r["question_id"], "right": r["reward"] == 1, "reward": r["reward"],
                  "subset": r["subset"], "difficulty": r["difficulty"], "category": cats[i][0] if i in cats else None,
                  "rerun": cats[i][1] if i in cats else None, "detail": cats[i][2] if i in cats else None,
                  "chars": len(r["answer"])} for i, r in enumerate(held)]

    for r in rows:
        r["run"] = label
        r["recipe"] = re.sub(r"_s\d$", "", label)
    print(f"{label}: {sum(not r['right'] for r in rows)} wrong answers")
    return rows


def table(rows, bench, key):
    # categories x recipes: wrong answers per category, mean over the recipe's runs
    sel = [r for r in rows if r["bench"] == bench and not r["right"]]
    groups = sorted({r[key] for r in rows if r["bench"] == bench})
    runs = {g: len({r["run"] for r in rows if r["bench"] == bench and r[key] == g}) for g in groups}
    n = {g: sum(1 for r in rows if r["bench"] == bench and r[key] == g) / runs[g] for g in groups}
    counts = collections.Counter((r["category"], r[key]) for r in sel)
    cats = sorted({r["category"] for r in sel}, key=lambda c: -sum(counts[(c, g)] for g in groups))
    lines = [f"| {bench} | " + " | ".join(groups) + " |", "|---" * (len(groups) + 1) + "|",
             "| questions | " + " | ".join(f"{n[g]:.0f}" for g in groups) + " |",
             "| right | " + " | ".join(f"{sum(1 for r in rows if r['bench'] == bench and r[key] == g and r['right']) / runs[g]:.1f}" for g in groups) + " |"]
    lines += [f"| {c} | " + " | ".join(f"{counts[(c, g)] / runs[g]:.1f}" for g in groups) + " |" for c in cats]
    return "\n".join(lines)


def gsm8k_style(rows, key):
    # share of answers in the ground truth style and accuracy in / out of it (the copied short style)
    out = ["| gsm8k style | gt-style answers | right if gt-style | right otherwise | median tokens |", "|---|---|---|---|---|"]
    for g in sorted({r[key] for r in rows}):
        sel = [r for r in rows if r["bench"] == "gsm8k" and r[key] == g]
        gt, other = [r for r in sel if r["gt_style"]], [r for r in sel if not r["gt_style"]]
        acc = lambda x: f"{100 * sum(r['right'] for r in x) / len(x):.1f}%" if x else "-"
        out.append(f"| {g} | {100 * len(gt) / len(sel):.1f}% | {acc(gt)} | {acc(other)} | {sorted(r['tokens'] for r in sel)[len(sel) // 2]} |")
    return "\n".join(out)


if __name__ == "__main__":
    rows = [r for label, folder in RESULTS.items() for r in analyse(label, folder)]
    with open("errors.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)

    md = ["# error analysis\n", "Wrong answers per category, mean over each recipe's seeds.\n"]
    for bench in ["gsm8k", "math500", "mbpp+", "humaneval+", "heldout_code"]:
        if any(r["bench"] == bench for r in rows):
            md += [table(rows, bench, "recipe"), ""]
    md += [gsm8k_style(rows, "recipe"), "", "Per run:\n"]
    for bench in ["gsm8k", "math500", "mbpp+", "humaneval+", "heldout_code"]:
        if any(r["bench"] == bench for r in rows):
            md += [table(rows, bench, "run"), ""]
    with open("errors.md", "w") as f:
        f.write("\n".join(md))
    print("\n".join(md[:12]))
