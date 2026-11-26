# held-out kodcode scored again the way evalplus treats mbpp+ / humaneval+ answers: code that runs at import (own asserts,
# example calls, prints, a stray "python" line) is dropped before the tests, for the recipes rescore.py did not finish.
# also measures what adding "from typing import *" gains. only answers whose code changes run again.
# needs the run folders with the answer text (DATA, not in git). 4 workers
import ast, json, sys, os
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/home/jayeshk7/Desktop/IITB_coursework/Sem3/FML/project/Multi-teacher-KD-vs-Post-train/5_recipes")
from reward import extract_code, run_tests, fraction_passed
from datasets import load_dataset

DATA = "/home/jayeshk7/Desktop/IITB_coursework/Sem3/FML/project/stage2_gpu1_mirror"
kod = load_dataset("KodCode/KodCode-Light-RL-10K", split="train").shuffle(seed=0).select(range(7473, 10000))
FN = [ex["test_info"][0]["function_name"] for ex in kod]
TEST = [ex["test"] for ex in kod]
NAMES = [("instruct", None), ("multi GT-SFT", "multi_gt_sft"), ("multi T-SFT", "multi_t_sft"), ("multi KD", "multi_kd"),
         ("KD math → GRPO code", "seq_kd_math_then_rl_code"), ("multi GRPO", "multi_rl")]
DONE = {"instruct", "multi GT-SFT", "multi T-SFT"}   # strip-only totals already in rescore_part1.log
TYPING = {"List", "Dict", "Tuple", "Optional", "Set", "Union", "Any", "Callable"}


def folder(key, s):
    # results on the ada: the run's own if it ran there, else its ada re-eval
    if key is None:
        return f"{DATA}/eval_instruct/results"
    own = f"{DATA}/inst_{key}_s{s}/results_best"
    return own if "Ada" in json.load(open(f"{own}/summary.json")).get("gpu", "") else f"{DATA}/eval_ada/inst_{key}_s{s}_best/results"


def strip(code):
    # keep only imports, functions, classes and constant assignments
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            keep.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and not any(isinstance(n, ast.Call) for n in ast.walk(node)):
            keep.append(node)
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


def job(args):
    # fraction of tests passed by one (row, code) pair
    i, code = args
    return fraction_passed(run_tests(code, TEST[i]))


def missing_typing(code):
    # uses List, Dict, ... without importing them
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    imported = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    return bool(({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} & TYPING) - imported) and "*" not in imported


out = {}
for label, key in NAMES:
    for s in ([0] if key is None else [0, 1, 2]):
        rows = [json.loads(l) for l in open(f"{folder(key, s)}/heldout_code.jsonl")]
        n = len(rows)
        old = [r["reward"] for r in rows]
        strip_jobs, typ = [], []
        for i, (r, fn) in enumerate(zip(rows, FN)):
            code = extract_code(r["answer"], fn) if r["ended"] else None
            if code is None:
                continue
            c1 = strip(code)
            try:
                same = ast.dump(ast.parse(c1)) == ast.dump(ast.parse(code))
            except SyntaxError:
                same = True
            if label not in DONE and not same:
                strip_jobs.append((i, c1))
            if missing_typing(c1):
                typ.append(i)
        codes = {}
        for i, (r, fn) in enumerate(zip(rows, FN)):
            if i in typ:
                codes[i] = strip(extract_code(r["answer"], fn))
        with ThreadPoolExecutor(4) as pool:
            v1 = list(old)
            for (i, _), sc in zip(strip_jobs, pool.map(job, strip_jobs)):
                v1[i] = sc
            a = [(i, codes[i]) for i in typ]
            b = [(i, "from typing import *\n" + codes[i]) for i in typ]
            s1 = list(pool.map(job, a))
            s2 = list(pool.map(job, b))
        res = {"stripped": 100 * sum(v1) / n if label not in DONE else None, "typing_rows": len(typ),
               "typing_gain_pp": 100 * (sum(s2) - sum(s1)) / n, "changed": len(strip_jobs)}
        out[f"{label} {s}"] = res
        print(label, s, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rescore2.json"), "w"), indent=1)
print("done", flush=True)
