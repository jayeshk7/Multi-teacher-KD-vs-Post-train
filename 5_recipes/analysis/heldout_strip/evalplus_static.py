# mbpp+ / humaneval+: how often the raw answer runs code at import, and whether evalplus's "solution" (what is tested)
# still does. needs the run folders with the answer text (DATA, not in git)
import ast, json

DATA = "/home/jayeshk7/Desktop/IITB_coursework/Sem3/FML/project/stage2_gpu1_mirror"


def runs(code):
    # true if any top-level statement runs at import (same skip rules as static.py), None if it does not parse
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and not any(isinstance(n, ast.Call) for n in ast.walk(node)):
            continue
        return True
    return False


def code_of(raw):
    # the raw answer without the code fences
    raw = raw.replace("```python", "").replace("```", "")
    return raw


for name, f in [("instruct", f"{DATA}/eval_instruct/results"), ("multi KD s0", f"{DATA}/inst_multi_kd_s0/results_best"),
                ("multi GRPO s0", f"{DATA}/inst_multi_rl_s0/results_best")]:
    for bench in ["mbpp", "humaneval"]:
        rows = [json.loads(l) for l in open(f"{f}/{bench}.jsonl")]
        raw = [runs(code_of(r["raw"])) for r in rows]
        sol = [runs(r["solution"]) for r in rows]
        print(f"{name:14s} {bench:9s} n {len(rows)}  raw answer runs code at import {100*sum(bool(x) for x in raw)/len(rows):5.1f}%   tested solution {100*sum(bool(x) for x in sol)/len(rows):4.1f}%")
