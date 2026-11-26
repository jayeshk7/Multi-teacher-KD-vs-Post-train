# what the import-time code is, per recipe (3 seeds pooled, % of all held-out answers).
# needs the run folders with the answer text (DATA, not in git)
import ast, json, sys, collections
sys.path.insert(0, "/home/jayeshk7/Desktop/IITB_coursework/Sem3/FML/project/Multi-teacher-KD-vs-Post-train/5_recipes")
from reward import extract_code
from datasets import load_dataset

DATA = "/home/jayeshk7/Desktop/IITB_coursework/Sem3/FML/project/stage2_gpu1_mirror"
kod = load_dataset("KodCode/KodCode-Light-RL-10K", split="train").shuffle(seed=0).select(range(7473, 10000))
FN = [ex["test_info"][0]["function_name"] for ex in kod]
NAMES = [("instruct", None), ("multi GT-SFT", "multi_gt_sft"), ("multi T-SFT", "multi_t_sft"), ("multi KD", "multi_kd"),
         ("KD math → GRPO code", "seq_kd_math_then_rl_code"), ("multi GRPO", "multi_rl")]


def folder(key, s):
    # results on the ada: the run's own if it ran there, else its ada re-eval
    if key is None:
        return f"{DATA}/eval_instruct/results"
    own = f"{DATA}/inst_{key}_s{s}/results_best"
    return own if "Ada" in json.load(open(f"{own}/summary.json")).get("gpu", "") else f"{DATA}/eval_ada/inst_{key}_s{s}_best/results"


def kinds(code):
    # kinds of top-level code that runs at import (same skip rules as static.py)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    k = set()
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
            continue
        if isinstance(n, ast.If) and "__name__" in ast.unparse(n.test):
            continue
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and not any(isinstance(x, ast.Call) for x in ast.walk(n)):
            continue
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Name) and n.value.id == "python":
            k.add("stray python line")
        elif isinstance(n, ast.Assert) or "assert" in ast.unparse(n) or "check" in ast.unparse(n).lower() or "test" in ast.unparse(n).lower():
            k.add("own tests")
        elif "input(" in ast.unparse(n):
            k.add("reads input()")
        else:
            k.add("example calls / prints")
    return k


for label, key in NAMES:
    c = collections.Counter()
    n = 0
    zero = collections.Counter()
    for s in ([0] if key is None else [0, 1, 2]):
        rows = [json.loads(l) for l in open(f"{folder(key, s)}/heldout_code.jsonl")]
        for r, fn in zip(rows, FN):
            n += 1
            code = extract_code(r["answer"], fn)
            for k in (kinds(code) if code else set()):
                c[k] += 1
                zero[k] += r["reward"] == 0
    print(f"{label:22s}", "; ".join(f"{k} {100*v/n:.1f}% (score 0 in {100*zero[k]/v:.0f}%)" for k, v in c.most_common()))
