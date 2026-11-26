# static look at held-out kodcode answers: does the code block run anything at import (own asserts, example calls,
# prints), and does it use typing names it never imports. per recipe, 3 seeds pooled.
# needs the run folders with the answer text (DATA, not in git)
import ast, json, sys, os, collections
sys.path.insert(0, "/home/jayeshk7/Desktop/IITB_coursework/Sem3/FML/project/Multi-teacher-KD-vs-Post-train/5_recipes")
from reward import extract_code
from datasets import load_dataset

DATA = "/home/jayeshk7/Desktop/IITB_coursework/Sem3/FML/project/stage2_gpu1_mirror"
kod = load_dataset("KodCode/KodCode-Light-RL-10K", split="train").shuffle(seed=0).select(range(7473, 10000))
FN = [ex["test_info"][0]["function_name"] for ex in kod]
NAMES = [("instruct", None), ("multi GT-SFT", "multi_gt_sft"), ("multi T-SFT", "multi_t_sft"), ("multi KD", "multi_kd"),
         ("KD code", "kd_code"), ("KD math → GRPO code", "seq_kd_math_then_rl_code"), ("multi GRPO", "multi_rl")]


def folder(key, s):
    # results on the ada: the run's own if it ran there, else its ada re-eval
    if key is None:
        return f"{DATA}/eval_instruct/results"
    own = f"{DATA}/inst_{key}_s{s}/results_best"
    return own if "Ada" in json.load(open(f"{own}/summary.json")).get("gpu", "") else f"{DATA}/eval_ada/inst_{key}_s{s}_best/results"


TYPING = {"List", "Dict", "Tuple", "Optional", "Set", "Union", "Any", "Callable"}


def kind(code):
    # ("runs code at import" or "clean", uses a typing name it never imports)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "syntax", False
    runs = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue      # docstring
        if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
            continue      # main guard: not run on import
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and not any(isinstance(n, ast.Call) for n in ast.walk(node)):
            continue      # constants
        runs = True
    imported = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    star_typing = any(isinstance(n, ast.ImportFrom) and n.module == "typing" and any(a.name == "*" for a in n.names) for n in ast.walk(tree))
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} & TYPING
    missing = bool(used - imported) and not star_typing
    return ("runs code at import" if runs else "clean"), missing


out = {}
for label, key in NAMES:
    c = collections.Counter()
    rew = collections.defaultdict(list)
    for s in ([0] if key is None else [0, 1, 2]):
        rows = [json.loads(l) for l in open(f"{folder(key, s)}/heldout_code.jsonl")]
        for r, fn in zip(rows, FN):
            code = extract_code(r["answer"], fn)
            if code is None:
                k, miss = "no code block", False
            else:
                k, miss = kind(code)
            g = k + (" + typing name not imported" if miss else "")
            c[g] += 1
            rew[g].append(r["reward"])
    n = sum(c.values())
    out[label] = {g: (round(100 * c[g] / n, 1), round(100 * sum(rew[g]) / len(rew[g]), 1)) for g in c}
    print(f"{label:22s}", "; ".join(f"{g}: {p}% of answers, score {sc}" for g, (p, sc) in sorted(out[label].items(), key=lambda x: -x[1][0])))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "static.json"), "w"), indent=1)
