# the slide and report example: held-out row 2358 (Prefill_22874_I), instruct vs multi grpo seed 0, plus instruct's
# code with the extra lines removed. needs the run folders with the answer text (DATA, not in git)
import ast, json, sys
sys.path.insert(0, "/home/jayeshk7/Desktop/IITB_coursework/Sem3/FML/project/Multi-teacher-KD-vs-Post-train/5_recipes")
from reward import extract_code, run_tests
from datasets import load_dataset

DATA = "/home/jayeshk7/Desktop/IITB_coursework/Sem3/FML/project/stage2_gpu1_mirror"
ex = load_dataset("KodCode/KodCode-Light-RL-10K", split="train").shuffle(seed=0).select(range(7473, 10000))[2358]
fn = ex["test_info"][0]["function_name"]
print(ex["question_id"], fn)
print("QUESTION:", ex["question"])
print("TESTS:\n", ex["test"])
for name, f in [("instruct", f"{DATA}/eval_instruct/results"), ("multi GRPO s0", f"{DATA}/inst_multi_rl_s0/results_best")]:
    r = [json.loads(l) for l in open(f"{f}/heldout_code.jsonl")][2358]
    code = extract_code(r["answer"], fn)
    print(name, "reward", r["reward"], "pytest:", run_tests(code, ex["test"]))
ins = [json.loads(l) for l in open(f"{DATA}/eval_instruct/results/heldout_code.jsonl")][2358]
code = extract_code(ins["answer"], fn)
keep = [n for n in ast.parse(code).body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
print("instruct stripped pytest:", run_tests(ast.unparse(ast.Module(body=keep, type_ignores=[])), ex["test"]))
