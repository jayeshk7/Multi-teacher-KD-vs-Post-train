# held-out code again with one formatting artifact fixed: answers that open a code block with ``` and then write a bare
# "python" line (the code teacher does this in 237 / 2527 answers, KD / T-SFT students inherit it). the bare line
# becomes the first statement of the code, a NameError, so every test fails. here it is turned into the block's language
# tag and only those answers are run again (reward.py's sandbox). analysis only: reported scores stay unchanged.
# needs the run folders with the answer text (not in git): run from analysis/ inside that folder.
# writes stray_python_rescore.json

import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "..")
from common import HELD_OUT, kodcode  # noqa: E402
from reward import WORKERS, score  # noqa: E402

STRAY = re.compile(r"```[a-z]*[ \t]*\n[ \t]*python[ \t]*\n")


def ada_folder(run):
    # the run's own results if it ran on the ada, else its ada re-eval
    own = f"../{run}/results_best"
    return own if "Ada" in json.load(open(f"{own}/summary.json")).get("gpu", "") else f"../eval_ada/{run}_best/results"


FOLDERS = {"instruct": "../eval_instruct/results", "code teacher": "../eval_teacher_code/results"}
FOLDERS.update({r: ada_folder(r) for r in sorted(os.path.basename(p) for p in glob.glob("../inst_*_s[0-9]"))})

kod = kodcode().select(HELD_OUT)
tests = [ex["test"] for ex in kod]
names = [ex["test_info"][0]["function_name"] for ex in kod]

out = {}
for label, folder in FOLDERS.items():
    rows = [json.loads(line) for line in open(f"{folder}/heldout_code.jsonl")]
    idx = [i for i, r in enumerate(rows) if STRAY.search(r["answer"])]
    fixed = [STRAY.sub("```python\n", rows[i]["answer"]) for i in idx]
    with ThreadPoolExecutor(WORKERS) as pool:
        new = list(pool.map(score, fixed, [tests[i] for i in idx], [names[i] for i in idx]))
    old_mean = sum(r["reward"] for r in rows) / len(rows)
    rewards = [r["reward"] for r in rows]
    for i, v in zip(idx, new):
        rewards[i] = v
    out[label] = {"n_stray": len(idx), "old": old_mean, "fixed": sum(rewards) / len(rewards),
                  "stray_old_mean": sum(rows[i]["reward"] for i in idx) / len(idx) if idx else None,
                  "stray_fixed_mean": sum(new) / len(new) if new else None}
    print(label, out[label], flush=True)

json.dump(out, open("stray_python_rescore.json", "w"), indent=1)
