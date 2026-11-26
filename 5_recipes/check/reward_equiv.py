# old vs new code reward on the same answers (17 nov change: file-size cap, pytest tmp dirs and TMPDIR inside the
# per-answer folder). answers: kodcode's reference solution of all 10000 rows, and instruct's 2527 held-out answers.
# each answer is scored by both versions back to back in the same thread, so both see the same cpu load. answers whose
# scores differ are scored 3 more times by each version: a flaky test differs within a version too
# run: python3 check/reward_equiv.py check/reward_before_17nov.py reward.py.new
import importlib.machinery
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import common

WORKERS = 12   # the server's cpu is shared (load ~90 of 96 cores on 17 nov)


def load(path, name):
    # SourceFileLoader also takes the .new file
    loader = importlib.machinery.SourceFileLoader(name, path)
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
    loader.exec_module(module)
    return module


old, new = load(sys.argv[1], "reward_old"), load(sys.argv[2], "reward_new")
kod = common.kodcode()
fn = [t[0]["function_name"] for t in kod["test_info"]]
items = [(f"ref_{i}", f"```python\n{s}\n```", t, f) for i, (s, t, f) in enumerate(zip(kod["solution"], kod["test"], fn))]
row = {q: i for i, q in enumerate(kod["question_id"])}
for r in map(json.loads, open("eval_instruct/results/heldout_code.jsonl")):
    i = row[r["question_id"]]
    items.append((f"instruct_{i}", r["answer"], kod[i]["test"], fn[i]))


def both(item):
    name, answer, test, f = item
    return old.score(answer, test, f), new.score(answer, test, f)


with ThreadPoolExecutor(WORKERS) as pool:
    scores = list(pool.map(both, items))
differ = [(it, s) for it, s in zip(items, scores) if s[0] != s[1]]
print(f"{len(items)} answers, {len(items) - len(differ)} same score, {len(differ)} differ")

with ThreadPoolExecutor(WORKERS) as pool:
    again = list(pool.map(lambda it: [both(it[0]) for _ in range(3)], differ))
with open("check/reward_equiv.jsonl", "w") as out:
    for (it, s), more in zip(differ, again):
        olds, news = [s[0]] + [m[0] for m in more], [s[1]] + [m[1] for m in more]
        flaky = len(set(olds)) > 1 or len(set(news)) > 1
        kind = "flaky" if flaky else "changed"
        print(f"  {it[0]:16s} old {olds} new {news} -> {kind}")
        out.write(json.dumps({"item": it[0], "old": olds, "new": news, "kind": kind}) + "\n")
