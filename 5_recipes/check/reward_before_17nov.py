# rlvr reward for code: the fraction of kodcode's unit tests that the answer's code passes.
# the code comes from the model, so every answer runs in its own process with limits.

import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

TIMEOUT = 10            # seconds per answer, stops infinite loops
MEMORY_KB = 4_000_000   # address space per answer, stops runaway memory
# checked 16 nov with another user keeping ~84 of 96 cores busy: kodcode's own solutions for 300 held-out rows pass
# the same 293/300 with 8 or 96 workers (32 s vs 5 s). capped at 32 per process because 2-3 runs share a gpu (and the
# cpu): 3 x 96 test processes at once would push heavier tests toward the 10 s limit
WORKERS = min(32, os.cpu_count())

# 12% of kodcode's test files never import the function, a few use modules only the solution imports.
# test functions in the answer are dropped, pytest would count them as passed
TEST_HEADER = ("from solution import *\n"
               "for _n in [n for n in dir() if n.lower().startswith('test')]:\n"
               "    del globals()[_n]\n")


def extract_code(answer, function_name):
    # last code block that defines the function, later blocks are often just example usage
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", answer, re.S)
    defining = [b for b in blocks if re.search(rf"\bdef\s+{re.escape(function_name)}\b", b)]
    return defining[-1] if defining else None


def run_tests(code, test):
    # returns pytest's summary line, e.g. "1 failed, 4 passed in 0.02s". empty means it was killed
    with tempfile.TemporaryDirectory() as folder:
        with open(f"{folder}/solution.py", "w") as f:
            f.write(code)
        with open(f"{folder}/test_solution.py", "w") as f:
            f.write(TEST_HEADER + test)

        command = (f"ulimit -v {MEMORY_KB}; exec {sys.executable} -m pytest -q --tb=no --noconftest "
                   "-p no:cacheprovider test_solution.py")
        # small env and a temp HOME, so the code can't read the hf token
        env = {"PATH": os.environ["PATH"], "HOME": folder, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        try:
            out = subprocess.run(["sh", "-c", command], cwd=folder, env=env, capture_output=True, timeout=TIMEOUT).stdout
        except subprocess.TimeoutExpired:
            return ""

    lines = out.decode(errors="replace").strip().splitlines()
    return lines[-1] if lines else ""


def fraction_passed(summary):
    counts = {name: int(n) for n, name in re.findall(r"(\d+) (passed|failed|errors?)", summary)}
    total = sum(counts.values())
    return counts.get("passed", 0) / total if total else 0.0


def score(answer, test, function_name):
    code = extract_code(answer, function_name)
    if code is None:
        return 0.0
    return fraction_passed(run_tests(code, test))


def code_reward(completions, test, function_name, **kwargs):
    # grpo calls this with one row per sampled answer. test and function_name are dataset columns
    answers = [c[0]["content"] for c in completions]
    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(score, answers, test, function_name))
