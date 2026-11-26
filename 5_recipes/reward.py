# code score: the fraction of kodcode's unit tests that the answer's code passes (grpo reward and held-out test).
# the code comes from the model, so every answer runs in its own process with limits

import os
import re
import subprocess
import sys
import tempfile

TIMEOUT = 10            # seconds per answer, stops infinite loops
MEMORY_KB = 4_000_000   # address space per answer, stops runaway memory
# largest file a test may write (256 mib in 512-byte blocks). one kodcode test writes 2 gib, a bigger write fails
FILE_BLOCKS = 524_288
# parallel test processes per run. kept at 32 because several runs share the server's cpu, and a busy cpu can make
# tests hit the 10 s limit
WORKERS = min(32, os.cpu_count())

# some kodcode test files never import the function, so import everything from the solution.
# test functions inside the answer are deleted, else pytest would count them as passed
TEST_HEADER = ("from solution import *\n"
               "for _n in [n for n in dir() if n.lower().startswith('test')]:\n"
               "    del globals()[_n]\n")


def extract_code(answer, function_name):
    # last code block that defines the function (later blocks are often just example usage)
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

        # pytest's temp files stay in the folder, so they are deleted with it, also after a timeout
        command = (f"ulimit -v {MEMORY_KB}; ulimit -f {FILE_BLOCKS}; exec {sys.executable} -m pytest -q --tb=no "
                   f"--noconftest -p no:cacheprovider --basetemp={folder}/tmp test_solution.py")
        # small env and a temp HOME, so the code can't read the hf token
        env = {"PATH": os.environ["PATH"], "HOME": folder, "TMPDIR": folder, "OMP_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1"}
        try:
            out = subprocess.run(["sh", "-c", command], cwd=folder, env=env, capture_output=True, timeout=TIMEOUT).stdout
        except subprocess.TimeoutExpired:
            return ""

    lines = out.decode(errors="replace").strip().splitlines()
    return lines[-1] if lines else ""


def fraction_passed(summary):
    # "1 failed, 4 passed" -> 0.8
    counts = {name: int(n) for n, name in re.findall(r"(\d+) (passed|failed|errors?)", summary)}
    total = sum(counts.values())
    return counts.get("passed", 0) / total if total else 0.0


def score(answer, test, function_name):
    # 0 if the answer has no code block that defines the function
    code = extract_code(answer, function_name)
    if code is None:
        return 0.0
    return fraction_passed(run_tests(code, test))
