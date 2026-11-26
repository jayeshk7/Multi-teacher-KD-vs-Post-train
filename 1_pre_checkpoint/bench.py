# benchmark a model before any training on math (gsm8k test) and python (mbpp+, humaneval+). greedy decoding, pass@1.
# run: MODEL=Qwen/Qwen2.5-Math-7B-Instruct python -u bench.py   (default: the student)
# needs bench_utils.py next to this file and: pip install evalplus

import os

import torch
from huggingface_hub import create_repo, upload_folder
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench_utils import benchmark

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
OUT_DIR = {"Qwen/Qwen2.5-0.5B-Instruct": "results_student", "Qwen/Qwen2.5-Math-7B-Instruct": "results_math_teacher",
           "Qwen/Qwen2.5-Coder-7B-Instruct": "results_code_teacher"}[MODEL]
HUB_RESULTS = "jajamken/FML_project"  # e.g. "jayeshk7/mtkd-results" (private hf dataset). empty = local files only
MAX_QUESTIONS = 2000  # more than any benchmark has (gsm8k test 1319, mbpp+ 378, humaneval+ 164), so all are used. 16 for a quick test

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()

benchmark(model, tokenizer, MODEL, OUT_DIR, MAX_QUESTIONS)

if HUB_RESULTS:
    create_repo(HUB_RESULTS, repo_type="dataset", private=True, exist_ok=True)
    upload_folder(repo_id=HUB_RESULTS, repo_type="dataset", folder_path=OUT_DIR, path_in_repo=OUT_DIR)
