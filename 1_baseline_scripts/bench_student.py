# benchmark the student before any distillation on math (gsm8k test) and python (mbpp+, humaneval+). greedy decoding, pass@1.
# needs bench_utils.py next to this file and: pip install evalplus

import torch
from huggingface_hub import create_repo, upload_folder
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench_utils import benchmark

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR = "results_student"
HUB_RESULTS = "jajamken/FML_project"  # e.g. "jayeshk7/mtkd-results" (private hf dataset). empty = local files only
MAX_QUESTIONS = 2000  # more than any benchmark has (gsm8k test 1319, mbpp+ 378, humaneval+ 164), so all are used. 16 for a quick test

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()

benchmark(model, tokenizer, MODEL, OUT_DIR, MAX_QUESTIONS)

if HUB_RESULTS:
    create_repo(HUB_RESULTS, repo_type="dataset", private=True, exist_ok=True)
    upload_folder(repo_id=HUB_RESULTS, repo_type="dataset", folder_path=OUT_DIR, path_in_repo=OUT_DIR)
