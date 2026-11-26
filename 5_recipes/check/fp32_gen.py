# fp32 check: instruct's gsm8k test generated exactly as its eval did (bench_utils.generate: same
# prompts, batches and order, greedy) but in fp32: no bf16 copy, no autocast, no tf32. once per gpu type. if the two
# gpu types then give (nearly) the same answers, the bf16 near-ties (check A, near_tie.py) are why answers differ
# run from 5_recipes/: CUDA_VISIBLE_DEVICES=3 python3 check/fp32_gen.py (~1 h, ~10 gb)
# N=16 for a quick test (other batches, so its answers are not expected to match the saved ones)
import json
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, ".")
import bench_utils
from bench_utils import MATH_SYSTEM, chat, is_right
from common import INSTRUCT, TOKENIZER, load_model

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
N = int(os.environ.get("N", 1319))
GPU = torch.cuda.get_device_name(0)
TAG = "ada" if "Ada" in GPU else "a6000"
OTHER = "a6000" if TAG == "ada" else "ada"
BF16 = {"ada": "eval_instruct/results/gsm8k.jsonl", "a6000": "eval_instruct_a6000/results/gsm8k.jsonl"}
OUT = f"check/fp32/{TAG}{'' if N == 1319 else '_test'}.jsonl"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
tokenizer.padding_side = "left"
model = load_model(INSTRUCT, "cuda:0", torch.float32).eval()

gsm = load_dataset("openai/gsm8k", "main", split="test").select(range(N))
t0 = time.time()
answers = bench_utils.generate(model, tokenizer, [chat(tokenizer, MATH_SYSTEM, ex["question"]) for ex in gsm])
golds = [ex["answer"].split("####")[-1].strip().replace(",", "") for ex in gsm]
right = [is_right(a, g) for a, g in zip(answers, golds)]

os.makedirs("check/fp32", exist_ok=True)
with open(OUT, "w") as f:
    for r, g, a in zip(right, golds, answers):
        f.write(json.dumps({"right": r, "gold": g, "answer": a}) + "\n")

bf16 = [json.loads(l)["answer"] for l in open(BF16[TAG])][:N]
print(f"{GPU}: {N} questions, {time.time() - t0:.0f} s, weights {model.dtype}, tf32 {torch.backends.cuda.matmul.allow_tf32}")
print(f"fp32 right {sum(right)} / {N}; same answer as this gpu type's bf16 eval {sum(a == b for a, b in zip(answers, bf16))} / {N}")
if N == 1319 and os.path.exists(f"check/fp32/{OTHER}.jsonl"):
    other = [json.loads(l) for l in open(f"check/fp32/{OTHER}.jsonl")]
    same = sum(a == o["answer"] for a, o in zip(answers, other))
    flips = sum(r != o["right"] for r, o in zip(right, other))
    print(f"fp32 {TAG} vs {OTHER}: same answer {same} / 1319, right {sum(right)} vs {sum(o['right'] for o in other)}, "
          f"right / wrong flips {flips} (bf16: 474 / 1319 same, LOG 17 Nov 09:15)")
