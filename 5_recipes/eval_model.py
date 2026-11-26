# val scores and all test sets for one model: a hub name, a model folder or a run's adapter (<run>/best).
# run: MODEL=Qwen/Qwen2.5-0.5B-Instruct OUT=eval_instruct python -u eval_model.py

import json
import os

import torch
from transformers import AutoTokenizer

from common import TOKENIZER, for_generation, load_model, rl_rows, test_eval, val_scores

MODEL = os.environ["MODEL"]
OUT = os.environ["OUT"]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# the 7b teachers load in bf16; a 0.5b model is merged in fp32 first, then copied to bf16 for generation
DTYPE = torch.bfloat16 if "7B" in MODEL else torch.float32

os.makedirs(OUT, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
tokenizer.padding_side = "left"
model = for_generation(load_model(MODEL, DEVICE, DTYPE)).eval()

scores = val_scores(model, tokenizer, {d: rl_rows(d, "val") for d in ["math", "code"]})
print(scores)
with open(f"{OUT}/val.json", "w") as f:
    json.dump(scores, f)

test_eval(model, tokenizer, MODEL, f"{OUT}/results")
