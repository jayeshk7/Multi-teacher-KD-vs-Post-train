# near-tie check: replay instruct's gsm8k test generation exactly as its eval ran (same model path,
# bench_utils.generate: same batches and order, bf16, greedy) and keep each step's top-2 logits. once per gpu type,
# then check/near_tie_compare.py: where the two gpus' answers split, how close were the top two logits?
# run from 5_recipes/: CUDA_VISIBLE_DEVICES=3 python3 check/near_tie.py (~20 min, ~8 gb)
# N=16 for a quick test (other batches, so its answers are not expected to match the saved ones)
import json
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList

sys.path.insert(0, ".")
import bench_utils
from bench_utils import MATH_SYSTEM, chat
from common import EOS, INSTRUCT, TOKENIZER, for_generation, load_model

N = int(os.environ.get("N", 1319))
GPU = torch.cuda.get_device_name(0)
TAG = "ada" if "Ada" in GPU else "a6000"
SAVED = {"ada": "eval_instruct/results/gsm8k.jsonl", "a6000": "eval_instruct_a6000/results/gsm8k.jsonl"}[TAG]
OUT = f"check/near_tie/{TAG}{'' if N == 1319 else '_test'}.pt"


class Record(LogitsProcessor):
    # keeps every step's top-2 logits, passes the scores on unchanged
    def __call__(self, input_ids, scores):
        self.dtype = scores.dtype
        self.steps.append(scores.topk(2, dim=-1))
        return scores


# same loading as eval_model.py: fp32, then the bf16 sampler copy
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
tokenizer.padding_side = "left"
model = for_generation(load_model(INSTRUCT, "cuda:0", torch.float32)).eval()

rec = Record()
batches = []
generate = model.generate


def recorded(**kw):
    rec.steps = []
    out = generate(**kw, logits_processor=LogitsProcessorList([rec]))
    ids = out[:, kw["input_ids"].shape[1]:].cpu()
    v = torch.stack([s.values for s in rec.steps], 1).cpu()
    i = torch.stack([s.indices for s in rec.steps], 1).cpu()
    assert v.shape[:2] == ids.shape, (v.shape, ids.shape)
    batches.append((ids, v, i))
    return out


model.generate = recorded

gsm = load_dataset("openai/gsm8k", "main", split="test")
prompts = [chat(tokenizer, MATH_SYSTEM, s["question"]) for s in gsm][:N]
t0 = time.time()
with torch.autocast("cuda", dtype=torch.bfloat16):   # as common.test_eval
    answers = bench_utils.generate(model, tokenizer, prompts)

# batch rows back to question order, the same sort as bench_utils.generate
order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
rows = [None] * len(prompts)
for b, (ids, v, i) in enumerate(batches):
    for j, q in enumerate(order[b * bench_utils.GEN_BATCH:(b + 1) * bench_utils.GEN_BATCH]):
        row = ids[j].tolist()
        end = next((t + 1 for t, x in enumerate(row) if x in EOS), len(row))
        rows[q] = {"ids": ids[j, :end], "top_v": v[j, :end], "top_i": i[j, :end]}

saved = [json.loads(l)["answer"] for l in open(SAVED)][:N]
same = sum(a == s for a, s in zip(answers, saved))
greedy = sum((r["top_i"][:, 0] == r["ids"]).sum().item() for r in rows)
steps = sum(len(r["ids"]) for r in rows)
bf16 = all(torch.equal(r["top_v"], r["top_v"].bfloat16().float()) for r in rows)
print(f"{GPU}: {N} questions, {time.time() - t0:.0f} s, scores {rec.dtype}, top-2 values bf16-exact {bf16}")
print(f"replay = saved answers ({SAVED}): {same}/{N}; top-1 = generated token at {greedy}/{steps} steps")
os.makedirs("check/near_tie", exist_ok=True)
torch.save({"gpu": GPU, "rows": rows, "answers": answers, "same_as_saved": same, "scores_dtype": str(rec.dtype),
            "bf16_exact": bf16}, OUT)
print("saved", OUT)
