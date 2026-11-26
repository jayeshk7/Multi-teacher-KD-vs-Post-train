# fp32 logit check: fp32 answers are identical on both gpu types (fp32_gen.py). are the logits
# bitwise equal, or only close? the first N gsm8k questions, prompt + the fp32 answer (same text on both gpus) in one
# forward pass, batch 1, in fp32 (no autocast, tf32 off) and in bf16 (the eval's model, for_generation under autocast).
# per answer position: top-20 logits + ids and a fingerprint of the whole logit row (any bit differs -> it differs).
# once per gpu type; the second run compares. run from 5_recipes/: CUDA_VISIBLE_DEVICES=3 python3 check/fp32_logits.py
import json
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from bench_utils import MATH_SYSTEM, chat
from common import INSTRUCT, TOKENIZER, for_generation, load_model

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
N = int(os.environ.get("N", 8))
GPU = torch.cuda.get_device_name(0)
TAG = "ada" if "Ada" in GPU else "a6000"
OTHER = "a6000" if TAG == "ada" else "ada"
OUT = f"check/fp32_logits/{TAG}.pt"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
gsm = load_dataset("openai/gsm8k", "main", split="test").select(range(N))
answers = [json.loads(l)["answer"] for l in open("check/fp32/ada.jsonl")][:N]
# int64 weights for the row fingerprint: sum of (bit pattern x weight), wraps mod 2 ^ 64, so the order of the sum
# does not matter and any changed bit changes it (barring a ~2 ^ -64 collision)
VOCAB = 151936
weights = torch.randint(1, 2 ** 31, (VOCAB,), generator=torch.Generator().manual_seed(0), dtype=torch.int64)


@torch.no_grad()
def run(model, autocast):
    out = []
    w = weights.to(model.device)
    for ex, ans in zip(gsm, answers):
        p = tokenizer(chat(tokenizer, MATH_SYSTEM, ex["question"]), add_special_tokens=False)["input_ids"]
        a = tokenizer(ans, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p + a], device=model.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            logits = model(input_ids=ids).logits[0, len(p) - 1:-1]   # the positions that predict the answer tokens
        dtype = str(logits.dtype)
        x = logits.float()   # bf16 -> fp32 is exact, so the fingerprint sees the bf16 bits too
        fp = (x.contiguous().view(torch.int32).to(torch.int64) * w[:x.shape[1]]).sum(-1)
        top = x.topk(20, dim=-1)
        out.append({"next": torch.tensor(a), "top_v": top.values.cpu(), "top_i": top.indices.cpu(), "fp": fp.cpu(),
                    "dtype": dtype})
    return out


res = {"gpu": GPU}
res["fp32"] = run(load_model(INSTRUCT, "cuda:0", torch.float32).eval(), False)
res["bf16"] = run(for_generation(load_model(INSTRUCT, "cuda:0", torch.float32)).eval(), True)
os.makedirs("check/fp32_logits", exist_ok=True)
torch.save(res, OUT)

for k in ["fp32", "bf16"]:
    rows = res[k]
    pos = sum(len(r["next"]) for r in rows)
    hit = sum((r["top_i"][:, 0] == r["next"]).sum().item() for r in rows)
    print(f"{GPU} {k}: logits {rows[0]['dtype']}, {N} answers, {pos} positions, top-1 = answer token {hit}/{pos}")

if os.path.exists(f"check/fp32_logits/{OTHER}.pt"):
    other = torch.load(f"check/fp32_logits/{OTHER}.pt")
    for k in ["fp32", "bf16"]:
        a, b = res[k], other[k]
        pos = sum(len(r["fp"]) for r in a)
        same_row = sum((x["fp"] == y["fp"]).sum().item() for x, y in zip(a, b))
        same_top1 = sum((x["top_i"][:, 0] == y["top_i"][:, 0]).sum().item() for x, y in zip(a, b))
        same_ids = torch.cat([(x["top_i"] == y["top_i"]).all(-1) for x, y in zip(a, b)])
        # value differences where both gpus rank the same top-20 tokens
        d = torch.cat([(x["top_v"] - y["top_v"]).abs()[(x["top_i"] == y["top_i"]).all(-1)] for x, y in zip(a, b)])
        gap = torch.cat([x["top_v"][:, 0] - x["top_v"][:, 1] for x in a])
        print(f"{k} {TAG} vs {OTHER}: whole logit row bitwise equal at {same_row}/{pos} positions, same top-1 "
              f"{same_top1}/{pos}, same top-20 ids {same_ids.sum().item()}/{pos}; top-20 |diff| where ids agree: max "
              f"{d.max().item():.3g}, mean {d.mean().item():.3g}, share exactly 0 {(d == 0).float().mean().item():.4f}; "
              f"top-2 gap ({TAG}) min {gap.min().item():.3g}, median {gap.median().item():.3g}")
