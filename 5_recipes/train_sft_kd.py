# one sft or kd stage with a lora adapter, starting from START (a hub model or an earlier stage's checkpoint).
# kd = kl to the teacher on its answers, t_sft = ce on the teacher's answers, gt_sft = ce on the ground truth answers.
# every quarter epoch the student is scored on the val rows of both domains; the best and last adapters are benchmarked
# run: SEED=0 START=Qwen/Qwen2.5-0.5B-Instruct DOMAINS=math,code LOSS=kd OUT=inst_multi_kd_s0 python -u train_sft_kd.py

import json
import os
import random
import time

import torch
import torch.nn.functional as F
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from common import TOKENIZER, Sampler, drift, kd_rows, load_model, lora, rl_rows, test_eval, val_scores
from kd_utils import collate, logits_at

TEACHERS = {"math": "Qwen/Qwen2.5-Math-7B-Instruct", "code": "Qwen/Qwen2.5-Coder-7B-Instruct"}

EPOCHS = 2
BATCH = 32          # prompts per optimizer step, all from one domain
MICRO = 4           # prompts per forward pass
LR = 1e-4           # lora needs a higher rate than full fine-tuning (1e-5 in the october runs)
CHECKS = 4          # val checks per epoch
EVAL_LAST = True    # also benchmark the last checkpoint, to see what picking on val changed

SEED = int(os.environ["SEED"])
START = os.environ["START"]                 # hub name, model folder or <run>/best of an earlier stage
DOMAINS = os.environ["DOMAINS"].split(",")  # math, code or math,code (multi)
LOSS = os.environ["LOSS"]                   # kd, t_sft or gt_sft
OUT = os.environ["OUT"]
TEXT = "gold" if LOSS == "gt_sft" else "response"   # the answers trained on: ground truth or the teacher's

# student on the first visible gpu, teachers on the second (both teachers need ~31 gb)
STUDENT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TEACHER_DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else STUDENT_DEVICE

random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUT, exist_ok=True)
# load_model() reads this to rebuild a checkpoint of this run
with open(f"{OUT}/start.json", "w") as f:
    json.dump({"start": START}, f)
t_start = time.time()

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
tokenizer.padding_side = "left"

train, val_kd = {}, {}
for d in DOMAINS:
    train[d], val_kd[d] = kd_rows(d)
# accuracy and drift are always measured on both domains: the checkpoint must be good at both
val = {d: rl_rows(d, "val") for d in ["math", "code"]}
drift_rows = {d: kd_rows(d)[1] for d in ["math", "code"]}

# fp32 weights, bf16 compute through autocast. only the adapter is trained
student = get_peft_model(load_model(START, STUDENT_DEVICE), lora())
net = student.get_base_model()   # model with the adapter layers inside (logits_at() needs .model and .lm_head)
sampler = Sampler(student)       # bf16 copy for the val answers
teachers = {}
if LOSS == "kd":
    teachers = {d: AutoModelForCausalLM.from_pretrained(TEACHERS[d], dtype=torch.bfloat16).to(TEACHER_DEVICE).eval()
                for d in DOMAINS}


def batch_loss(rows, d):
    # summed over answer tokens: kl to the teacher (kd) or ce on the answer's own tokens (t_sft, gt_sft)
    ids, att, mask = collate([r["prompt"] for r in rows], [r[TEXT] for r in rows])
    with torch.autocast(net.device.type, dtype=torch.bfloat16):
        s = logits_at(net, ids, att, mask, net.device)
        if LOSS == "kd":
            with torch.no_grad():
                t = logits_at(teachers[d], ids, att, mask, net.device)

    s_logp = torch.log_softmax(s, -1)
    del s
    if LOSS != "kd":
        return F.nll_loss(s_logp, ids.roll(-1, dims=1)[mask].to(net.device), reduction="sum")

    t_logp = torch.log_softmax(t, -1)
    del t
    return (t_logp.exp() * (t_logp - s_logp)).sum()


@torch.no_grad()
def val_loss():
    # the training loss per answer token on the val rows of each trained domain
    out = {}
    for d in DOMAINS:
        total = sum(batch_loss(val_kd[d][i:i + MICRO], d).item() for i in range(0, len(val_kd[d]), MICRO))
        out[f"val_loss_{d}"] = total / sum(len(r[TEXT]) for r in val_kd[d])
    return out


def make_batches():
    # every batch is from one domain, so it needs one teacher. batches of the two domains are mixed
    batches = []
    for d in DOMAINS:
        rows = train[d][:]
        random.shuffle(rows)
        batches += [(d, rows[i:i + BATCH]) for i in range(0, len(rows) - BATCH + 1, BATCH)]
    random.shuffle(batches)
    return batches


steps_per_epoch = sum(len(train[d]) // BATCH for d in DOMAINS)
total_steps = EPOCHS * steps_per_epoch
# evenly spaced, the last one on the final step
check_steps = {round(k * total_steps / (EPOCHS * CHECKS)) for k in range(1, EPOCHS * CHECKS + 1)}
params = [p for p in student.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, fused=torch.cuda.is_available())
# linear warmup over the first 3% of steps, then cosine decay to 0
schedule = get_cosine_schedule_with_warmup(opt, int(0.03 * total_steps), total_steps)

log = {"settings": {"seed": SEED, "start": START, "domains": DOMAINS, "loss": LOSS, "epochs": EPOCHS, "batch": BATCH,
                    "lr": LR, "lora": {"r": lora().r, "alpha": lora().lora_alpha}, "trainable": sum(p.numel() for p in params),
                    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
       "train": [], "val": []}
best = -1.0
val_seconds = 0.0


def check(step):
    # val loss, drift and val scores. the best adapter so far is saved
    global best, val_seconds
    t0 = time.time()
    entry = {"step": step, "epoch": step / steps_per_epoch, **val_loss(),
             **{f"drift_{d}": drift(student, rows) for d, rows in drift_rows.items()},
             **val_scores(sampler.update(student), tokenizer, val)}
    val_seconds += time.time() - t0
    log["val"].append(entry)
    print(entry)
    # step 0 (the start) is scored but never kept, so a second stage always starts from a trained checkpoint
    if step > 0 and entry["val"] > best:
        best = entry["val"]
        log["best_step"] = step
        student.save_pretrained(f"{OUT}/best")
    student.train()


check(0)
step = 0
for epoch in range(EPOCHS):
    for d, batch in make_batches():
        # sorted by length so micro batches have less padding
        batch.sort(key=lambda r: len(r["prompt"]) + len(r[TEXT]))
        tokens = sum(len(r[TEXT]) for r in batch)

        batch_total = 0.0
        for i in range(0, BATCH, MICRO):
            # divided by the tokens of the whole batch, so the micro batches add up to a per-token mean
            loss = batch_loss(batch[i:i + MICRO], d) / tokens
            loss.backward()
            batch_total += loss.item()

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        schedule.step()
        opt.zero_grad()

        step += 1
        log["train"].append({"step": step, "domain": d, "loss": batch_total})
        if step % 10 == 0:
            print(f"step {step}/{total_steps}  {d}  loss {batch_total:.4f}")
        if step in check_steps:
            check(step)

if EVAL_LAST:
    student.save_pretrained(f"{OUT}/last")
log["train_h"] = (time.time() - t_start - val_seconds) / 3600
log["val_h"] = val_seconds / 3600
with open(f"{OUT}/log.json", "w") as f:
    json.dump(log, f)

# benchmark the saved adapters, merged into their start the same way every model is loaded
del student, net, sampler, teachers, opt
torch.cuda.empty_cache()
for name in ["best"] + (["last"] if EVAL_LAST else []):
    model = load_model(f"{OUT}/{name}", STUDENT_DEVICE).eval()
    test_eval(model, tokenizer, f"{OUT}/{name}", f"{OUT}/results_{name}")
    del model
