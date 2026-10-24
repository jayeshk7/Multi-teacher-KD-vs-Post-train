# ablation of kd_train.py: kl only (ALPHA = 1.0, no ground truth ce), both teachers at once. everything else the same.
# needs data/{math,code}.jsonl generated using gen_data.py, kd_utils.py and bench_utils.py next to this file

import json
import os
import random

import torch
from huggingface_hub import create_repo, snapshot_download, upload_folder
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from bench_utils import benchmark
from kd_utils import losses, val_all

STUDENT = "Qwen/Qwen2.5-0.5B-Instruct"
TEACHERS = {"math": "Qwen/Qwen2.5-Math-7B-Instruct", "code": "Qwen/Qwen2.5-Coder-7B-Instruct"}

ALPHA = 1.0       # weight of the kl loss, the gold ce loss gets 1 - ALPHA. the unused loss is still computed and logged
TEMPERATURE = 1.0 # kd temperature of the kl. 1 = the models' own distributions, > 1 softens both (hinton et al. 2015).
                  # the train log's kl is at this T, val kl is always at T = 1

EPOCHS = 2        # over both domains, so the student sees the same data as in the sequential runs
BATCH = 32        # prompts per optimizer step, all from one domain
MICRO = 4         # prompts per forward pass, lower it if out of memory
LR = 1e-5
VAL_SIZE = 200    # validation set size per domain (math and code)
SEED = int(os.environ.get("SEED", 0))  # 0, 1 or 2, sets the data order of every epoch. run as SEED=1 python -u <script>

OUT_DIR = "abl_kl_only_joint" + (f"_seed{SEED}" if SEED else "")
HUB_DATA = "jajamken/FML_project"     # hf dataset written by gen_data.py. empty == use local data/ folder
HUB_REPO = "jajamken/FML_project"     # student and results go here. empty == local only

# two 48 gb gpus (server, like the sequential runs, CUDA_VISIBLE_DEVICES=<two gpus>): the student alone, both teachers
# on the second gpu. one 96 gb gpu (molab): all on cuda:0. MICRO = 4 on both: the smoke test peaked at 43 gb on the
# student gpu, MICRO = 8 would need about 80 gb there, too much next to the teachers' 30.5 gb on one molab gpu
DEVICE = {"student": "cuda:0", "math": "cuda:1", "code": "cuda:1"}

MAX_QUESTIONS = 2000  # more than any benchmark has (gsm8k test 1319, mbpp+ 378, humaneval+ 164) so all are used

random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

if HUB_DATA:
    snapshot_download(HUB_DATA, repo_type="dataset", local_dir="data")

train_set, val_set = {}, {}
for d in TEACHERS:
    # iterating over the dict gives us the key not the value
    rows = [json.loads(line) for line in open(f"data/{d}.jsonl")]
    train_set[d], val_set[d] = rows[:-VAL_SIZE], rows[-VAL_SIZE:]

tokenizer = AutoTokenizer.from_pretrained(STUDENT)
# only for generation in benchmark(): generate() continues each row from its last position, so prompts are padded on the
# left ([PAD, PAD, prompt]). training batches don't use the tokenizer, collate() pads them on the right
tokenizer.padding_side = "left"

# fp32 weights for adam, bf16 compute through autocast in kd_utils.losses()
student = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.float32).to(DEVICE["student"])


def make_batches():
    # every batch is from one domain, so it needs only one teacher. the batches of the two domains are mixed
    batches = []
    for d in TEACHERS:
        rows = train_set[d][:]
        random.shuffle(rows)
        batches += [(d, rows[i:i + BATCH]) for i in range(0, len(rows) - BATCH + 1, BATCH)]

    random.shuffle(batches)
    return batches


def distill():
    # both teachers stay loaded, each batch uses the teacher of its domain. eval mode (.eval()) turns off dropout behaviour
    teachers = {d: AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16).to(DEVICE[d]).eval() for d, m in TEACHERS.items()}
    # train() is just a flag like eval()
    student.train()

    # one schedule over the batches of both domains
    total_optimizer_steps = EPOCHS * sum(len(train_set[d]) // BATCH for d in TEACHERS)
    opt = torch.optim.AdamW(student.parameters(), lr=LR, fused=True)

    # linear warmup over the first 3% of steps, then cosine decay to 0 (the schedule alpaca uses)
    lr_scheduler = get_cosine_schedule_with_warmup(opt, int(0.03 * total_optimizer_steps), total_optimizer_steps)

    log = {"train": [], "val": [val_all(student, teachers, val_set, MICRO, 0)]}
    step = 0

    for epoch in range(EPOCHS):
        for d, batch in make_batches():
            # sort by length so micro batches have less padding
            batch.sort(key=lambda r: len(r["prompt"]) + len(r["response"]))

            # number of teacher answer tokens in the batch, divides the summed kl
            kl_tokens = sum(len(r["response"]) for r in batch)

            # number of gold answer tokens in the batch, divides the summed ce
            ce_tokens = sum(len(r["gold"]) for r in batch)

            # per-token kl and ce of this batch, for logging only. the gradients add up in .grad through loss.backward()
            batch_kl, batch_ce = 0.0, 0.0
            for i in range(0, BATCH, MICRO):
                kl, ce = losses(student, teachers[d], batch[i:i + MICRO], TEMPERATURE)

                # divide by the tokens of the whole batch so the sum over micro batches is a per-token mean
                loss = ALPHA * kl / kl_tokens + (1 - ALPHA) * ce / ce_tokens
                loss.backward()
                batch_kl += kl.item() / kl_tokens
                batch_ce += ce.item() / ce_tokens

            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            lr_scheduler.step()
            opt.zero_grad()

            step += 1
            log["train"].append({"step": step, "domain": d, "kl": batch_kl, "ce": batch_ce})

            if step % 10 == 0:
                print(f"step {step}/{total_optimizer_steps}  {d}  kl {batch_kl:.4f}  ce {batch_ce:.4f}")

        log["val"].append(val_all(student, teachers, val_set, MICRO, step))

    # the teachers are not needed for the benchmark, free their memory
    del teachers, opt
    torch.cuda.empty_cache()
    return log


# ---- run: distill both teachers into the student at once, benchmark ----

log = distill()

# save in bf16 like the baselines. the in-memory model is not used after this: .to(bf16) also rounds rope's inv_freq
# buffer to bf16, while from_pretrained recomputes it exactly. so the benchmark loads the saved checkpoint
student.to(torch.bfloat16)
student.save_pretrained(f"{OUT_DIR}/joint")
# tokenizer.save_pretrained(f"{OUT_DIR}/joint")
del student
torch.cuda.empty_cache()

with open(f"{OUT_DIR}/joint/log.json", "w") as f:
    json.dump(log, f)

# benchmark the saved student, loaded exactly like the baselines
model = AutoModelForCausalLM.from_pretrained(f"{OUT_DIR}/joint", dtype=torch.bfloat16).to(DEVICE["student"]).eval()
benchmark(model, tokenizer, "joint", f"{OUT_DIR}/results_joint", MAX_QUESTIONS)

# upload, the local copies get deleted later
if HUB_REPO:
    create_repo(HUB_REPO, private=True, exist_ok=True)
    upload_folder(repo_id=HUB_REPO, folder_path=OUT_DIR, path_in_repo=OUT_DIR)
