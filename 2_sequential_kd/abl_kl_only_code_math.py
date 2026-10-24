# ablation of seq_code_then_math.py: kl only (ALPHA = 1.0, no gold ce), code then math. everything else the same.
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

DISTILL_ORDER = ["code", "math"]  # which teacher is distilled first
ALPHA = 1.0       # weight of the kl loss, the gold ce loss gets 1 - ALPHA. the unused loss is still computed and logged
TEMPERATURE = 1.0 # kd temperature of the kl. 1 = the models' own distributions, > 1 softens both (hinton et al. 2015).
                  # the train log's kl is at this T, val kl is always at T = 1

EPOCHS = 2
BATCH = 32        # prompts per optimizer step
MICRO = 4         # prompts per forward pass, lower it if out of memory
LR = 1e-5
VAL_SIZE = 200    # validation set size per domain (math and code)
SEED = int(os.environ.get("SEED", 0))  # 0, 1 or 2, sets the data order of every epoch. run as SEED=1 python -u <script>

OUT_DIR = "abl_kl_only_code_math" + (f"_seed{SEED}" if SEED else "")  # seed 0 keeps the name of the first runs
HUB_DATA = "jajamken/FML_project"     # hf dataset written by gen_data.py. empty == use local data/ folder
HUB_REPO = "jajamken/FML_project"     # students and results go here. empty == local only

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
# TODO: understand how this mixed precision training is going on
student = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.float32).to(DEVICE["student"])

# both teachers stay loaded so val can be measured on both domains, the one not being trained shows forgetting
# eval mode (.eval()) turns off dropout behaviour
# since during validation we will need the teacher logits to compute the KL (teacher || student) we need the teacher models in memory
teachers = {d: AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16).to(DEVICE[d]).eval() for d, m in TEACHERS.items()}


def distill(d):
    teacher = teachers[d]
    # train() is just a flag like eval()
    student.train()

    total_optimizer_steps = EPOCHS * (len(train_set[d]) // BATCH)
    opt = torch.optim.AdamW(student.parameters(), lr=LR, fused=True)

    # linear warmup over the first 3% of steps, then cosine decay to 0 (the schedule alpaca uses)
    lr_scheduler = get_cosine_schedule_with_warmup(opt, int(0.03 * total_optimizer_steps), total_optimizer_steps)

    log = {"train": [], "val": [val_all(student, teachers, val_set, MICRO, 0)]}
    step = 0

    for epoch in range(EPOCHS):
        # rows is a list with one dict per training example
        rows = train_set[d][:]
        len_rows = len(rows)

        # get a new order in every epoch so the last batch which is dropped in every epoch is also seen during training
        # and also the batches are not the exact same everytime
        random.shuffle(rows)

        for b in range(0, len_rows - BATCH + 1, BATCH):
            batch = rows[b:b + BATCH]
            # sort by length so micro batches have less padding
            batch.sort(key=lambda r: len(r["prompt"]) + len(r["response"]))

            # number of teacher answer tokens in the batch, divides the summed kl
            kl_tokens = sum(len(r["response"]) for r in batch)

            # number of gold answer tokens in the batch, divides the summed ce
            ce_tokens = sum(len(r["gold"]) for r in batch)

            # per-token kl and ce of this batch, for logging only. the gradients add up in .grad through loss.backward()
            batch_kl, batch_ce = 0.0, 0.0
            for i in range(0, BATCH, MICRO):
                kl, ce = losses(student, teacher, batch[i:i + MICRO], TEMPERATURE)

                # TODO: ask again why dividing by the whole batch's token count gives a per-token mean
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
            log["train"].append({"step": step, "kl": batch_kl, "ce": batch_ce})

            if step % 10 == 0:
                print(f"{d} step {step}/{total_optimizer_steps}  kl {batch_kl:.4f}  ce {batch_ce:.4f}")

        log["val"].append(val_all(student, teachers, val_set, MICRO, step))

    del opt
    torch.cuda.empty_cache()
    return log


# ---- run: distill teacher 1, benchmark, distill teacher 2, benchmark ----

for stage, d in enumerate(DISTILL_ORDER, 1):
    name = f"stage{stage}_{d}"
    log = distill(d)

    # save in bf16 like the baselines. the in-memory model is not used after this: .to(bf16) also rounds rope's
    # inv_freq buffer to bf16, while from_pretrained recomputes it exactly. so the benchmark and the next stage
    # both load the saved checkpoint
    student.to(torch.bfloat16)
    student.save_pretrained(f"{OUT_DIR}/{name}")
    # tokenizer.save_pretrained(f"{OUT_DIR}/{name}")
    del student
    torch.cuda.empty_cache()

    with open(f"{OUT_DIR}/{name}/log.json", "w") as f:
        json.dump(log, f)

    # benchmark the saved student after every distill step, loaded exactly like the baselines
    model = AutoModelForCausalLM.from_pretrained(f"{OUT_DIR}/{name}", dtype=torch.bfloat16).to(DEVICE["student"]).eval()
    benchmark(model, tokenizer, name, f"{OUT_DIR}/results_{name}", MAX_QUESTIONS)
    del model
    torch.cuda.empty_cache()

    # upload after every stage
    if HUB_REPO:
        create_repo(HUB_REPO, private=True, exist_ok=True)
        upload_folder(repo_id=HUB_REPO, folder_path=OUT_DIR, path_in_repo=OUT_DIR)

    # the next stage continues from the saved student, fp32 weights for adam
    student = AutoModelForCausalLM.from_pretrained(f"{OUT_DIR}/{name}", dtype=torch.float32).to(DEVICE["student"])
