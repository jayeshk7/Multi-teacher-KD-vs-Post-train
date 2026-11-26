# rlvr on code: the student samples answers to kodcode prompts, the reward is the fraction of unit tests passed.
# trl's grpo trainer does the sampling and the loss, only a lora adapter is trained.
# needs reward.py next to this file and bench_utils.py, kd_utils.py in ../1_pre_checkpoint. pip install trl peft datasets pytest evalplus

import json
import os
import sys

import torch
from datasets import load_dataset
from huggingface_hub import create_repo, hf_hub_download, upload_folder
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# bench_utils.py and kd_utils.py are shared with the october scripts
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1_pre_checkpoint"))
from bench_utils import CODE_SYSTEM, benchmark
from kd_utils import VOCAB_SIZE, collate
from reward import code_reward

START_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"

SMOKE = False           # overfit test on 8 prompts, reward should climb. False == the pilot

TRAIN_SIZE = 7269      # same count as kd's code stage. kd's val rows are all after this index, so they stay unseen
HELD_OUT = 500          # prompts for the reward curve, never trained on. kd never saw them either
MAX_STEPS = 454         # one pass over TRAIN_SIZE
BENCHMARK = True

PROMPTS_PER_STEP = 16
NUM_GENERATIONS = 8     # answers per prompt, the group that the advantage is computed in
MICRO = 32              # answers per forward pass, lower it if out of memory
MAX_ANSWER = 512        # tokens. the code teacher's answers are 400 at p95
EVAL_BATCH = 100        # keep HELD_OUT a multiple of this, batches are averaged equally

LR = 1e-4
LORA_RANK = 16
LORA_ALPHA = 32

EVAL_EVERY = 50         # steps between held-out reward (sampled and greedy) and drift measurements
DRIFT_ROWS = 200        # kd's val rows of each domain
DRIFT_BATCH = 8

SEED = int(os.environ.get("SEED", 0))
OUT_DIR = "grpo_code" + (f"_seed{SEED}" if SEED else "")
HUB_REPO = ""           # results and adapter go here. empty == local only
HUB_DATA = "jajamken/FML_project"
MAX_QUESTIONS = 2000    # more than any benchmark has

if SMOKE:
    TRAIN_SIZE, PROMPTS_PER_STEP, MAX_STEPS, BENCHMARK, OUT_DIR = 8, 8, 200, False, "grpo_smoke"

os.makedirs(OUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
tokenizer.padding_side = "left"

# same shuffle and prompt as gen_data.py
kod = load_dataset("KodCode/KodCode-Light-RL-10K", split="train").shuffle(seed=0)


def to_row(ex):
    user = ("Please solve the programming task below in Python. \n\n" + ex["question"].strip()
            + "\n\nNote that the function declaration is " + ex["test_info"][0]["function_declaration"].strip()
            + ". Your code should be wrapped in a markdown code block.")
    return {"prompt": [{"role": "system", "content": CODE_SYSTEM}, {"role": "user", "content": user}],
            "test": ex["test"], "function_name": ex["test_info"][0]["function_name"]}


train_set = kod.select(range(TRAIN_SIZE)).map(to_row, remove_columns=kod.column_names)
held_out = kod.select(range(7473, 7473 + HELD_OUT)).map(to_row, remove_columns=kod.column_names)
if SMOKE:
    held_out = train_set    # the test is whether it can overfit these

# teacher answers on kd's val rows, fixed text to measure how far the model moved on each domain
drift_rows = {}
for d in ["math", "code"]:
    path = hf_hub_download(HUB_DATA, f"{d}.jsonl", repo_type="dataset")
    drift_rows[d] = [json.loads(line) for line in open(path)][-DRIFT_ROWS:]


@torch.no_grad()
def drift(model, rows):
    # kl(start || now) per token. disable_adapter() gives back the starting model
    total, tokens = 0.0, 0
    for i in range(0, len(rows), DRIFT_BATCH):
        batch = rows[i:i + DRIFT_BATCH]
        ids, att, mask = collate([r["prompt"] for r in batch], [r["response"] for r in batch])
        ids, att, mask = ids.to(model.device), att.to(model.device), mask.to(model.device)

        with torch.autocast(model.device.type, dtype=torch.bfloat16):
            now = model(input_ids=ids, attention_mask=att).logits[mask][:, :VOCAB_SIZE].float()
            with model.disable_adapter():
                start = model(input_ids=ids, attention_mask=att).logits[mask][:, :VOCAB_SIZE].float()

        start_logp = torch.log_softmax(start, -1)
        total += (start_logp.exp() * (start_logp - torch.log_softmax(now, -1))).sum().item()
        tokens += mask.sum().item()
    return total / tokens


@torch.no_grad()
def greedy_reward(model):
    # benchmarks are greedy. sampled reward can rise just from less randomness, base is 0.329 sampled and 0.486 greedy
    rewards = []
    for i in range(0, len(held_out), EVAL_BATCH):
        rows = held_out[i:i + EVAL_BATCH]
        text = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in rows["prompt"]]
        enc = tokenizer(text, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

        with torch.autocast(model.device.type, dtype=torch.bfloat16):
            out = model.generate(**enc, max_new_tokens=MAX_ANSWER, do_sample=False, repetition_penalty=1.0, use_cache=True)

        answers = tokenizer.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        rewards += code_reward([[{"role": "assistant", "content": a}] for a in answers], rows["test"], rows["function_name"])
    return sum(rewards) / len(rewards), sum(r == 1.0 for r in rewards)


class MeasureCallback(TrainerCallback):
    # runs right after each sampled held-out reward measurement
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        model.eval()
        reward, passed = greedy_reward(model)
        row = {"step": state.global_step, "greedy_reward": reward, "greedy_all_passed": passed,
               **{f"drift_{d}": drift(model, rows) for d, rows in drift_rows.items()}}
        state.log_history.append(row)
        print(row)


config = GRPOConfig(
    output_dir=OUT_DIR,
    seed=SEED,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=MICRO,
    gradient_accumulation_steps=PROMPTS_PER_STEP * NUM_GENERATIONS // MICRO,
    per_device_eval_batch_size=EVAL_BATCH,
    num_generations=NUM_GENERATIONS,
    num_generations_eval=1,
    max_completion_length=MAX_ANSWER,
    learning_rate=LR,
    # same schedule as kd
    lr_scheduler_type="cosine",
    warmup_steps=int(0.03 * MAX_STEPS),
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=EVAL_EVERY,
    eval_on_start=True,
    save_strategy="no",
    report_to="none",
)

# fp32 weights for adam, bf16 compute through autocast (GRPOConfig has bf16 on by default)
model = AutoModelForCausalLM.from_pretrained(START_MODEL, dtype=torch.float32)

trainer = GRPOTrainer(
    model=model,
    args=config,
    reward_funcs=code_reward,
    train_dataset=train_set,
    eval_dataset=held_out,
    processing_class=tokenizer,
    peft_config=LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules="all-linear", task_type="CAUSAL_LM"),
    callbacks=[MeasureCallback()],
)
trainer.train()

trainer.save_model(f"{OUT_DIR}/adapter")
with open(f"{OUT_DIR}/log.json", "w") as f:
    json.dump(trainer.state.log_history, f)

if BENCHMARK:
    device = model.device
    del trainer, model
    torch.cuda.empty_cache()

    # load what was saved, not the model in memory. no merge: bf16 rounding of merged weights could cancel a small update
    saved = AutoModelForCausalLM.from_pretrained(START_MODEL, dtype=torch.float32).to(device)
    saved = PeftModel.from_pretrained(saved, f"{OUT_DIR}/adapter").eval()
    with torch.autocast(device.type, dtype=torch.bfloat16):
        benchmark(saved, tokenizer, OUT_DIR, f"{OUT_DIR}/results", MAX_QUESTIONS)

if HUB_REPO:
    create_repo(HUB_REPO, exist_ok=True)
    upload_folder(repo_id=HUB_REPO, folder_path=OUT_DIR, path_in_repo=OUT_DIR)
