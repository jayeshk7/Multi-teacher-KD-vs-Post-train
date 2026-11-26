# one grpo stage with a lora adapter on the student's own sampled answers, starting from START (a hub model or an
# earlier stage's checkpoint). one pass over the same prompts the sft and kd runs train on.
# every 50 steps the student is scored on the val rows of both domains; the best adapter is benchmarked
# run: SEED=0 START=inst_kd_math_s0/best DOMAINS=code OUT=inst_seq_kd_math_then_rl_code_s0 python -u train_grpo.py

import contextlib
import json
import os
import time

import torch
import trl.trainer.grpo_trainer as grpo_module
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.models.utils import _override_model_generation_config

from common import TOKENIZER, Sampler, drift, ended, kd_rows, load_model, lora, reward, rl_rows, test_eval, val_scores

PROMPTS_PER_STEP = 16
NUM_GENERATIONS = 8     # answers per prompt, the group the advantage is computed in
MICRO = 8               # answers per forward pass
MAX_ANSWER = 768        # tokens per sampled answer (95% of the teachers' answers are shorter than 470)
LR = float(os.environ.get("LR", 1e-4))  # 1e-4, picked from short runs at 5e-5, 1e-4 and 2e-4
CHECK_EVERY = 50        # steps between val scores
SAMPLE_EVERY = 5        # steps between saved training answers

SEED = int(os.environ["SEED"])
START = os.environ["START"]                 # hub name, model folder or <run>/best of an earlier stage
DOMAINS = os.environ["DOMAINS"].split(",")  # code, math or math,code (multi)
OUT = os.environ["OUT"]
STEPS = int(os.environ.get("STEPS", 0))     # stop early after STEPS steps (only the short lr runs), 0 = full pass

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT, exist_ok=True)
# load_model() reads this to rebuild a checkpoint of this run
with open(f"{OUT}/start.json", "w") as f:
    json.dump({"start": START}, f)
t_start = time.time()

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
tokenizer.padding_side = "left"

train = [r for d in DOMAINS for r in rl_rows(d, "train")]
val = {d: rl_rows(d, "val") for d in ["math", "code"]}
drift_rows = {d: kd_rows(d)[1] for d in ["math", "code"]}
# the lr schedule always spans one full pass, so a short run follows the start of a real run
TOTAL = len(train) // PROMPTS_PER_STEP
STOP = STEPS or TOTAL

# fp32 weights, bf16 compute (grpo's default). trl adds the adapter
model = load_model(START, DEVICE)
samples = open(f"{OUT}/samples.jsonl", "w")


def rl_reward(completions, completion_ids, trainer_state, **kwargs):
    rewards = reward(completions, completion_ids, **kwargs)
    # every few steps all answers are saved, to read what the model writes and why it scores
    if trainer_state.global_step % SAMPLE_EVERY == 0:
        for c, ids, row, d, r in zip(completions, completion_ids, kwargs["id"], kwargs["domain"], rewards):
            samples.write(json.dumps({"step": trainer_state.global_step, "id": row, "domain": d, "reward": r,
                                      "tokens": len(ids), "ended": ended(ids), "answer": c[0]["content"]}) + "\n")
        samples.flush()
    return rewards


checks = []
best = -1.0
val_seconds = 0.0


def check(model, step):
    # val scores and drift of both domains, the best adapter so far is saved. model is trl's peft model
    global best, val_seconds
    t0 = time.time()
    entry = {"step": step, **{f"drift_{d}": drift(model, rows) for d, rows in drift_rows.items()},
             **val_scores(sampler.update(model), tokenizer, val)}
    val_seconds += time.time() - t0
    checks.append(entry)
    print(entry)
    # step 0 (the start) is scored but never kept
    if step > 0 and entry["val"] > best:
        best = entry["val"]
        model.save_pretrained(f"{OUT}/best")
        checks[-1]["best"] = True
    model.train()


class Check(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % CHECK_EVERY == 0 or state.global_step == STOP:
            check(model, state.global_step)
        if state.global_step >= STOP:
            control.should_training_stop = True


config = GRPOConfig(
    output_dir=OUT,
    seed=SEED,
    max_steps=TOTAL,
    per_device_train_batch_size=MICRO,
    gradient_accumulation_steps=PROMPTS_PER_STEP * NUM_GENERATIONS // MICRO,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_ANSWER,
    learning_rate=LR,
    # same schedule as the sft and kd runs. the rest are trl 1.17.1 defaults: loss_type "dapo", beta 0 (no kl
    # penalty), epsilon 0.2, one update per batch, sampling temperature 1
    lr_scheduler_type="cosine",
    warmup_steps=int(0.03 * TOTAL),
    logging_steps=1,
    eval_strategy="no",
    save_strategy="no",
    report_to="none",
)

trainer = GRPOTrainer(model=model, args=config, reward_funcs=rl_reward, train_dataset=Dataset.from_list(train),
                      processing_class=tokenizer, peft_config=lora(), callbacks=[Check()])
sampler = Sampler(trainer.model)


@contextlib.contextmanager
def sample_with_bf16_copy(model_wrapped, accelerator, generation_kwargs=None, **kwargs):
    # trl samples the training answers inside this context: give it the bf16 copy (common.Sampler) of the current model
    gen = sampler.update(trainer.model)
    with _override_model_generation_config(gen, generation_kwargs=generation_kwargs):
        yield gen


grpo_module.unwrap_model_for_generation = sample_with_bf16_copy
check(trainer.model, 0)
trainer.train()

log = {"settings": {"seed": SEED, "start": START, "domains": DOMAINS, "steps": STOP, "schedule_steps": TOTAL, "lr": LR,
                    "lora": {"r": 16, "alpha": 32}, "prompts_per_step": PROMPTS_PER_STEP, "num_generations": NUM_GENERATIONS, "max_answer": MAX_ANSWER,
                    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
       "train": trainer.state.log_history, "val": checks,
       "train_h": (time.time() - t_start - val_seconds) / 3600, "val_h": val_seconds / 3600}
with open(f"{OUT}/log.json", "w") as f:
    json.dump(log, f)

# benchmark the best adapter (not for the short lr runs)
del trainer, model, sampler
torch.cuda.empty_cache()
if not os.environ.get("STEPS"):
    saved = load_model(f"{OUT}/best", DEVICE).eval()
    test_eval(saved, tokenizer, f"{OUT}/best", f"{OUT}/results_best")
