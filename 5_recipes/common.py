# shared code for the final runs: data, grpo reward, model loading, generation, validation and test evaluation.
# every checkpoint is a lora adapter, load_model() rebuilds a model by merging the adapters of its stages in order

import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from datasets import load_dataset
from evalplus.data import write_jsonl
from huggingface_hub import hf_hub_download
from math_verify import parse, verify
from peft import LoraConfig, PeftModel
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM

from bench_utils import CODE_SYSTEM, MATH_SYSTEM, benchmark, chat, is_right
from kd_utils import collate, logits_at
from reward import WORKERS, score

INSTRUCT = "Qwen/Qwen2.5-0.5B-Instruct"
TOKENIZER = INSTRUCT
EOS = [151645, 151643]                    # <|im_end|> ends a chat answer, <|endoftext|> a document
HUB_DATA = "jajamken/FML_project"         # teacher answers made by gen_data.py
VAL_SIZE = 200                            # the last 200 rows of each domain are the val set
HELD_OUT = range(7473, 10000)             # kodcode rows no run trains on: the held-out code test
VAL_MAX = 1024                            # max tokens per val answer
TEST_MAX = 1600                           # max tokens per test answer, same as bench_utils
GEN_BATCH = 128                           # prompts per generate call
DRIFT_BATCH = 4                           # rows per forward pass in drift()
MAX_QUESTIONS = 2000                      # more than any benchmark has, so every question is used


def kd_rows(domain):
    # token ids of the prompt, the teacher's answer and the ground truth answer, split into train and val
    path = hf_hub_download(HUB_DATA, f"{domain}.jsonl", repo_type="dataset")
    rows = [json.loads(line) for line in open(path)]
    return rows[:-VAL_SIZE], rows[-VAL_SIZE:]


def kept_index(domain):
    # dataset row of each kd row (gen_data.py dropped the answers that hit its length limit)
    path = hf_hub_download(HUB_DATA, f"{domain}_cut.jsonl", repo_type="dataset")
    cut = {json.loads(line)["index"] for line in open(path)}
    return [i for i in range(7473) if i not in cut]


def kodcode():
    # same shuffle as gen_data.py
    return load_dataset("KodCode/KodCode-Light-RL-10K", split="train").shuffle(seed=0)


def math_row(ex):
    # gsm8k row -> chat prompt and the final answer the reward checks
    final = ex["answer"].split("####")[-1].strip().replace(",", "")
    return {"prompt": [{"role": "system", "content": MATH_SYSTEM}, {"role": "user", "content": ex["question"]}],
            "domain": "math", "answer": final, "test": "", "function_name": ""}


def code_row(ex):
    # kodcode row -> its own rl prompt (same as gen_data.py) and the tests the reward runs
    user = ("Please solve the programming task below in Python. \n\n" + ex["question"].strip()
            + "\n\nNote that the function declaration is " + ex["test_info"][0]["function_declaration"].strip()
            + ". Your code should be wrapped in a markdown code block.")
    return {"prompt": [{"role": "system", "content": CODE_SYSTEM}, {"role": "user", "content": user}],
            "domain": "code", "answer": "", "test": ex["test"], "function_name": ex["test_info"][0]["function_name"]}


def rl_rows(domain, split):
    # chat prompts and reward fields for the same rows as kd_rows(). split is "train" or "val"
    idx = kept_index(domain)
    idx = idx[:-VAL_SIZE] if split == "train" else idx[-VAL_SIZE:]
    if domain == "math":
        rows = [math_row(ex) for ex in load_dataset("openai/gsm8k", "main", split="train").select(idx)]
    else:
        rows = [code_row(ex) for ex in kodcode().select(idx)]
    # id = dataset row, to follow one prompt through the saved answers
    for i, r in zip(idx, rows):
        r["id"] = f"{domain}_{i}"
    return rows


def ended(ids):
    # true if the answer stopped by itself instead of hitting the token limit
    return len(ids) > 0 and ids[-1] in EOS


def reward(completions, completion_ids, domain, answer, test, function_name, **kwargs):
    # grpo reward, one value per sampled answer. a cut answer gets 0
    texts = [c[0]["content"] for c in completions]
    ok = [ended(ids) for ids in completion_ids]

    def one(text, ok, domain, answer, test, function_name):
        if not ok:
            return 0.0
        return float(is_right(text, answer)) if domain == "math" else score(text, test, function_name)

    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(one, texts, ok, domain, answer, test, function_name))


def lora():
    # rank 16 on every linear layer of attention and mlp, alpha = 2 x rank. LORA_R changes the rank (one rank-48 run)
    r = int(os.environ.get("LORA_R", 16))
    return LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")


def load_model(ref, device, dtype=torch.float32):
    # ref is a hub name, a model folder or a run's adapter (<run>/best). an adapter is merged in fp32 into the start
    # model named in <run>/start.json, which can itself be an adapter
    if os.path.exists(f"{ref}/adapter_config.json"):
        start = json.load(open(f"{os.path.dirname(ref.rstrip('/'))}/start.json"))["start"]
        model = PeftModel.from_pretrained(load_model(start, device, dtype), ref).merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(ref, dtype=dtype).to(device)
    # stop at both end tokens
    model.generation_config.eos_token_id = EOS
    model.generation_config.pad_token_id = EOS[1]
    return model


class Sampler:
    # bf16 copy of the model with the adapter merged in. it writes every answer (val, test, grpo samples), because
    # generating through peft's lora layers was ~2x slower. update() copies the current weights into it
    def __init__(self, model):
        net = model.get_base_model() if isinstance(model, PeftModel) else model
        self.model = AutoModelForCausalLM.from_config(net.config, dtype=torch.bfloat16).to(net.device).eval()
        self.model.generation_config = copy.deepcopy(net.generation_config)
        self.update(model)

    @torch.no_grad()
    def update(self, model):
        net = model.get_base_model() if isinstance(model, PeftModel) else model
        params = dict(self.model.named_parameters())
        for name, module in net.named_modules():
            if isinstance(module, LoraLayer):
                params[f"{name}.weight"].copy_(module.base_layer.weight + module.get_delta_weight("default"))
        for name, p in net.named_parameters():
            if "lora_" not in name and not name.endswith("base_layer.weight"):
                params[name.replace(".base_layer", "")].copy_(p)
        return self.model


def for_generation(model):
    # the bf16 model that generates. not model.to(bfloat16): that also rounds the rope frequencies to bf16
    return model if model.dtype == torch.bfloat16 else Sampler(model).model


@torch.no_grad()
def drift(model, rows):
    # how far the model moved from its start: kl(start || now) per answer token on the teachers' val answers
    net = model.get_base_model()
    total, tokens = 0.0, 0
    for i in range(0, len(rows), DRIFT_BATCH):
        batch = rows[i:i + DRIFT_BATCH]
        ids, att, mask = collate([r["prompt"] for r in batch], [r["response"] for r in batch])
        with torch.autocast(net.device.type, dtype=torch.bfloat16):
            now = logits_at(net, ids, att, mask, net.device)
            with model.disable_adapter():
                start = logits_at(net, ids, att, mask, net.device)
        start_logp = torch.log_softmax(start, -1)
        total += (start_logp.exp() * (start_logp - torch.log_softmax(now, -1))).sum().item()
        tokens += mask.sum().item()
    return total / tokens


@torch.no_grad()
def generate(model, tokenizer, texts, max_new):
    # greedy, shortest prompts first. returns the answers and whether each one ended by itself
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    answers, ends = [None] * len(texts), [None] * len(texts)
    eos = torch.tensor(EOS, device=model.device)

    for start in range(0, len(order), GEN_BATCH):
        idx = order[start:start + GEN_BATCH]
        enc = tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.autocast(model.device.type, dtype=torch.bfloat16):
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0,
                                 eos_token_id=EOS, pad_token_id=EOS[1])
        for i, ids in zip(idx, out[:, enc.input_ids.shape[1]:]):
            answers[i] = tokenizer.decode(ids, skip_special_tokens=True)
            ends[i] = bool(torch.isin(ids, eos).any())

    return answers, ends


def score_rows(rows, answers):
    # right final answer (math) or fraction of tests passed (code). unlike the reward, a cut answer is not set to 0
    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(lambda r, a: float(is_right(a, r["answer"])) if r["domain"] == "math"
                             else score(a, r["test"], r["function_name"]), rows, answers))


def val_scores(model, tokenizer, val):
    # greedy scores on the val rows of each domain. "val" is their mean and picks the checkpoint
    model.eval()
    out = {}
    for d, rows in val.items():
        texts = [tokenizer.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in rows]
        answers, ends = generate(model, tokenizer, texts, VAL_MAX)
        s = score_rows(rows, answers)
        out[f"val_{d}"] = sum(s) / len(s)
        out[f"val_{d}_cut"] = 1 - sum(ends) / len(ends)
    out["val"] = sum(out[f"val_{d}"] for d in val) / len(val)
    return out


def math_right(answer, gold):
    # math-500: math-verify compares latex answers (fractions, intervals, ...), not only numbers
    try:
        return bool(verify(parse(f"${gold}$"), parse(answer)))
    except Exception:
        return False


def test_eval(model, tokenizer, name, out_dir):
    # every test set: gsm8k, mbpp(+) and humaneval(+) through bench_utils, then math-500 and held-out kodcode.
    # every answer is saved for the error analysis
    t0 = time.time()
    model = for_generation(model)
    with torch.autocast(model.device.type, dtype=torch.bfloat16):
        benchmark(model, tokenizer, name, out_dir, MAX_QUESTIONS)
    summary = json.load(open(f"{out_dir}/summary.json"))

    print(f"{name}: math-500")
    m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    answers, ends = generate(model, tokenizer, [chat(tokenizer, MATH_SYSTEM, ex["problem"]) for ex in m500], TEST_MAX)
    right = [math_right(a, ex["answer"]) for a, ex in zip(answers, m500)]
    write_jsonl(f"{out_dir}/math500.jsonl", [{"unique_id": ex["unique_id"], "subject": ex["subject"], "level": ex["level"],
                                              "gold": ex["answer"], "right": r, "ended": e, "answer": a}
                                             for ex, r, e, a in zip(m500, right, ends, answers)])

    print(f"{name}: held-out kodcode")
    kod = kodcode().select(HELD_OUT)
    rows = [code_row(ex) for ex in kod]
    answers, ends = generate(model, tokenizer, [tokenizer.apply_chat_template(r["prompt"], tokenize=False,
                                                add_generation_prompt=True) for r in rows], TEST_MAX)
    rewards = score_rows(rows, answers)
    write_jsonl(f"{out_dir}/heldout_code.jsonl", [{"question_id": ex["question_id"], "subset": ex["subset"],
                                                   "difficulty": ex["gpt_difficulty"], "reward": s, "ended": e, "answer": a}
                                                  for ex, s, e, a in zip(kod, rewards, ends, answers)])

    summary.update({"math500": sum(right) / len(right), "n_math500": len(right),
                    "heldout_code": sum(rewards) / len(rewards), "heldout_code_all_passed": sum(s == 1.0 for s in rewards) / len(rewards),
                    "n_heldout_code": len(rewards), "test_h": (time.time() - t0) / 3600,
                    # cpu load: a busy cpu can make code tests time out
                    "loadavg": os.getloadavg()[2], "cpus": os.cpu_count(),
                    # greedy answers depend on the gpu type
                    "gpu": torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else "cpu"})
    print(summary)
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f)
    return summary
