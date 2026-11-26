# benchmark a model on math (gsm8k test) and python (mbpp, mbpp+, humaneval, humaneval+) using greedy decoding, pass@1.
# pass@1 is the fraction of problems the model solves in only 1 attempt. since we use greedy decoding our model only gets one chance

import json
import multiprocessing
import os
import re

import torch
from datasets import load_dataset
from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
from evalplus.evaluate import evaluate
from evalplus.sanitize import sanitize

MAX_OUTPUT_LENGTH = 1600
GEN_BATCH = 128   # prompts per generate call

# every model gets these same prompts
MATH_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."  # math teacher's default
CODE_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."     # code teacher's default

# evalplus's prompt for chat models: the answer is prefilled up to the start of a python code block
CODE_ASK = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
CODE_PREFILL = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"

# evalplus runs the tests in worker processes. fork them, python 3.14+ defaults to forkserver which reruns the main script
multiprocessing.set_start_method("fork")


def chat(tokenizer, system, user, prefill=""):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefill


def generate(model, tokenizer, prompts):
    # needs tokenizer.padding_side = "left"
    # shortest prompts first so each batch has little padding, answers go back in the original order
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    answers = [None] * len(prompts)

    for start in range(0, len(order), GEN_BATCH):
        idx = order[start:start + GEN_BATCH]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

        with torch.no_grad():
            # qwen instruct models ship repetition_penalty 1.1 in their generation config, turn it off for plain greedy decoding
            out = model.generate(**enc, max_new_tokens=MAX_OUTPUT_LENGTH, do_sample=False, repetition_penalty=1.0)

        for i, ids in zip(idx, out[:, enc.input_ids.shape[1]:]):
            answers[i] = tokenizer.decode(ids, skip_special_tokens=True)

        print(f"  {min(start + GEN_BATCH, len(prompts))}/{len(prompts)}")

    return answers


def is_right(text, gold):
    # last \boxed{...}, or the last number if nothing is boxed
    found = re.findall(r"\\boxed\{([^{}]*)\}", text) or re.findall(r"-?\d[\d,]*\.?\d*", text)
    try:
        return abs(float(re.sub(r"[^\d.\-]", "", found[-1])) - float(gold)) < 1e-6
    except (IndexError, ValueError):
        return False


def evalplus_bench(model, tokenizer, dataset, problems, out_dir, max_questions):
    # dataset is "mbpp" or "humaneval". returns pass/fail on the original tests (base) and on base + extra tests (plus)
    asked = problems[:max_questions]
    prompts = [chat(tokenizer, CODE_SYSTEM, f"{CODE_ASK}\n```\n{p['prompt'].strip()}\n```\n", f"{CODE_PREFILL}\n```python\n")
               for p in asked]

    raw = dict(zip([p["task_id"] for p in asked], generate(model, tokenizer, prompts)))

    # the code ends where the model closes the code block. evalplus needs a row for every task, skipped ones get no code
    samples = []
    for p in problems:
        answer = raw.get(p["task_id"], "")
        samples.append({"task_id": p["task_id"], "solution": sanitize(answer.split("```")[0], p["entry_point"]), "raw": answer})

    write_jsonl(f"{out_dir}/{dataset}.jsonl", samples)

    # evalplus reuses an old results file if there is one, delete it so a rerun is scored fresh
    res_path = f"{out_dir}/{dataset}_eval_results.json"
    if os.path.exists(res_path):
        os.remove(res_path)

    evaluate(dataset, samples=f"{out_dir}/{dataset}.jsonl")

    res = json.load(open(res_path))["eval"]
    base = [res[p["task_id"]][0]["base_status"] == "pass" for p in asked]
    plus = [res[p["task_id"]][0]["base_status"] == res[p["task_id"]][0]["plus_status"] == "pass" for p in asked]

    print(f"{dataset} {sum(base) / len(base):.3f}, {dataset}+ {sum(plus) / len(plus):.3f} on {len(asked)} tasks")

    return base, plus


def benchmark(model, tokenizer, name, out_dir, max_questions):
    # writes the answers, evalplus's results and summary.json to out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ---- math: gsm8k test ----
    print(f"{name}: gsm8k")
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm = gsm.select(range(min(max_questions, len(gsm))))

    answers = generate(model, tokenizer, [chat(tokenizer, MATH_SYSTEM, sample["question"]) for sample in gsm])
    golds = [sample["answer"].split("####")[-1].strip().replace(",", "") for sample in gsm]
    right = [is_right(a, g) for a, g in zip(answers, golds)]

    write_jsonl(f"{out_dir}/gsm8k.jsonl", [{"right": r, "gold": g, "answer": a} for r, g, a in zip(right, golds, answers)])
    print(f"gsm8k accuracy {sum(right) / len(right):.3f} on {len(right)} questions")

    # ---- python: mbpp+ and humaneval+ with evalplus ----
    print(f"{name}: mbpp+")
    mbpp_base, mbpp_plus = evalplus_bench(model, tokenizer, "mbpp", list(get_mbpp_plus().values()), out_dir, max_questions)

    print(f"{name}: humaneval+")
    he_base, he_plus = evalplus_bench(model, tokenizer, "humaneval", list(get_human_eval_plus().values()), out_dir, max_questions)

    summary = {"model": name, "gsm8k": sum(right) / len(right),
               "mbpp": sum(mbpp_base) / len(mbpp_base), "mbpp_plus": sum(mbpp_plus) / len(mbpp_plus),
               "humaneval": sum(he_base) / len(he_base), "humaneval_plus": sum(he_plus) / len(he_plus),
               "n_gsm8k": len(right), "n_mbpp": len(mbpp_base), "n_humaneval": len(he_base)}
    print(summary)

    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f)
