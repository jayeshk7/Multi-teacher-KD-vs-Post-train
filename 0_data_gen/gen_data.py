# teachers write answers for the training prompts: math teacher on gsm8k train, code teacher on kodcode.
# saves token ids of the prompt, the teacher's answer and the dataset's gold answer to data/{domain}.jsonl.
# only the answer tokens are saved, distillation runs the teacher again on them to get its full distribution.

import json
import os
import re

import torch
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from transformers import AutoModelForCausalLM, AutoTokenizer

TEACHERS = {"math": "Qwen/Qwen2.5-Math-7B-Instruct", "code": "Qwen/Qwen2.5-Coder-7B-Instruct"}
SYSTEM = {
    "math": "Please reason step by step, and put your final answer within \\boxed{}.",  # math teacher's default
    "code": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",     # code teacher's default
}
TOTAL_PROMPTS = 7473          # prompts per domain (= all of gsm8k train). use 512 for a quick test
MAX_OUTPUT_LENGTH = 1600
BATCH_SIZE = 128
HF_HUB_DATA = "jajamken/FML_project" 
IM_END_TOKENID = 151645

# all teachers and the student share this tokenizer
tokenizer = AutoTokenizer.from_pretrained(TEACHERS["math"])
tokenizer.padding_side = "left"

gsm = load_dataset("openai/gsm8k", "main", split="train").select(range(TOTAL_PROMPTS))

# only select TOTAL_PROMPTS number of samples and not the entire dataset
kod = load_dataset("KodCode/KodCode-Light-RL-10K", split="train").shuffle(seed=0).select(range(TOTAL_PROMPTS))

prompts = { 
    "math": [math_sample["question"] for math_sample in gsm],
    # kodcode's own rl prompt (code-r1 repo): its tests call the declared function, so the name must match
    "code": ["Please solve the programming task below in Python. \n\n" + code_sample["question"].strip()
             + "\n\nNote that the function declaration is " + code_sample["test_info"][0]["function_declaration"].strip()
             + ". Your code should be wrapped in a markdown code block." for code_sample in kod],
}

# final number of each gsm8k gold answer, to check the math teacher's accuracy. the replace() is for numbers like 1,203
finals = [data["answer"].split("####")[-1].strip().replace(",", "") for data in gsm]

# ground truth answers for the ce loss. gsm8k: drop the <<48/2=24>> calculator notes and box the final number like the teacher
golds = {
    "math": [re.sub(r"<<.*?>>", "", ex["answer"].split("####")[0]).strip() + f"\nThe answer is \\boxed{{{f}}}." for ex, f in zip(gsm, finals)],
    "code": ["```python\n" + ex["solution"].strip() + "\n```" for ex in kod],
}


def is_right(text, gold):
    # same as in the bench scripts: last \boxed{...}, or the last number if nothing is boxed
    found = re.findall(r"\\boxed\{([^{}]*)\}", text) or re.findall(r"-?\d[\d,]*\.?\d*", text)
    try:
        return abs(float(re.sub(r"[^\d.\-]", "", found[-1])) - float(gold)) < 1e-6
    except (IndexError, ValueError):
        return False


os.makedirs("data", exist_ok=True)

for domain in ["math", "code"]:
    teacher = AutoModelForCausalLM.from_pretrained(TEACHERS[domain], dtype=torch.bfloat16).cuda().eval()
    texts = [tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM[domain]}, {"role": "user", "content": p}],
                                           tokenize=False,
                                           add_generation_prompt=True) for p in prompts[domain] ]

    # n_cut is the number of prompt responses that were cut off because it hit our MAX_OUTPUT_LENGTH. 
    rows, n_cut, n_correct = [], 0, 0

    # the cut answers themselves, saved separately to check later whether they were on the right track
    cut_rows = []

    for batch_start in range(0, TOTAL_PROMPTS, BATCH_SIZE):
        enc = tokenizer(texts[batch_start:batch_start + BATCH_SIZE], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = teacher.generate(**enc, max_new_tokens=MAX_OUTPUT_LENGTH, do_sample=False, repetition_penalty=1.0)

        # only the generated part of each row, the prompt columns are cut off
        responses = out[:, enc.input_ids.shape[1]:].tolist()

        for offset, response in enumerate(responses):
            # position of this prompt in gsm / kod, to look up the question and gold answer later
            index = batch_start + offset

            # no <|im_end|> means it hit MAX_OUTPUT_LENGTH, drop it
            if IM_END_TOKENID not in response:
                n_cut += 1
                cut_rows.append({"index": index, "response": tokenizer.decode(response, skip_special_tokens=True)})
                continue

            # drop the PAD tokens after <|im_end|> (rows that finished before the longest one in the batch)
            response = response[:response.index(IM_END_TOKENID) + 1]
            # drop the left PAD tokens of the prompt
            prompt = enc.input_ids[offset][enc.attention_mask[offset].bool()].tolist()

            gold = tokenizer(golds[domain][index], add_special_tokens=False).input_ids + [IM_END_TOKENID]
            rows.append({"prompt": prompt, "response": response, "gold": gold})

            if domain == "math":
                n_correct += is_right(tokenizer.decode(response), finals[index])

        print(f"{domain} {min(batch_start + BATCH_SIZE, TOTAL_PROMPTS)}/{TOTAL_PROMPTS}")

    # writing the final answers in a jsonl to be used for training
    with open(f"data/{domain}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # writing the cut answers in a separate jsonl for later analysis
    with open(f"data/{domain}_cut.jsonl", "w") as f:
        for row in cut_rows:
            f.write(json.dumps(row) + "\n")

    print(f"{domain}: kept {len(rows)}, cut {n_cut}")

    # answer lengths in tokens, to check that MAX_OUTPUT_LENGTH is enough
    for key in ["response", "gold"]:
        lens = sorted(len(row[key]) for row in rows)
        print(f"{domain} {key} tokens: median {lens[len(lens) // 2]}, p95 {lens[int(0.95 * len(lens))]}, max {lens[-1]}")

    if domain == "math":
        # sanity check for the prompt format. reported 95.2% on gsm8k test
        print(f"math teacher accuracy: {n_correct / (len(rows) + n_cut):.3f} (cut answers count as wrong)")

    del teacher
    torch.cuda.empty_cache()

if HF_HUB_DATA:
    create_repo(HF_HUB_DATA, repo_type="dataset", private=True, exist_ok=True)
    upload_folder(repo_id=HF_HUB_DATA, repo_type="dataset", folder_path="data")
