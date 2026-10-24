# distillation losses for the kd scripts

import torch
import torch.nn.functional as F

# the output layer has more rows (151936 for student and 152064 for teacher) than the tokenizer has tokens (151665)
# we slice the logits to the real 151665 tokens so the KL compares the same tokens
VOCAB_SIZE = 151665

# qwen's pad token <|endoftext|> as in its tokenizer_config.json. we pad the shorter sequences using this
PAD_TOKENID = 151643


def collate(prompts, answers):
    seqs = [p + a for p, a in zip(prompts, answers)]
    max_seq_len = max(len(s) for s in seqs)

    # 3 batch x max_seq_len matrices, filled per sequence below

    # token ids, PAD after the end of shorter sequences
    ids = torch.full((len(seqs), max_seq_len), PAD_TOKENID)

    # attention mask: 1 = real token, 0 = pad so no token attends to pads
    att = torch.zeros(len(seqs), max_seq_len, dtype=torch.long)

    # loss mask: True where the logit predicts an answer token
    mask = torch.zeros(len(seqs), max_seq_len, dtype=torch.bool)

    for i, (prompt, seq) in enumerate(zip(prompts, seqs)):
        ids[i, :len(seq)] = torch.tensor(seq)
        att[i, :len(seq)] = 1

        # we dont care about the token prediction for the prompt since that is already given
        # so we start from len_prompt-1 and go to len_seq-1
        # the answer tokens are predicted at positions len(prompt) - 1 to len(seq) - 2
        mask[i, len(prompt) - 1:len(seq) - 1] = True

    # still on the cpu, logits_at() moves them to each model's gpu
    return ids, att, mask


def logits_at(model, ids, att, mask, device):
    # output layer only on answer positions, saves a lot of memory with a 152k vocab
    ids, att, mask = ids.to(model.device), att.to(model.device), mask.to(model.device)

    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state

    # the losses are computed on the student's gpu
    return model.lm_head(h[mask])[:, :VOCAB_SIZE].to(device).float()


def losses(student, teacher, rows, temperature=1.0):
    # both summed over answer tokens: kl to the teacher on its own answers, ce on the gold answers
    # kl at temperature T (hinton et al. 2015): both logits divided by T, kl times T ^ 2 so its gradient keeps its size.
    # ce stays at T = 1
    prompts = [r["prompt"] for r in rows]
    ids, att, mask = collate(prompts, [r["response"] for r in rows])

    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            t = logits_at(teacher, ids, att, mask, student.device)
        s = logits_at(student, ids, att, mask, student.device)

    # skipped at T = 1, the division would allocate two more [tokens, 151665] fp32 tensors
    if temperature != 1:
        t, s = t / temperature, s / temperature

    t_logp = torch.log_softmax(t, -1)
    s_logp = torch.log_softmax(s, -1)
    # free each [tokens, 151665] fp32 tensor once it is used (~2.3 gb at 3865 tokens). backward keeps what it needs
    # (s_logp and the teacher probs), so the raw logits and t_logp can go before the gold forward
    del t, s
    kl = (t_logp.exp() * (t_logp - s_logp)).sum() * temperature ** 2
    del t_logp

    ids, att, mask = collate(prompts, [r["gold"] for r in rows])

    with torch.autocast("cuda", dtype=torch.bfloat16):
        s = logits_at(student, ids, att, mask, student.device)

    # the target at each answer position is the next token
    ce = F.cross_entropy(s, ids.roll(-1, dims=1)[mask].to(student.device), reduction="sum")

    return kl, ce


@torch.no_grad()
def val_losses(student, teacher, rows, micro):
    kl, ce = 0.0, 0.0

    for i in range(0, len(rows), micro):
        # val kl always at T = 1, so runs with different temperatures can be compared
        a, b = losses(student, teacher, rows[i:i + micro], temperature=1.0)
        kl += a.item()
        ce += b.item()

    # per-token means
    return kl / sum(len(r["response"]) for r in rows), ce / sum(len(r["gold"]) for r in rows)


def val_all(student, teachers, val, micro, step):
    # val kl and ce of both domains, each against its own teacher
    entry = {"step": step}

    for d in teachers:
        kl, ce = val_losses(student, teachers[d], val[d], micro)
        entry[f"{d}_kl"], entry[f"{d}_ce"] = kl, ce
        print(f"{d} val at step {step}: kl {kl:.4f}  ce {ce:.4f}")

    return entry
