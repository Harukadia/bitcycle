"""Recycler inference: parallel prefill, then the exact recurrence with a KV cache.

k_prefill=1 matches a standard model's prefill cost; k_prefill=2 runs two prompt
passes and starts decoding from the second pass (better quality, 2x prefill).
"""

import torch
import torch.nn.functional as F

from bitcycle.recycler import RecyclerLM, recycle_shift


@torch.no_grad()
def prefill(model: RecyclerLM, input_ids, k_prefill: int = 1):
    """Returns (logits, h, cache) of the last of k_prefill parallel passes over the prompt."""
    recycled = None
    for i in range(k_prefill):
        logits, h, cache = model.pass_once(input_ids, recycled, use_cache=i == k_prefill - 1)
        recycled = recycle_shift(h, model.adapter)
    return logits, h, cache


@torch.no_grad()
def step(model: RecyclerLM, tokens, h_prev, cache):
    """Decode one position: input = Emb(token) + A(h_prev). tokens: [B, 1], h_prev: [B, 1, D]."""
    return model.pass_once(tokens, model.adapter(h_prev), past_key_values=cache, use_cache=True)


@torch.no_grad()
def teacher_forced_logits(model: RecyclerLM, input_ids, *, prompt_len: int = 1, k_prefill: int = 1):
    """Logits for every position, decoding the tokens after the prompt one at a time."""
    logits, h, cache = prefill(model, input_ids[:, :prompt_len], k_prefill)
    out = [logits]
    for t in range(prompt_len, input_ids.shape[1]):
        logits, h, cache = step(model, input_ids[:, t : t + 1], h[:, -1:], cache)
        out.append(logits)
    return torch.cat(out, dim=1)


def recurrent_nll(model: RecyclerLM, input_ids, *, prompt_len: int = 1, k_prefill: int = 1):
    """Per-token NLL of the tokens after the prompt under the exact decode recurrence: [B, L - prompt_len]."""
    logits = teacher_forced_logits(model, input_ids, prompt_len=prompt_len, k_prefill=k_prefill)
    targets = input_ids[:, prompt_len:]
    return F.cross_entropy(logits[:, prompt_len - 1 : -1].transpose(1, 2), targets, reduction="none")


@torch.no_grad()
def generate(model: RecyclerLM, input_ids, *, max_new_tokens: int, k_prefill: int = 1, eos_token_id=None):
    """Greedy decoding. Rows that emit eos keep emitting it; stops once every row has."""
    logits, h, cache = prefill(model, input_ids, k_prefill)
    out = input_ids
    done = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)
    for i in range(max_new_tokens):
        token = logits[:, -1:].argmax(-1)
        if eos_token_id is not None:
            token = token.masked_fill(done[:, None], eos_token_id)
            done |= token[:, 0] == eos_token_id
        out = torch.cat([out, token], dim=1)
        if done.all() or i == max_new_tokens - 1:
            break
        logits, h, cache = step(model, token, h[:, -1:], cache)
    return out
