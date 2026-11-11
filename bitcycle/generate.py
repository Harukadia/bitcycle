"""Recycler inference: parallel prefill, then the exact recurrence with a KV cache.

k_prefill=1 matches a standard model's prefill cost; k_prefill=2 runs two prompt
passes and starts decoding from the second pass (better quality, 2x prefill).

Batches of different lengths: left-pad the prompts so they end in the same column and
pass attention_mask (1 = token, 0 = pad). Recycling restarts at each row's first token,
and positions count tokens only, so every row decodes as if it were alone. Tokens after
the prompt may be right-padded too (scoring continuations of different lengths).
"""

import torch
import torch.nn.functional as F

from bitcycle.recycler import RecyclerLM, recycle_shift


@torch.no_grad()
def prefill(model: RecyclerLM, input_ids, k_prefill: int = 1, attention_mask=None, logits_to_keep: int = 0):
    """Returns (logits, h, cache) of the last of k_prefill parallel passes over the prompt.

    logits_to_keep > 0 returns logits for only the last that many prompt positions (0: all).
    """
    padding = {}
    reset = None
    if attention_mask is not None:
        padding = dict(attention_mask=attention_mask, position_ids=(attention_mask.cumsum(-1) - 1).clamp(min=0))
        reset = F.pad(attention_mask[:, :-1] == 0, (1, 0))  # nothing to recycle after a pad
    recycled = None
    for i in range(k_prefill):
        last = i == k_prefill - 1
        logits, h, cache = model.pass_once(input_ids, recycled, use_cache=last,
                                           logits_to_keep=logits_to_keep if last else 1, **padding)
        recycled = recycle_shift(h, model.adapter, reset)
    return logits, h, cache


@torch.no_grad()
def step(model: RecyclerLM, tokens, h_prev, cache, attention_mask=None):
    """Decode one position: input = Emb(token) + A(h_prev). tokens: [B, 1], h_prev: [B, 1, D].

    attention_mask, if given, covers the cached positions and this one.
    """
    padding = {}
    if attention_mask is not None:
        padding = dict(attention_mask=attention_mask, position_ids=attention_mask.sum(-1, keepdim=True) - 1)
    return model.pass_once(tokens, model.adapter(h_prev), past_key_values=cache, use_cache=True, **padding)


@torch.no_grad()
def teacher_forced_logits(model: RecyclerLM, input_ids, *, prompt_len: int = 1, k_prefill: int = 1, attention_mask=None,
                          logits_to_keep: int = 0):
    """Logits for every position, decoding the tokens after the prompt one at a time.

    logits_to_keep > 0 keeps only the last that many prompt positions (0: all).
    """
    mask = None if attention_mask is None else attention_mask[:, :prompt_len]
    logits, h, cache = prefill(model, input_ids[:, :prompt_len], k_prefill, mask, logits_to_keep)
    out = [logits]
    for t in range(prompt_len, input_ids.shape[1]):
        mask = None if attention_mask is None else attention_mask[:, : t + 1]
        logits, h, cache = step(model, input_ids[:, t : t + 1], h[:, -1:], cache, mask)
        out.append(logits)
    return torch.cat(out, dim=1)


def recurrent_nll(model: RecyclerLM, input_ids, *, prompt_len: int = 1, k_prefill: int = 1, attention_mask=None):
    """Per-token NLL of the tokens after the prompt under the exact decode recurrence: [B, L - prompt_len].

    Right-padded positions get a value too; mask them with attention_mask[:, prompt_len:].
    """
    logits = teacher_forced_logits(model, input_ids, prompt_len=prompt_len, k_prefill=k_prefill,
                                   attention_mask=attention_mask, logits_to_keep=1)
    targets = input_ids[:, prompt_len:]
    return F.cross_entropy(logits[:, :-1].transpose(1, 2), targets, reduction="none")


@torch.no_grad()
def generate(model: RecyclerLM, input_ids, *, max_new_tokens: int, k_prefill: int = 1, eos_token_id=None,
             attention_mask=None):
    """Greedy decoding. Rows that emit eos keep emitting it; stops once every row has."""
    logits, h, cache = prefill(model, input_ids, k_prefill, attention_mask, logits_to_keep=1)
    out, mask = input_ids, attention_mask
    done = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)
    for i in range(max_new_tokens):
        token = logits[:, -1:].argmax(-1)
        if eos_token_id is not None:
            token = token.masked_fill(done[:, None], eos_token_id)
            done |= token[:, 0] == eos_token_id
        out = torch.cat([out, token], dim=1)
        if done.all() or i == max_new_tokens - 1:
            break
        if mask is not None:
            mask = F.pad(mask, (0, 1), value=1)
        logits, h, cache = step(model, token, h[:, -1:], cache, mask)
    return out
