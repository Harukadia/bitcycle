"""Sequence-parallel training approximation of the Recycler recurrence (paper Eq. 4).

h(1) = T(Emb(x));  h(i)_t = T(Emb(x_t) + A(h(i-1)_{t-1}));  L = sum_i NTP(h(i)).
Pass i is exact for positions <= i, so k >= seq_len recovers the full recurrence.
"""

import torch
import torch.nn.functional as F

from bitcycle.recycler import RecyclerLM, recycle_shift


def multi_pass(model: RecyclerLM, input_ids, k: int = 2, reset_mask=None):
    """Yield (logits, h) for each of the k passes.

    With a frozen backbone, pass 1 cannot depend on any trainable parameter,
    so it runs without autograd: same gradients, half the activation memory.
    """
    frozen = not any(p.requires_grad for p in model.base.parameters())
    recycled = None
    for i in range(k):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not (i == 0 and frozen)):
            logits, h, _ = model.pass_once(input_ids, recycled)
        yield logits, h
        recycled = recycle_shift(h, model.adapter, reset_mask)


def multi_pass_loss(model: RecyclerLM, input_ids, labels, *, k: int = 2, reset_mask=None):
    """Sum of next-token cross-entropy over all passes; labels of -100 are ignored."""
    total, metrics = 0, {}
    for i, (logits, _) in enumerate(multi_pass(model, input_ids, k, reset_mask), start=1):
        loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten())
        total = total + loss
        metrics[f"loss_{i}"] = loss.item()
    return total, metrics
