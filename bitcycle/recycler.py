"""Recycler: feed the previous token's top-layer hidden state into the next step.

next input = Emb(x_t) + A(h_{t-1}),  A(h) = Linear(RMSNorm(h)), Linear zero-initialized
(ByteDance Seed, "Recycler: Cost-Free Latent Reasoning via Hidden State Recycling", 2026).
"""

import torch
import torch.nn.functional as F
from torch import nn


class RecycleAdapter(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.proj.weight)  # model == base model at init

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(h))


class RecyclerLM(nn.Module):
    """Wraps a Hugging Face causal LM and adds the recycle adapter."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        weight = base.get_input_embeddings().weight
        self.adapter = RecycleAdapter(base.config.hidden_size).to(weight.device, weight.dtype)

    def pass_once(self, input_ids, recycled=None, *, past_key_values=None, use_cache=False):
        """One backbone pass. Returns (logits, post-norm top hidden state, cache)."""
        x = self.base.get_input_embeddings()(input_ids)
        if recycled is not None:
            x = x + recycled
        out = self.base.get_decoder()(inputs_embeds=x, past_key_values=past_key_values, use_cache=use_cache)
        h = out.last_hidden_state
        return self.base.get_output_embeddings()(h), h, out.past_key_values


def recycle_shift(h: torch.Tensor, adapter: RecycleAdapter, reset_mask=None) -> torch.Tensor:
    """Recycled input per position: A(h[t-1]) at t, zero at t = 0 and where reset_mask is True."""
    r = F.pad(adapter(h[:, :-1]), (0, 0, 1, 0))
    if reset_mask is not None:
        r = r.masked_fill(reset_mask[..., None], 0)
    return r
