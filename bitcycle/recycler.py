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
        # In the adapter's own dtype: it may train in float32 on a float16 backbone.
        return self.proj(self.norm(h.to(self.proj.weight.dtype)))


class RecyclerLM(nn.Module):
    """Wraps a Hugging Face causal LM and adds the recycle adapter.

    recycle picks the top-layer state fed back: "post_norm" (the decoder's output) or
    "pre_norm" (the residual stream entering the final norm, `decoder.norm`).
    """

    def __init__(self, base: nn.Module, recycle: str = "post_norm"):
        super().__init__()
        assert recycle in ("post_norm", "pre_norm"), recycle
        self.base = base
        self.recycle = recycle
        weight = base.get_input_embeddings().weight
        self.adapter = RecycleAdapter(base.config.hidden_size).to(weight.device, weight.dtype)

    def pass_once(self, input_ids, recycled=None, *, attention_mask=None, position_ids=None,
                  past_key_values=None, use_cache=False, logits_to_keep: int = 0):
        """One backbone pass. Returns (logits, the top-layer state to recycle, cache).

        logits_to_keep > 0 computes logits for only the last that many positions (0: all).
        """
        x = self.base.get_input_embeddings()(input_ids)
        if recycled is not None:
            x = x + recycled.to(x.dtype)
        decoder = self.base.get_decoder()
        seen = []
        hook = None
        if self.recycle == "pre_norm":
            hook = decoder.norm.register_forward_hook(lambda module, args, out: seen.append(args[0]))
        try:
            out = decoder(inputs_embeds=x, attention_mask=attention_mask, position_ids=position_ids,
                          past_key_values=past_key_values, use_cache=use_cache)
        finally:
            if hook is not None:
                hook.remove()
        top = out.last_hidden_state
        logits = self.base.get_output_embeddings()(top[:, -logits_to_keep:])
        return logits, seen[0] if seen else top, out.past_key_values


def recycle_shift(h: torch.Tensor, adapter: RecycleAdapter, reset_mask=None) -> torch.Tensor:
    """Recycled input per position: A(h[t-1]) at t, zero at t = 0 and where reset_mask is True."""
    r = F.pad(adapter(h[:, :-1]), (0, 0, 1, 0))
    if reset_mask is not None:
        r = r.masked_fill(reset_mask[..., None], 0)
    return r
