# bitcycle

Hidden-state recycling for 1.58-bit LLMs.

bitcycle is an unofficial, open implementation of **Recycler** (ByteDance Seed,
*Recycler: Cost-Free Latent Reasoning via Hidden State Recycling*, 2026-10-25)
for Hugging Face causal LMs, with BitNet b1.58 quantization-aware training built in.

> **Status: Gate 0.** The implementation is complete and checked against exact
> references (see [Correctness](#correctness)). No trained checkpoints yet, and
> no claims about quality. Results land here as they are measured.

## Why

Standard decoding passes only a sampled token to the next step and throws the
top-layer hidden state away. Recycler keeps it:

```
input_t = Emb(x_t) + A(h_{t-1}),   A(h) = Linear(RMSNorm(h)),  Linear zero-initialized
```

The paper reports a model approaching a 2.4x larger baseline at the same decoding
cost and KV-cache size. Ternary (1.58-bit) weights cut memory and bandwidth but
lose capacity per parameter. The question bitcycle exists to answer: **can recycled
hidden states buy that capacity back, at no extra decoding cost?** As far as we
know, no published work combines the two.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install git+https://github.com/Harukadia/bitcycle
```

## Usage

```python
import torch
from transformers import AutoModelForCausalLM

from bitcycle.generate import generate, recurrent_nll
from bitcycle.multipass import multi_pass_loss
from bitcycle.quant import to_bitlinear
from bitcycle.recycler import RecyclerLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
to_bitlinear(base)              # optional: ternary QAT for every Linear except lm_head
model = RecyclerLM(base)        # adds the zero-initialized adapter: starts identical to base

ids = torch.randint(0, base.config.vocab_size, (2, 64))

# Training: k sequence-parallel passes, next-token loss summed over passes (paper Eq. 4)
loss, metrics = multi_pass_loss(model, ids, labels=ids, k=2)
loss.backward()

# Retrofit with a frozen backbone: only the adapter trains, pass 1 runs without autograd
base.requires_grad_(False)

# Inference: parallel prefill (k_prefill=1 or 2), then the exact recurrence with a KV cache
out = generate(model, ids[:, :8], max_new_tokens=32, k_prefill=1)

# Honest metric: NLL under the real decode recurrence, not the k-pass training loss
nll = recurrent_nll(model, ids, prompt_len=8)
```

For SFT with single-pass prefill, pass `reset_mask` marking prompt positions (and
`-100` labels on them) to `multi_pass_loss`; this matches `generate(..., k_prefill=1)`
exactly (tested). The same mask marks document starts in packed pretraining.

## Correctness

All tests run on CPU in float64 against tiny random Llama, Qwen2, and ternary
BitNet models, so any mismatch is a logic bug, not rounding.

| Property | Test |
|---|---|
| Zero-init adapter reproduces the base model exactly (logits and greedy outputs) | `test_recycler.py`, `test_generate.py` |
| k passes are exact for positions <= k; k = seq_len equals the token-by-token recurrence | `test_multipass.py` |
| Cached decode equals full recompute, for k_prefill = 1 and 2 | `test_generate.py` |
| Training with a prompt reset mask equals single-pass-prefill decoding | `test_generate.py` |
| Frozen backbone: only the adapter gets gradients; skipping autograd in pass 1 changes nothing | `test_multipass.py` |
| Ternary weights, per-token int8 activations, straight-through gradients | `test_quant.py` |

```bash
pip install -e ".[test]" && pytest -q
```

## Choices the paper leaves open

| Question | bitcycle default | Why |
|---|---|---|
| Recycle the top state before or after the final norm? | After (`last_hidden_state`) | No hooks needed; the adapter's own RMSNorm makes the two nearly equivalent |
| Quantize the adapter in ternary models? | No, keep it full precision | It is d x d (6.6M params at d = 2560). Keeping the backbone's ternary weights untouched lets existing 1-bit kernels run unchanged, and keeps quantization error out of the recurrence |
| Adapter bias, RMSNorm eps | No bias, eps 1e-6 | Smallest choice; zero-init then gives an exact identity |
| Packed sequences | Recycled state reset at document starts | Prevents state leaking across documents |

## Roadmap

1. Gate 0: implementation and exactness tests (this release)
2. Gate 1: tiny 2x2 study, {fp16, ternary} x {baseline, Recycler}
3. Gate 2: retrofit BitNet b1.58 2B4T with an adapter-only SFT and evaluate on math and reasoning
4. Gate 3: continued pretraining retrofit (compute permitting)
5. Gate 4: CPU inference with bitnet.cpp

## Not affiliated

This project is not affiliated with or endorsed by ByteDance. It is written from
the public paper. The method may be covered by patents; the Apache-2.0 license
here only covers contributions to this repository. This is not legal advice.

## Citation

Cite the original paper:

```
ByteDance Seed. Recycler: Cost-Free Latent Reasoning via Hidden State Recycling. 2026.
https://seed.bytedance.com/en/public_papers/recycler-cost-free-latent-reasoning-via-hidden-state-recycling
```

## License

Apache-2.0
