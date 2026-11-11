"""BitNet b1.58 quantization-aware training: ternary weights, int8 activations.

Matches transformers' online BitNet quantizer (WeightQuant / ActQuant), but
computes in the input dtype so float64 tests stay exact.
"""

import torch
import torch.nn.functional as F
from torch import nn


def quantize_weights(w: torch.Tensor) -> torch.Tensor:
    """Ternary absmean quantization: values land on {-1, 0, +1} * mean(|w|)."""
    scale = w.abs().mean().clamp(min=1e-5)
    return (w / scale).round().clamp(-1, 1) * scale


def quantize_activations(x: torch.Tensor) -> torch.Tensor:
    """Per-token int8 absmax quantization over the last dimension."""
    scale = 127 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    return (x * scale).round().clamp(-128, 127) / scale


class BitLinear(nn.Linear):
    """nn.Linear with fake-quantized weights and inputs; gradients pass straight through.

    No RMSNorm is inserted: BitNet architectures already carry their sub-norms.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + (quantize_activations(x) - x).detach()
        w = self.weight + (quantize_weights(self.weight) - self.weight).detach()
        return F.linear(x, w, self.bias)


def to_bitlinear(model: nn.Module, skip: tuple[str, ...] = ("lm_head",)) -> nn.Module:
    """Swap every nn.Linear (except names in `skip`) for a BitLinear sharing its parameters."""
    for parent in list(model.modules()):
        for name, child in parent.named_children():
            if type(child) is nn.Linear and name not in skip:
                layer = BitLinear(child.in_features, child.out_features, bias=child.bias is not None)
                layer.weight, layer.bias = child.weight, child.bias
                setattr(parent, name, layer)
    return model
