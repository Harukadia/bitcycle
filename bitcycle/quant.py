"""BitNet b1.58 quantization-aware training: ternary weights, int8 activations.

Bit for bit the operations of transformers' online BitNet quantizer (WeightQuant /
ActQuant): computed in float32 and cast back, scales applied as reciprocals. Released
BitNet weights sit on rounding ties, where other orders flip ternary values. Unlike
transformers, float64 inputs stay float64, so float64 tests stay exact.
"""

import torch
import torch.nn.functional as F
from torch import nn


def _wide(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.promote_types(x.dtype, torch.float32))


def quantize_weights(w: torch.Tensor) -> torch.Tensor:
    """Ternary absmean quantization: values land on {-1, 0, +1} * mean(|w|)."""
    x = _wide(w)
    scale = 1 / x.abs().mean().clamp(min=1e-5)
    return ((x * scale).round().clamp(-1, 1) / scale).to(w.dtype)


def quantize_activations(x: torch.Tensor) -> torch.Tensor:
    """Per-token int8 absmax quantization over the last dimension."""
    y = _wide(x)
    scale = 127 / y.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    return ((y * scale).round().clamp(-128, 127) / scale).to(x.dtype)


class BitLinear(nn.Linear):
    """nn.Linear with fake-quantized weights and inputs; gradients pass straight through.

    No RMSNorm is inserted: BitNet architectures already carry their sub-norms.
    """

    frozen = False  # set by freeze_weights: the weight is already quantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + (quantize_activations(x) - x).detach()
        w = self.weight if self.frozen else self.weight + (quantize_weights(self.weight) - self.weight).detach()
        return F.linear(x, w, self.bias)


@torch.no_grad()
def freeze_weights(model: nn.Module) -> nn.Module:
    """Quantize every BitLinear weight once, in place, for a backbone that will not train.

    The stored value is the one forward computed, so outputs stay bit for bit the same;
    each forward then skips a pass over all the weights and its temporary copy.
    """
    for layer in model.modules():
        if isinstance(layer, BitLinear):
            layer.weight.copy_(layer.weight + (quantize_weights(layer.weight) - layer.weight))
            layer.frozen = True
    return model


def to_bitlinear(model: nn.Module, skip: tuple[str, ...] = ("lm_head",)) -> nn.Module:
    """Swap every nn.Linear (except names in `skip`) for a BitLinear sharing its parameters."""
    for parent in list(model.modules()):
        for name, child in parent.named_children():
            if type(child) is nn.Linear and name not in skip:
                layer = BitLinear(child.in_features, child.out_features, bias=child.bias is not None)
                layer.weight, layer.bias = child.weight, child.bias
                setattr(parent, name, layer)
    return model
