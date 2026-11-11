import torch
from torch import nn

from bitcycle.quant import BitLinear, quantize_activations, quantize_weights, to_bitlinear


def test_weights_are_ternary_times_absmean():
    w = torch.randn(16, 32, dtype=torch.float64)
    scale = w.abs().mean()
    q = quantize_weights(w)
    assert set((q / scale).round().unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.allclose(q / scale, (q / scale).round())


def test_activations_fit_int8_per_token():
    x = torch.randn(2, 5, 32, dtype=torch.float64)
    q = quantize_activations(x)
    scale = 127 / x.abs().amax(dim=-1, keepdim=True)
    codes = q * scale
    assert torch.allclose(codes, codes.round())
    assert codes.min() >= -128 and codes.max() <= 127


def test_activation_quantization_is_per_token():
    x = torch.randn(1, 4, 32, dtype=torch.float64)
    y = x.clone()
    y[0, 2] *= 100
    qx, qy = quantize_activations(x), quantize_activations(y)
    keep = [0, 1, 3]
    assert torch.equal(qx[0, keep], qy[0, keep])


def test_bitlinear_passes_gradient_straight_through():
    layer = BitLinear(32, 16, bias=False).double()
    x = torch.randn(3, 32, dtype=torch.float64, requires_grad=True)
    layer(x).sum().backward()
    w_q = quantize_weights(layer.weight.detach())
    x_q = quantize_activations(x.detach())
    assert torch.allclose(layer.weight.grad, torch.ones(3, 16, dtype=torch.float64).T @ x_q)
    assert torch.allclose(x.grad, torch.ones(3, 16, dtype=torch.float64) @ w_q)


def test_to_bitlinear_keeps_lm_head_full_precision():
    model = nn.ModuleDict({"proj": nn.Linear(8, 8), "lm_head": nn.Linear(8, 4)})
    to_bitlinear(model)
    assert type(model["proj"]) is BitLinear
    assert type(model["lm_head"]) is nn.Linear
