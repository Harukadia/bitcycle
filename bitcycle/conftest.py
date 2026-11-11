import pytest
import torch
from transformers import AttentionInterface, AutoModelForCausalLM, BitNetConfig, LlamaConfig, Qwen2Config
from transformers.masking_utils import AttentionMaskInterface, eager_mask

from bitcycle.quant import to_bitlinear
from bitcycle.recycler import RecyclerLM

# Hugging Face computes RMSNorm and the eager softmax in float32 even in float64 models.
# Two equivalent computation orders then differ by float32 rounding on some CPUs (CI failed
# at 1e-10; ternary models can even flip an int8 activation). The tests run everything in
# float64 instead, so a mismatch is a logic bug, not rounding.


def float64_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Eager attention (repeat_kv, scaled scores, mask, softmax) without the float32 softmax."""
    key = key.repeat_interleave(module.num_key_value_groups, dim=1)
    value = value.repeat_interleave(module.num_key_value_groups, dim=1)
    weights = query @ key.transpose(2, 3) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask
    weights = weights.softmax(-1)
    return (weights @ value).transpose(1, 2).contiguous(), weights


AttentionInterface.register("float64", float64_attention)
AttentionMaskInterface.register("float64", eager_mask)  # the same additive causal mask as eager


def float64_norm(norm, x):
    """Hugging Face RMSNorm without the float32 round trip."""
    return norm.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon))

TINY = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
    tie_word_embeddings=True,
    bos_token_id=None,
    eos_token_id=None,
)

CONFIGS = {
    "llama": lambda: LlamaConfig(**TINY),
    "qwen2": lambda: Qwen2Config(**TINY),
    "bitnet_ternary": lambda: BitNetConfig(**TINY),
}


def make_base(arch: str):
    torch.manual_seed(0)
    config = CONFIGS[arch]()
    config._attn_implementation = "float64"
    base = AutoModelForCausalLM.from_config(config).double().eval()
    for module in base.modules():
        if hasattr(module, "variance_epsilon"):  # the model's RMSNorms
            module.forward = lambda x, norm=module: float64_norm(norm, x)
    if arch.endswith("_ternary"):
        to_bitlinear(base)
    return base


def randomize_adapter(model: RecyclerLM) -> RecyclerLM:
    """Non-zero adapter so recycled states actually change the computation."""
    torch.manual_seed(1)
    torch.nn.init.normal_(model.adapter.proj.weight, std=0.5)
    return model


def top_states(model: RecyclerLM, inputs_embeds):
    """Reference for tests: (the state `model` recycles, the post-norm state) at every position."""
    decoder = model.base.get_decoder()
    seen = []
    hook = decoder.norm.register_forward_hook(lambda module, args, out: seen.append(args[0]))
    post = decoder(inputs_embeds=inputs_embeds).last_hidden_state
    hook.remove()
    return (seen[0] if model.recycle == "pre_norm" else post), post


@pytest.fixture(params=list(CONFIGS))
def base(request):
    return make_base(request.param)


@pytest.fixture(params=["post_norm", "pre_norm"])
def model(base, request):
    return randomize_adapter(RecyclerLM(base, recycle=request.param))


@pytest.fixture
def ids():
    torch.manual_seed(2)
    return torch.randint(0, TINY["vocab_size"], (2, 9))
