import pytest
import torch
from transformers import AutoModelForCausalLM, BitNetConfig, LlamaConfig, Qwen2Config

from bitcycle.quant import to_bitlinear
from bitcycle.recycler import RecyclerLM

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
    config._attn_implementation = "eager"
    base = AutoModelForCausalLM.from_config(config).double().eval()
    if arch.endswith("_ternary"):
        to_bitlinear(base)
    return base


def randomize_adapter(model: RecyclerLM) -> RecyclerLM:
    """Non-zero adapter so recycled states actually change the computation."""
    torch.manual_seed(1)
    torch.nn.init.normal_(model.adapter.proj.weight, std=0.5)
    return model


@pytest.fixture(params=list(CONFIGS))
def base(request):
    return make_base(request.param)


@pytest.fixture
def model(base):
    return randomize_adapter(RecyclerLM(base))


@pytest.fixture
def ids():
    torch.manual_seed(2)
    return torch.randint(0, TINY["vocab_size"], (2, 9))
