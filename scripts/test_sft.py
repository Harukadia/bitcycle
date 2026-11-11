import json

import pytest
import torch

from bitcycle.conftest import char_tokenizer, make_base
from bitcycle.generate import teacher_forced_logits
from scripts.sft import Config, collate, format_example, lora_rank, make_model, train

pytest.importorskip("peft")


def tiny(tmp_path, **kw):
    base = dict(out=str(tmp_path), arm="adapter", steps=4, batch=2, max_len=24, lr=1e-2, save_every=2, seed=0,
                log_every=1)
    return Config(**{**base, **kw})


def data():
    tok = char_tokenizer()
    return [format_example(tok, q, a, 24) for q, a in
            [("how many", "two. #### 2"), ("what is one plus one", "two"), ("ok", "yes, it is. #### 1")]]


def trainable(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


def test_format_example_masks_the_prompt_and_ends_with_eos():
    tok = char_tokenizer()
    ids, labels, reset = format_example(tok, "how many", "two", max_len=64)
    prompt = len(tok("Question: how many\nAnswer:").input_ids)
    assert len(ids) == len(labels) == len(reset) == prompt + len(" two") + 1
    assert labels[:prompt] == [-100] * prompt and labels[prompt:] == ids[prompt:]
    assert reset == [True] * prompt + [False] * (len(ids) - prompt)
    assert ids[-1] == tok.eos_token_id
    assert len(format_example(tok, "how many", "two", max_len=10)[0]) == 10


def test_collate_right_pads_without_loss():
    ids, labels, reset = collate([([1, 2, 3], [-100, 2, 3], [True, False, False]), ([4], [4], [False])])
    assert ids.tolist() == [[1, 2, 3], [4, 0, 0]]
    assert labels.tolist() == [[-100, 2, 3], [4, -100, -100]]
    assert reset[1].tolist() == [False, True, True]


@pytest.mark.parametrize("arm, has_adapter, has_lora", [
    ("lora", False, True), ("adapter", True, False), ("adapter_lora", True, True)])
def test_arms_train_only_their_own_parameters(tmp_path, arm, has_adapter, has_lora):
    model = make_model(tiny(tmp_path, arm=arm), make_base("bitnet_ternary"))
    names = trainable(model)
    assert any(n.startswith("adapter.") for n in names) == has_adapter
    assert any("lora_" in n for n in names) == has_lora
    assert all(n.startswith("adapter.") or "lora_" in n for n in names)
    assert all(p.dtype == torch.float64 for n, p in model.named_parameters() if n in names)  # never narrower


def test_untrained_arms_start_as_the_base_model(tmp_path):
    ids = torch.randint(0, 32, (2, 9), generator=torch.Generator().manual_seed(2))
    base = make_base("bitnet_ternary")
    expected = base(ids).logits
    model = make_model(tiny(tmp_path, arm="adapter_lora"), base)
    torch.testing.assert_close(teacher_forced_logits(model, ids, prompt_len=4), expected, rtol=0, atol=1e-10)


def test_lora_rank_matches_the_adapter_size():
    base = make_base("bitnet_ternary")
    d = base.config.hidden_size
    per_rank = sum(m.in_features + m.out_features for n, m in base.named_modules()
                   if n.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
    r = lora_rank(base)
    assert abs(r * per_rank - (d * d + d)) <= per_rank / 2


def test_run_names_show_the_passes_and_the_base_model_has_one_name():
    assert Config(arm="lora", k=2).passes() == 1 and Config(arm="adapter", k=2).passes() == 2
    assert Config(arm="lora").name() == "lora_k1_steps2000_seed0"
    assert Config(arm="base", seed=3).name() == "base"


def test_training_changes_only_the_arm_and_resumes_exactly(tmp_path):
    cfg = tiny(tmp_path / "whole", arm="adapter_lora")
    whole = train(cfg, make_model(cfg, make_base("bitnet_ternary")), data())
    base_after = make_base("bitnet_ternary")
    for n, p in whole.base.named_parameters():
        if not p.requires_grad:
            assert torch.equal(p, base_after.state_dict()[n.replace("base_layer.", "")])
    cfg = tiny(tmp_path / "split", arm="adapter_lora")
    train(Config(**{**cfg.__dict__, "session_steps": 2}), make_model(cfg, make_base("bitnet_ternary")), data())
    resumed = train(cfg, make_model(cfg, make_base("bitnet_ternary")), data())
    for (n, a), b in zip(whole.state_dict().items(), resumed.state_dict().values()):
        assert torch.equal(a, b), n
    rows = [json.loads(line) for line in (tmp_path / "split" / f"{cfg.name()}.jsonl").read_text().splitlines()]
    assert sum(r["type"] == "config" for r in rows) == 1
    assert [r["step"] for r in rows if r["type"] == "train"] == [1, 2, 3, 4]
