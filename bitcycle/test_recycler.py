import torch

from bitcycle.recycler import RecyclerLM, recycle_shift


def test_zero_init_adapter_reproduces_base_model(base, ids):
    model = RecyclerLM(base)
    expected = base(ids).logits
    logits, h, _ = model.pass_once(ids, None)
    assert torch.equal(logits, expected)
    logits2, _, _ = model.pass_once(ids, recycle_shift(h, model.adapter))
    assert torch.equal(logits2, expected)


def test_pass_once_returns_post_norm_top_hidden_state(base, ids):
    _, h, _ = RecyclerLM(base).pass_once(ids, None)
    assert torch.equal(h, base.get_decoder()(ids).last_hidden_state)


def test_pre_norm_recycles_the_state_before_the_final_norm(base, ids):
    logits, h, _ = RecyclerLM(base, recycle="pre_norm").pass_once(ids, None)
    decoder = base.get_decoder()
    post = decoder(ids).last_hidden_state
    assert not torch.allclose(h, post)
    assert torch.allclose(decoder.norm(h), post)
    assert torch.equal(logits, base(ids).logits)


def test_recycle_shift_feeds_previous_position(model, ids):
    _, h, _ = model.pass_once(ids, None)
    r = recycle_shift(h, model.adapter)
    assert torch.equal(r[:, 0], torch.zeros_like(r[:, 0]))
    assert torch.allclose(r[:, 1:], model.adapter(h[:, :-1]))


def test_recycle_shift_zeroes_reset_positions(model, ids):
    _, h, _ = model.pass_once(ids, None)
    reset = torch.zeros_like(ids, dtype=torch.bool)
    reset[:, 4] = True
    r = recycle_shift(h, model.adapter, reset)
    assert torch.equal(r[:, 4], torch.zeros_like(r[:, 4]))
    assert torch.allclose(r[:, 5], model.adapter(h[:, 4]))
