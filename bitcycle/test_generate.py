import pytest
import torch

from bitcycle.conftest import top_states
from bitcycle.generate import generate, recurrent_nll, teacher_forced_logits
from bitcycle.multipass import multi_pass
from bitcycle.recycler import RecyclerLM, recycle_shift


def no_cache_logits(model, ids, prompt_len, k_prefill):
    """Reference decode: recompute the whole prefix at every step, no KV cache."""
    emb = model.base.get_input_embeddings()(ids)
    inputs = emb[:, :prompt_len]
    if k_prefill == 2:
        inputs = inputs + recycle_shift(top_states(model, inputs)[0], model.adapter)
    h, post = top_states(model, inputs)
    hs, posts = [h[:, -1]], list(post.unbind(1))
    for t in range(prompt_len, ids.shape[1]):
        inputs = torch.cat([inputs, emb[:, t : t + 1] + model.adapter(hs[-1])[:, None]], dim=1)
        h, post = (s[:, -1] for s in top_states(model, inputs))
        hs.append(h)
        posts.append(post)
    return model.base.get_output_embeddings()(torch.stack(posts, dim=1))


@pytest.mark.parametrize("k_prefill", [1, 2])
def test_cached_decode_matches_full_recompute(model, ids, k_prefill):
    with torch.no_grad():
        expected = no_cache_logits(model, ids, 4, k_prefill)
    got = teacher_forced_logits(model, ids, prompt_len=4, k_prefill=k_prefill)
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-10)


def test_training_with_prompt_reset_matches_single_pass_prefill(model, ids):
    reset = torch.zeros_like(ids, dtype=torch.bool)
    reset[:, :4] = True
    with torch.no_grad():
        *_, (trained, _) = multi_pass(model, ids, k=ids.shape[1], reset_mask=reset)
    decoded = teacher_forced_logits(model, ids, prompt_len=4, k_prefill=1)
    torch.testing.assert_close(trained, decoded, rtol=0, atol=1e-10)


def test_recurrent_nll_scores_tokens_after_the_prompt(model, ids):
    nll = recurrent_nll(model, ids, prompt_len=4)
    logits = teacher_forced_logits(model, ids, prompt_len=4)
    expected = -logits[:, 3:-1].log_softmax(-1).gather(-1, ids[:, 4:, None])[..., 0]
    assert nll.shape == (2, 5)
    assert torch.allclose(nll, expected)


@pytest.mark.parametrize("k_prefill", [1, 2])
def test_greedy_generate_follows_its_own_recurrent_logits(model, ids, k_prefill):
    out = generate(model, ids[:, :4], max_new_tokens=6, k_prefill=k_prefill)
    logits = teacher_forced_logits(model, out, prompt_len=4, k_prefill=k_prefill)
    assert out.shape == (2, 10)
    assert torch.equal(out[:, 4:], logits[:, 3:-1].argmax(-1))


def test_zero_adapter_generates_like_the_base_model(base, ids):
    out = generate(RecyclerLM(base), ids[:, :4], max_new_tokens=6)
    expected = ids[:, :4]
    with torch.no_grad():
        for _ in range(6):
            expected = torch.cat([expected, base(expected).logits[:, -1:].argmax(-1)], dim=1)
    assert torch.equal(out, expected)


def test_generate_stops_at_eos(model, ids):
    free = generate(model, ids[:1, :4], max_new_tokens=6)
    eos = free[0, 5].item()
    end = 4 + free[0, 4:].tolist().index(eos) + 1
    out = generate(model, ids[:1, :4], max_new_tokens=6, eos_token_id=eos)
    assert torch.equal(out, free[:, :end])


def test_finished_rows_keep_emitting_eos(model, ids):
    free = generate(model, ids[:, :4], max_new_tokens=6)
    eos = free[0, 4].item()
    out = generate(model, ids[:, :4], max_new_tokens=6, eos_token_id=eos)
    assert (out[0, 4:] == eos).all()
