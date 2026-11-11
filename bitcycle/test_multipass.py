import torch
import torch.nn.functional as F

from bitcycle.multipass import multi_pass, multi_pass_loss
from bitcycle.recycler import RecyclerLM, recycle_shift


def exact_recurrence_logits(model, ids, reset_mask=None):
    """Eq. 3 token by token: recompute the prefix without cache, feed A(h_{t-1}) into step t."""
    emb = model.base.get_input_embeddings()(ids)
    inputs, hs = emb[:, :1], []
    for t in range(ids.shape[1]):
        h = model.base.get_decoder()(inputs_embeds=inputs).last_hidden_state[:, -1]
        hs.append(h)
        if t + 1 < ids.shape[1]:
            r = model.adapter(h)
            if reset_mask is not None:
                r = r.masked_fill(reset_mask[:, t + 1, None], 0)
            inputs = torch.cat([inputs, emb[:, t + 1 : t + 2] + r[:, None]], dim=1)
    return model.base.get_output_embeddings()(torch.stack(hs, dim=1))


def ce(logits, ids):
    return F.cross_entropy(logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten())


def test_enough_passes_recover_the_exact_recurrence(model, ids):
    with torch.no_grad():
        *_, (logits, _) = multi_pass(model, ids, k=ids.shape[1])
        assert torch.allclose(logits, exact_recurrence_logits(model, ids), atol=1e-10)


def test_pass_k_is_exact_up_to_position_k(model, ids):
    with torch.no_grad():
        *_, (logits, _) = multi_pass(model, ids, k=3)
        exact = exact_recurrence_logits(model, ids)
    assert torch.allclose(logits[:, :3], exact[:, :3], atol=1e-10)
    assert not torch.allclose(logits[:, 3:], exact[:, 3:], atol=1e-10)


def test_reset_mask_restarts_the_recurrence(model, ids):
    reset = torch.zeros_like(ids, dtype=torch.bool)
    reset[:, 5] = True
    with torch.no_grad():
        *_, (logits, _) = multi_pass(model, ids, k=ids.shape[1], reset_mask=reset)
        assert torch.allclose(logits, exact_recurrence_logits(model, ids, reset), atol=1e-10)


def test_loss_sums_next_token_loss_over_passes(model, ids):
    loss, metrics = multi_pass_loss(model, ids, ids, k=2)
    passes = [ce(logits, ids) for logits, _ in multi_pass(model, ids, k=2)]
    assert torch.allclose(loss, passes[0] + passes[1])
    assert metrics["loss_1"] == passes[0].item() and metrics["loss_2"] == passes[1].item()


def test_loss_ignores_masked_labels(model, ids):
    labels = ids.clone()
    labels[:, :5] = -100
    loss, _ = multi_pass_loss(model, ids, labels, k=1)
    (logits, _), = multi_pass(model, ids, k=1)
    assert torch.allclose(loss, ce(logits[:, 4:], ids[:, 4:]))


def test_frozen_backbone_trains_only_the_adapter(model, ids):
    model.base.requires_grad_(False)
    loss, _ = multi_pass_loss(model, ids, ids, k=2)
    loss.backward()
    assert all(p.grad is None for p in model.base.parameters())

    # Reference with pass 1 kept in the graph: gradients must be identical.
    ref = model.adapter.proj.weight.grad.clone()
    model.zero_grad()
    logits1, h1, _ = model.pass_once(ids, None)
    logits2, _, _ = model.pass_once(ids, recycle_shift(h1, model.adapter))
    (ce(logits1, ids) + ce(logits2, ids)).backward()
    assert torch.allclose(ref, model.adapter.proj.weight.grad)
