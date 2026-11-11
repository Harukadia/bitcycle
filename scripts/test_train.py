import json

import torch

from bitcycle.generate import recurrent_nll
from bitcycle.quant import BitLinear
from scripts.train import Config, eval_batch, evaluate, make_model, pack, train


def tiny(tmp_path, **kw):
    base = dict(out=str(tmp_path), run="t", precision="ternary", k=2, steps=6, batch=4, seq_len=8,
                vocab=32, d_model=16, layers=1, heads=2, ff=32, lr=1e-2, save_every=3, eval_every=0, seed=0)
    return Config(**{**base, **kw})


def data():
    g = torch.Generator().manual_seed(0)
    docs = [torch.randint(1, 32, (int(n),), generator=g).tolist() for n in torch.randint(3, 12, (40,), generator=g)]
    return pack(docs, eos=0, seq_len=8), eval_batch(docs, 6)


def test_pack_marks_the_first_token_of_each_document():
    inputs, reset = pack([[5], [6, 7], [8, 9, 10]], eos=0, seq_len=4)
    assert inputs.tolist() == [[5, 0, 6, 7], [0, 8, 9, 10]]
    assert reset.tolist() == [[False, False, True, False], [False, True, False, False]]


def test_eval_batch_keeps_documents_long_enough_and_truncates_them():
    assert eval_batch([[1, 2, 3], [4, 5], [6, 7, 8, 9]], 3).tolist() == [[1, 2, 3], [6, 7, 8]]


def test_make_model_quantizes_only_the_ternary_arm(tmp_path):
    ternary = make_model(tiny(tmp_path))
    full = make_model(tiny(tmp_path, precision="full"))
    assert any(isinstance(m, BitLinear) for m in ternary.base.modules())
    assert not any(isinstance(m, BitLinear) for m in full.base.modules())
    assert ternary.base.get_output_embeddings().weight is ternary.base.get_input_embeddings().weight


def test_evaluate_is_the_mean_recurrent_nll(tmp_path):
    model = make_model(tiny(tmp_path))
    torch.nn.init.normal_(model.adapter.proj.weight, std=0.5)
    _, val = data()
    assert torch.isclose(torch.tensor(evaluate(model, val, batch=3)), recurrent_nll(model, val).mean())


def test_resuming_matches_an_uninterrupted_run(tmp_path):
    (inputs, reset), val = data()
    whole = train(tiny(tmp_path / "whole"), inputs, reset, val)
    train(tiny(tmp_path / "split", session_steps=3, log_every=1), inputs, reset, val)
    resumed = train(tiny(tmp_path / "split", log_every=1), inputs, reset, val)
    for a, b in zip(whole.state_dict().values(), resumed.state_dict().values()):
        assert torch.equal(a, b)
    rows = [json.loads(line) for line in (tmp_path / "split" / "t.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows if r["type"] == "train"] == [1, 2, 3, 4, 5, 6]  # the second call went on from 3
    assert sum(r["type"] == "config" for r in rows) == 1


def test_run_log_records_config_training_and_final_eval(tmp_path):
    (inputs, reset), val = data()
    train(tiny(tmp_path, log_every=2), inputs, reset, val)
    rows = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert rows[0]["type"] == "config" and rows[0]["precision"] == "ternary"
    assert [r["step"] for r in rows if r["type"] == "train"] == [2, 4, 6]
    assert {"loss_1", "loss_2"} <= rows[1].keys()
    assert rows[-1]["type"] == "eval" and rows[-1]["step"] == 6
