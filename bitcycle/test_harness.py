import pytest
import torch

from bitcycle.generate import generate, recurrent_nll, teacher_forced_logits

pytest.importorskip("lm_eval")
from lm_eval.api.instance import Instance  # noqa: E402

from bitcycle.harness import RecyclerEval  # noqa: E402


def requests(kind, arguments):
    return [Instance(request_type=kind, doc={}, arguments=a, idx=i) for i, a in enumerate(arguments)]


def test_loglikelihood_scores_each_continuation_under_the_recurrence(model, tok):
    # Contexts and continuations of different lengths, a trailing space, an empty context.
    pairs = [("the cat", " sat"), ("a", " dog ran"), ("is it ", "here?"), ("", "hi")]
    got = RecyclerEval(model, tok, batch_size=3).loglikelihood(requests("loglikelihood", pairs))
    for (context, continuation), (logprob, greedy) in zip(pairs, got):
        stripped = context.rstrip()  # lm-eval moves the context's trailing spaces into the continuation
        ctx = tok(stripped).input_ids or [tok.eos_token_id]
        cont = tok(context[len(stripped) :] + continuation).input_ids
        ids = torch.tensor([ctx + cont])
        expected = -recurrent_nll(model, ids, prompt_len=len(ctx)).sum().item()
        predicted = teacher_forced_logits(model, ids, prompt_len=len(ctx))[0, len(ctx) - 1 : -1].argmax(-1)
        assert logprob == pytest.approx(expected, rel=1e-9)
        assert greedy == bool((predicted == torch.tensor(cont)).all())


def test_generate_until_decodes_greedily_and_cuts_at_the_stop_string(model, tok):
    contexts = ["the cat", "a", "is it"]

    def alone(context):
        out = generate(model, torch.tensor([tok(context).input_ids]), max_new_tokens=8,
                       eos_token_id=tok.eos_token_id)[0, len(context) :].tolist()
        return tok.decode(out[: out.index(0)] if 0 in out else out)

    free = [alone(c) for c in contexts]
    assert free[0], "the first row should generate some text to cut"
    stop = free[0][len(free[0]) // 2]
    lm = RecyclerEval(model, tok, batch_size=2)
    got = lm.generate_until(requests("generate_until", [(c, {"until": [stop], "max_gen_toks": 8}) for c in contexts]))
    assert got == [f.split(stop)[0] for f in free]


def test_a_batch_that_runs_out_of_memory_is_retried_at_half_the_size(model, tok):
    sizes = []

    def run(items):
        sizes.append(len(items))
        if len(items) > 2:
            raise torch.OutOfMemoryError
        return items

    assert RecyclerEval(model, tok, batch_size=4)._batched([3, 1, 2, 5, 4], run, lambda x: x) == [3, 1, 2, 5, 4]
    assert sizes == [4, 2, 2, 1]
