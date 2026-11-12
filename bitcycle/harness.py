"""lm-evaluation-harness adapter: score and generate with a RecyclerLM's exact decode recurrence.

    import lm_eval
    lm_eval.simple_evaluate(model=RecyclerEval(model, tokenizer), tasks=["gsm8k", "arc_challenge"])

Requires the `eval` extra (lm-eval). Requests are sorted by length and run in batches:
prompts left-padded, continuations right-padded (bitcycle.generate).
"""

import torch
from lm_eval.api.model import LM

from bitcycle.generate import generate, teacher_forced_logits
from bitcycle.recycler import RecyclerLM


class RecyclerEval(LM):
    def __init__(self, model: RecyclerLM, tokenizer, *, batch_size: int = 8, k_prefill: int = 1,
                 max_gen_toks: int = 256):
        super().__init__()
        self.model, self.tokenizer = model.eval(), tokenizer
        self.batch_size, self.k_prefill, self.max_gen_toks = batch_size, k_prefill, max_gen_toks

    def _context(self, text: str) -> list[int]:
        """Token ids of a context; an empty one (with no BOS) is conditioned on eos, as lm-eval does."""
        return self.tokenizer(text).input_ids or [self.tokenizer.eos_token_id]

    def _pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        """lm-eval's split: the context's trailing spaces move into the continuation."""
        stripped = context.rstrip()
        continuation = context[len(stripped) :] + continuation
        whole = self.tokenizer(stripped + continuation).input_ids
        ctx = self.tokenizer(stripped).input_ids
        return (ctx, whole[len(ctx) :]) if ctx else ([self.tokenizer.eos_token_id], whole)

    def _batched(self, kind, requests, items, run, length):
        """run() on batches of items, longest first; results in the original order.

        A batch that runs out of GPU memory is retried at half the size, which then stays.
        Each finished batch goes to lm-eval's response cache (simple_evaluate(use_cache=...)),
        so an evaluation that is cut short resumes where it stopped.
        """
        order = sorted(range(len(items)), key=lambda i: -length(items[i]))
        results = [None] * len(items)
        start, size = 0, self.batch_size
        while start < len(order):
            chunk = order[start : start + size]
            try:
                out = run([items[i] for i in chunk])
            except torch.OutOfMemoryError:
                if size == 1:
                    raise
                out = None
            if out is None:  # outside the handler, so the failed batch's tensors are already released
                size //= 2
                torch.cuda.empty_cache()
                continue
            for i, result in zip(chunk, out):
                results[i] = result
                self.cache_hook.add_partial(kind, requests[i].args, result)
            start += len(chunk)
            print(f"eval {start}/{len(order)} batch {size}", flush=True)
        return results

    def _padded(self, rows, width, length):
        """[B, length] ids and mask: each row's prompt ends at column `width`, its rest follows."""
        ids = torch.zeros(len(rows), length, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, (prompt, rest) in enumerate(rows):
            span = slice(width - len(prompt), width + len(rest))
            ids[i, span] = torch.tensor(prompt + rest)
            mask[i, span] = 1
        device = next(self.model.parameters()).device
        return ids.to(device), mask.to(device)

    def _score(self, pairs):
        width = max(len(c) for c, _ in pairs)
        ids, mask = self._padded(pairs, width, width + max(len(x) for _, x in pairs))
        logits = teacher_forced_logits(self.model, ids, prompt_len=width, k_prefill=self.k_prefill,
                                       attention_mask=mask, logits_to_keep=1)
        logprobs = logits.to(torch.promote_types(logits.dtype, torch.float32)).log_softmax(-1).cpu()
        out = []
        for i, (_, cont) in enumerate(pairs):
            target, lp = torch.tensor(cont), logprobs[i, : len(cont)]
            out.append((lp.gather(-1, target[:, None]).sum().item(), bool((lp.argmax(-1) == target).all())))
        return out

    def loglikelihood(self, requests):
        pairs = [self._pair(*r.args) for r in requests]
        return self._batched("loglikelihood", requests, pairs, self._score, lambda p: len(p[0]) + len(p[1]))

    def _generate(self, jobs):
        width = max(len(c) for c, _ in jobs)
        ids, mask = self._padded([(c, []) for c, _ in jobs], width, width)
        limits = [kw.get("max_gen_toks", self.max_gen_toks) for _, kw in jobs]
        eos = self.tokenizer.eos_token_id
        out = generate(self.model, ids, max_new_tokens=max(limits), k_prefill=self.k_prefill, eos_token_id=eos,
                       attention_mask=mask)
        texts = []
        for row, limit, (_, kw) in zip(out[:, width:].tolist(), limits, jobs):
            row = row[:limit]
            text = self.tokenizer.decode(row[: row.index(eos)] if eos in row else row, skip_special_tokens=True)
            until = kw.get("until") or []
            for stop in [until] if isinstance(until, str) else until:
                text = text.split(stop)[0]
            texts.append(text)
        return texts

    def generate_until(self, requests):
        jobs = [(self._context(context), kwargs) for context, kwargs in (r.args for r in requests)]
        return self._batched("generate_until", requests, jobs, self._generate, lambda j: len(j[0]))

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("perplexity tasks are not used in bitcycle's evaluations")
