"""Gate 1: tiny BitNet-architecture LMs on TinyStories, full precision or ternary, with or without Recycler.

    python scripts/train.py --precision ternary --k 2 --seed 0 --steps 3000

k=1 is the baseline (plain next-token loss); k=2 is Recycler's two-pass training.
Batches are drawn from (seed, step) alone, so model + optimizer + step resume exactly.
--session-steps stops early with a checkpoint (Kaggle sessions end); rerun the same
command to continue. The log is JSONL in --out, one file per run.
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, BitNetConfig

from bitcycle.generate import recurrent_nll
from bitcycle.multipass import multi_pass_loss
from bitcycle.quant import to_bitlinear
from bitcycle.recycler import RecyclerLM

EOS = "<|endoftext|>"
REPO = "datasets/roneneldan/TinyStories"


@dataclass
class Config:
    out: str = "results/gate1"
    run: str = ""
    precision: str = "ternary"  # full | ternary
    k: int = 2  # 1 = baseline
    recycle: str = "post_norm"  # post_norm | pre_norm (ablation)
    adapter: str = "full"  # full | ternary (ablation)
    seed: int = 0
    steps: int = 3000
    batch: int = 32
    seq_len: int = 128
    vocab: int = 4096
    d_model: int = 128
    layers: int = 4
    heads: int = 4
    ff: int = 344
    lr: float = 1e-3
    warmup: float = 0.02
    weight_decay: float = 0.1
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 200
    session_steps: int = 0  # stop after this many steps in this call; 0 = run to the end
    checkpoint: int = 0  # 1 = gradient checkpointing: same gradients, less memory, slower
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def name(self) -> str:
        ablations = ("_prenorm" if self.recycle == "pre_norm" else "") + ("_ternaryadapter" if self.adapter == "ternary" else "")
        return self.run or f"{self.precision}_k{self.k}_steps{self.steps}_seed{self.seed}{ablations}"


def pack(docs, eos: int, seq_len: int):
    """Concatenate documents with eos and cut into rows; reset marks each document's first token."""
    stream = torch.cat([torch.tensor([*d, eos], dtype=torch.int32) for d in docs])
    stream = stream[: len(stream) // seq_len * seq_len]
    reset = torch.zeros_like(stream, dtype=torch.bool)
    reset[1:] = stream[:-1] == eos
    return stream.view(-1, seq_len), reset.view(-1, seq_len)


def eval_batch(docs, length: int):
    """The first `length` tokens of every document at least that long."""
    return torch.tensor([d[:length] for d in docs if len(d) >= length])


def make_model(cfg: Config) -> RecyclerLM:
    torch.manual_seed(cfg.seed)
    config = BitNetConfig(
        vocab_size=cfg.vocab,
        hidden_size=cfg.d_model,
        intermediate_size=cfg.ff,
        num_hidden_layers=cfg.layers,
        num_attention_heads=cfg.heads,
        num_key_value_heads=cfg.heads,
        max_position_embeddings=cfg.seq_len,
        tie_word_embeddings=True,
        bos_token_id=None,
        eos_token_id=None,
    )
    base = AutoModelForCausalLM.from_config(config)
    if cfg.precision == "ternary":
        to_bitlinear(base)
    if cfg.checkpoint:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = RecyclerLM(base, recycle=cfg.recycle)
    if cfg.adapter == "ternary":
        to_bitlinear(model.adapter)
    return model.to(cfg.device)


@torch.no_grad()
def evaluate(model: RecyclerLM, val, batch: int = 64) -> float:
    """Mean per-token NLL under the exact decode recurrence."""
    device = next(model.parameters()).device
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(val), batch):
        nll = recurrent_nll(model, val[i : i + batch].to(device))
        total, count = total + nll.sum().item(), count + nll.numel()
    model.train()
    return total / count


def lr_at(cfg: Config, step: int) -> float:
    """Linear warmup, then cosine decay to 10% of the peak."""
    warm = max(1, int(cfg.warmup * cfg.steps))
    if step < warm:
        return cfg.lr * (step + 1) / warm
    progress = (step - warm) / max(1, cfg.steps - warm)
    return cfg.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def log(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def train(cfg: Config, inputs, reset, val) -> RecyclerLM:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path, ckpt_path = out / f"{cfg.name()}.jsonl", out / f"{cfg.name()}.pt"

    model = make_model(cfg)
    params = list(model.parameters())
    opt = torch.optim.AdamW(
        [{"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
         {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95),
    )
    step = 0
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=cfg.device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
    else:
        log(log_path, {"type": "config", **asdict(cfg), "run": cfg.name(), "params": sum(p.numel() for p in params)})

    stop = min(cfg.steps, step + cfg.session_steps) if cfg.session_steps else cfg.steps
    started, start_step = time.time(), step
    while step < stop:
        g = torch.Generator().manual_seed(cfg.seed * 1_000_003 + step)
        idx = torch.randint(len(inputs), (cfg.batch,), generator=g)
        ids, mask = inputs[idx].long().to(cfg.device), reset[idx].to(cfg.device)
        for group in opt.param_groups:
            group["lr"] = lr_at(cfg, step)
        loss, metrics = multi_pass_loss(model, ids, ids, k=cfg.k, reset_mask=mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1

        if cfg.log_every and step % cfg.log_every == 0:
            rate = (step - start_step) * cfg.batch * cfg.seq_len / (time.time() - started)
            log(log_path, {"type": "train", "step": step, "loss": loss.item(), **metrics,
                           "lr": lr_at(cfg, step - 1), "tokens_per_s": round(rate)})
        if cfg.eval_every and step % cfg.eval_every == 0 and step < cfg.steps:
            log(log_path, {"type": "eval", "step": step, "val_nll": evaluate(model, val)})
        if step % cfg.save_every == 0 or step == stop:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, ckpt_path)

    if start_step < step == cfg.steps:
        log(log_path, {"type": "eval", "step": step, "val_nll": evaluate(model, val)})
    return model


def load_docs(name: str, max_bytes: int | None, cache: Path) -> list[str]:
    """Stories from a TinyStories text file; reads only the first max_bytes (None = all)."""
    path = cache / f"{name}.{max_bytes or 'all'}.txt"
    if not path.exists():
        from huggingface_hub import HfFileSystem

        with HfFileSystem().open(f"{REPO}/{name}", "rb") as f:
            path.write_bytes(f.read(max_bytes or -1))
    docs = [s.strip() for s in path.read_text(encoding="utf-8", errors="ignore").split(EOS)]
    return [d for d in docs[:-1] if d]  # the last story may be cut mid-way


def tokenizer_for(docs: list[str], vocab: int, cache: Path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    path = cache / f"bpe{vocab}.json"
    if path.exists():
        return Tokenizer.from_file(str(path))
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[EOS],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(docs, trainer)
    tok.save(str(path))
    return tok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(Config):
        parser.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    parser.add_argument("--data", default="data", help="cache for text, tokenizer and token arrays")
    parser.add_argument("--train-bytes", type=int, default=200_000_000, help="bytes of the train file to read")
    parser.add_argument("--eval-len", type=int, default=128)
    parser.add_argument("--eval-docs", type=int, default=512)
    parser.add_argument("--prepare-only", action="store_true", help="build the token arrays, then exit")
    args = parser.parse_args()
    cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})

    cache = Path(args.data)
    cache.mkdir(parents=True, exist_ok=True)
    arrays = cache / f"tokens_{args.train_bytes}_v{cfg.vocab}_l{cfg.seq_len}_e{args.eval_len}.pt"
    if not arrays.exists():
        train_docs = load_docs("TinyStoriesV2-GPT4-train.txt", args.train_bytes, cache)
        val_docs = load_docs("TinyStoriesV2-GPT4-valid.txt", None, cache)
        tok = tokenizer_for(train_docs, cfg.vocab, cache)
        encode = lambda docs: [e.ids for e in tok.encode_batch(docs)]  # noqa: E731
        inputs, reset = pack(encode(train_docs), tok.token_to_id(EOS), cfg.seq_len)
        torch.save({"inputs": inputs, "reset": reset, "val": eval_batch(encode(val_docs), args.eval_len)}, arrays)
    if args.prepare_only:
        return
    data = torch.load(arrays)
    train(cfg, data["inputs"], data["reset"], data["val"][: args.eval_docs])


if __name__ == "__main__":
    main()
