"""Gate 2: retrofit a frozen BitNet b1.58 2B4T with Recycler by SFT on GSM8K, then evaluate.

    python scripts/sft.py --arm adapter --steps 2000 --seed 0     # train, then evaluate
    python scripts/sft.py --arm base --eval-only                   # arm (a): the untouched model

Arms, trained on the same data for the same steps:
  lora          LoRA on every Linear with as many parameters as the adapter; no Recycler (k=1)
  adapter       the Recycler adapter only
  adapter_lora  both
The backbone is the bf16 master checkpoint made ternary by to_bitlinear, bit for bit
transformers' own BitNet quantizer. Examples use lm-eval's gsm8k format; prompt positions get
no loss and no recycled input, which is how generate(k_prefill=1) decodes. Only the arm's
parameters train. Everything runs in float32 (see load_base); on a GPU with gradient checkpointing.
Resumable like scripts/train.py; the log is JSONL in --out, the evaluation <run>_eval.json.
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from bitcycle.multipass import multi_pass_loss
from bitcycle.quant import to_bitlinear
from bitcycle.recycler import RecyclerLM

LINEARS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class Config:
    out: str = "results/gate2"
    arm: str = "adapter"  # base (evaluation only) | lora | adapter | adapter_lora
    model: str = "microsoft/bitnet-b1.58-2B-4T-bf16"
    k: int = 2  # training passes for arms with the adapter
    seed: int = 0
    steps: int = 2000
    batch: int = 8
    max_len: int = 512
    lr: float = 1e-4
    warmup: float = 0.03
    log_every: int = 10
    save_every: int = 100
    session_steps: int = 0  # stop after this many steps in this call; 0 = run to the end
    tasks: str = "gsm8k,arc_challenge,hellaswag"
    eval_limit: int = 0  # examples per task; 0 = all
    eval_batch: int = 8
    k_prefill: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def passes(self) -> int:
        return 1 if self.arm == "lora" else self.k

    def name(self) -> str:
        return "base" if self.arm == "base" else f"{self.arm}_k{self.passes()}_steps{self.steps}_seed{self.seed}"


def format_example(tokenizer, question: str, answer: str, max_len: int):
    """lm-eval gsm8k format -> (ids, labels, reset): no loss and no recycled input on the prompt."""
    prompt = tokenizer(f"Question: {question}\nAnswer:").input_ids
    response = tokenizer(" " + answer, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    ids = (prompt + response)[:max_len]
    labels = ([-100] * len(prompt) + response)[:max_len]
    reset = ([True] * len(prompt) + [False] * len(response))[:max_len]
    return ids, labels, reset


def collate(examples):
    """Right-pad: pads come after every real token, so causal attention never sees them."""
    width = max(len(ids) for ids, _, _ in examples)
    ids, labels, reset = ([x + [pad] * (width - len(x)) for x in column]
                          for column, pad in zip(zip(*examples), (0, -100, True)))
    return torch.tensor(ids), torch.tensor(labels), torch.tensor(reset)


def lora_rank(base: nn.Module) -> int:
    """The LoRA rank whose parameter count is closest to the adapter's (d * d + d)."""
    per_rank = sum(m.in_features + m.out_features for name, m in base.named_modules()
                   if name.rsplit(".", 1)[-1] in LINEARS and isinstance(m, nn.Linear))
    d = base.config.hidden_size
    return max(1, round((d * d + d) / per_rank))


def make_model(cfg: Config, base: nn.Module) -> RecyclerLM:
    """Freeze the backbone and add the arm's trainable parameters, at least float32."""
    torch.manual_seed(cfg.seed)
    base.requires_grad_(False)
    if "lora" in cfg.arm:
        from peft import LoraConfig, inject_adapter_in_model

        r = lora_rank(base)
        inject_adapter_in_model(LoraConfig(r=r, lora_alpha=2 * r, target_modules=LINEARS), base)
    model = RecyclerLM(base)
    model.adapter.requires_grad_(cfg.arm in ("adapter", "adapter_lora"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.promote_types(p.dtype, torch.float32))
    return model.to(cfg.device)


def load_base(name: str) -> nn.Module:
    """bf16 master weights made ternary by to_bitlinear, in float32.

    Not float16: on GSM8K text 2B4T's relu2(gate) * up reaches 1.7e10 and its residual stream
    1.6e5, past float16's 65504, so every logit is NaN. Kaggle's T4 and P100 have no bfloat16.
    transformers' own loader would quantize too, but needs accelerate and on CPU torch.compile.
    """
    config = AutoConfig.from_pretrained(name)
    config.quantization_config = None
    return to_bitlinear(AutoModelForCausalLM.from_pretrained(name, config=config, dtype=torch.float32))


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


def train(cfg: Config, model: RecyclerLM, data) -> RecyclerLM:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path, ckpt_path = out / f"{cfg.name()}.jsonl", out / f"{cfg.name()}.pt"
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    step = 0
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=cfg.device)
        model.load_state_dict(state["trainable"], strict=False)
        opt.load_state_dict(state["opt"])
        step = state["step"]
    else:
        log(log_path, {"type": "config", **asdict(cfg), "run": cfg.name(), "trainable": sum(p.numel() for p in params)})

    model.train()
    stop = min(cfg.steps, step + cfg.session_steps) if cfg.session_steps else cfg.steps
    started, start_step = time.time(), step
    while step < stop:
        g = torch.Generator().manual_seed(cfg.seed * 1_000_003 + step)
        batch = [data[i] for i in torch.randint(len(data), (cfg.batch,), generator=g).tolist()]
        ids, labels, reset = (t.to(cfg.device) for t in collate(batch))
        for group in opt.param_groups:
            group["lr"] = lr_at(cfg, step)
        loss, metrics = multi_pass_loss(model, ids, labels, k=cfg.passes(), reset_mask=reset)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1

        if cfg.log_every and step % cfg.log_every == 0:
            rate = (step - start_step) / (time.time() - started)
            log(log_path, {"type": "train", "step": step, "loss": loss.item(), **metrics,
                           "lr": lr_at(cfg, step - 1), "steps_per_s": round(rate, 3)})
        if step % cfg.save_every == 0 or step == stop:
            trainable = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
            torch.save({"trainable": trainable, "opt": opt.state_dict(), "step": step}, ckpt_path)
    model.eval()
    return model


def evaluate(cfg: Config, model: RecyclerLM, tokenizer) -> dict:
    import lm_eval

    from bitcycle.harness import RecyclerEval

    lm = RecyclerEval(model, tokenizer, batch_size=cfg.eval_batch, k_prefill=cfg.k_prefill)
    res = lm_eval.simple_evaluate(model=lm, tasks=cfg.tasks.split(","), limit=cfg.eval_limit or None)
    row = {"run": cfg.name(), "arm": cfg.arm, "k_prefill": cfg.k_prefill, "limit": cfg.eval_limit,
           "results": res["results"]}
    path = Path(cfg.out) / f"{cfg.name()}_eval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, default=float), encoding="utf-8")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(Config):
        parser.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    parser.add_argument("--eval-only", action="store_true", help="evaluate the saved run (or the base model)")
    args = parser.parse_args()
    cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    base = load_base(cfg.model)
    if cfg.arm == "base":
        model = RecyclerLM(base).to(cfg.device)  # zero adapter: exactly the base model
    else:
        if cfg.device.startswith("cuda"):
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = make_model(cfg, base)
        if args.eval_only:
            state = torch.load(Path(cfg.out) / f"{cfg.name()}.pt", map_location=cfg.device)
            model.load_state_dict(state["trainable"], strict=False)
        else:
            from datasets import load_dataset

            rows = load_dataset("openai/gsm8k", "main", split="train")
            model = train(cfg, model, [format_example(tokenizer, r["question"], r["answer"], cfg.max_len) for r in rows])
            done = torch.load(Path(cfg.out) / f"{cfg.name()}.pt", map_location="cpu")["step"] == cfg.steps
            if not done:
                return
    print(json.dumps(evaluate(cfg, model.eval(), tokenizer)["results"], indent=2, default=float))


if __name__ == "__main__":
    main()
