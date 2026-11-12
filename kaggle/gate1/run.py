"""Kaggle entry point for Gate 1: the {full, ternary} x {baseline, Recycler} study.

Clones bitcycle, builds the TinyStories token arrays once, then trains every arm, each GPU
taking the next one. Results land in /kaggle/working/results/gate1.

A version can end early (session limit, network, a failed arm) and the next one starts with an
empty /kaggle/working. So results live in the private dataset harukadia/bitcycle-gate1-results:
`kaggle/sync.ps1 gate1` copies a version's output there and pushes the next version, which
copies them back, skips the finished arms and resumes the rest.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch  # preinstalled on Kaggle; used only to count GPUs

REF = os.environ.get("BITCYCLE_REF", "main")
SEEDS = [0]
SIZE = ["--d-model", "384", "--layers", "8", "--heads", "6", "--ff", "1024", "--seq-len", "256", "--batch", "64",
        "--train-bytes", "600000000", "--data", "/tmp/data"]  # outside /kaggle/working: not part of the output
STEPS = 5000
OUT = "/kaggle/working/results/gate1"
STOP = time.time() + 11 * 3600  # a session ends after 12 hours; training is checkpointed
# scripts/train.py Config fields per arm. Baselines also get 2x steps, matching Recycler's two-pass
# compute. Measured on a T4: the ternary two-pass arms run out of 16 GB at batch 64 without
# gradient checkpointing.
ARMS = [dict(precision=p, k=k, steps=s, checkpoint=int((p, k) == ("ternary", 2)))
        for p in ("ternary", "full") for k, s in ((1, STEPS), (2, STEPS), (1, 2 * STEPS))]
# Ablations of the ternary Recycler arm: recycle the pre-norm state; a ternary adapter.
ARMS += [dict(precision="ternary", k=2, steps=STEPS, checkpoint=1, **flag)
         for flag in ({"recycle": "pre_norm"}, {"adapter": "ternary"})]


def retry(cmd, **kw) -> None:
    """Setup steps use the network, which fails now and then: try again before giving up."""
    for wait in (30, 120, 600):
        if subprocess.run(cmd, **kw).returncode == 0:
            return
        time.sleep(wait)
    subprocess.run(cmd, check=True, **kw)


os.makedirs(OUT, exist_ok=True)
for f in Path("/kaggle/input").glob("**/*"):  # the results dataset
    if f.suffix in (".jsonl", ".pt"):
        shutil.copy(f, OUT)
retry(["git", "clone", "--depth", "1", "--branch", REF, "https://github.com/Harukadia/bitcycle"])
retry([sys.executable, "-m", "pip", "install", "-q", "-e", "bitcycle"])
sys.path.insert(0, "bitcycle")
from scripts.train import Config  # noqa: E402

train = [sys.executable, "scripts/train.py", *SIZE]
retry([*train, "--prepare-only"], cwd="bitcycle")


def finished(cfg: Config) -> bool:
    """The run's log has the evaluation of its last step."""
    log = Path(OUT, cfg.name() + ".jsonl")
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return any(r["type"] == "eval" and r["step"] == cfg.steps for r in rows)


runs = iter([{**arm, "seed": seed} for seed in SEEDS for arm in ARMS])  # shared: each GPU takes the next run


def queue(gpu: int) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    for run in runs:
        flags = [x for name, value in run.items() for x in (f"--{name}", str(value))]
        for _ in range(3):  # a run resumes from its checkpoint, so a failed one is tried again
            if finished(Config(**run)):
                break
            try:
                subprocess.run([*train, *flags, "--out", OUT], cwd="bitcycle", env=env,
                               timeout=max(1, STOP - time.time()))
            except subprocess.TimeoutExpired:
                return


threads = [threading.Thread(target=queue, args=(g,)) for g in range(max(1, torch.cuda.device_count()))]
for t in threads:
    t.start()
for t in threads:
    t.join()
