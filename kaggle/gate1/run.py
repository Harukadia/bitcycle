"""Kaggle entry point for Gate 1: the {full, ternary} x {baseline, Recycler} study.

Clones bitcycle, builds the TinyStories token arrays once, then trains every arm,
each GPU taking the next one. Logs land in /kaggle/working/results/gate1 and come back with
`kaggle kernels output harukadia/bitcycle-gate1`. Push with `kaggle kernels push -p kaggle/gate1`.
kernel-metadata.json lists this kernel as its own source: the last version's results are copied
back first, so finished arms are skipped and unfinished ones resume.
"""

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import torch  # preinstalled on Kaggle; used only to count GPUs

REF = os.environ.get("BITCYCLE_REF", "main")
SEEDS = [0]
SIZE = ["--d-model", "384", "--layers", "8", "--heads", "6", "--ff", "1024",
        "--seq-len", "256", "--batch", "64", "--train-bytes", "600000000", "--data", "/kaggle/working/data"]
STEPS = 5000
# Baselines also get 2x steps, matching Recycler's two-pass compute.
# Measured on a T4: the ternary two-pass arms run out of 16 GB at batch 64 without gradient checkpointing.
FIT = {("ternary", 2): ["--checkpoint", "1"]}
ARMS = [["--precision", p, "--k", str(k), "--steps", str(s), *FIT.get((p, k), [])]
        for p in ("ternary", "full") for k, s in ((1, STEPS), (2, STEPS), (1, 2 * STEPS))]
# Ablations of the ternary Recycler arm: recycle the pre-norm state; a ternary adapter.
ARMS += [["--precision", "ternary", "--k", "2", "--steps", str(STEPS), *FIT["ternary", 2], *flag]
         for flag in (["--recycle", "pre_norm"], ["--adapter", "ternary"])]
OUT = "/kaggle/working/results/gate1"

last = next(Path("/kaggle/input").glob("**/results/gate1"), None)  # the last version's output
if last:
    shutil.copytree(last, OUT, dirs_exist_ok=True)

subprocess.run(["git", "clone", "--depth", "1", "--branch", REF, "https://github.com/Harukadia/bitcycle"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "bitcycle"], check=True)
train = [sys.executable, "scripts/train.py", *SIZE]
subprocess.run([*train, "--prepare-only"], cwd="bitcycle", check=True)


runs = iter([(seed, arm) for seed in SEEDS for arm in ARMS])  # shared: each GPU takes the next run


def queue(gpu: int) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    for seed, arm in runs:  # a failed run does not stop the ones after it
        subprocess.run([*train, *arm, "--seed", str(seed), "--out", OUT], cwd="bitcycle", env=env)


threads = [threading.Thread(target=queue, args=(g,)) for g in range(max(1, torch.cuda.device_count()))]
for t in threads:
    t.start()
for t in threads:
    t.join()
