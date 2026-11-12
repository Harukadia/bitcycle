"""Kaggle entry point for Gate 1: the {full, ternary} x {baseline, Recycler} study.

Clones bitcycle, builds the TinyStories token arrays once, then trains every arm,
one queue per GPU. Logs land in /kaggle/working/results/gate1 and come back with
`kaggle kernels output harukadia/bitcycle-gate1`. Push with `kaggle kernels push -p kaggle/gate1`.
"""

import os
import subprocess
import sys
import threading

import torch  # preinstalled on Kaggle; used only to count GPUs

REF = os.environ.get("BITCYCLE_REF", "main")
SEEDS = [0]
SIZE = ["--d-model", "384", "--layers", "8", "--heads", "6", "--ff", "1024",
        "--seq-len", "256", "--batch", "64", "--train-bytes", "600000000", "--data", "/kaggle/working/data"]
STEPS = 5000
# Baselines also get 2x steps, matching Recycler's two-pass compute.
ARMS = [["--precision", p, "--k", str(k), "--steps", str(s)]
        for p in ("ternary", "full") for k, s in ((1, STEPS), (2, STEPS), (1, 2 * STEPS))]
# Ablations of the ternary Recycler arm: recycle the pre-norm state; a ternary adapter.
ARMS += [["--precision", "ternary", "--k", "2", "--steps", str(STEPS), *flag]
         for flag in (["--recycle", "pre_norm"], ["--adapter", "ternary"])]

subprocess.run(["git", "clone", "--depth", "1", "--branch", REF, "https://github.com/Harukadia/bitcycle"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "bitcycle"], check=True)
train = [sys.executable, "scripts/train.py", *SIZE]
subprocess.run([*train, "--prepare-only"], cwd="bitcycle", check=True)


def queue(gpu: int, arms) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    for seed in SEEDS:
        for arm in arms:
            cmd = [*train, *arm, "--seed", str(seed), "--out", "/kaggle/working/results/gate1"]
            subprocess.run(cmd, cwd="bitcycle", env=env, check=True)


gpus = max(1, torch.cuda.device_count())
threads = [threading.Thread(target=queue, args=(g, ARMS[g::gpus])) for g in range(gpus)]
for t in threads:
    t.start()
for t in threads:
    t.join()
