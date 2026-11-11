"""Kaggle entry point for Gate 2: retrofit arms of BitNet b1.58 2B4T, then lm-eval.

Clones bitcycle and runs scripts/sft.py for every arm, one queue per GPU. Results land in
/kaggle/working/results/gate2: training logs, the trainable parameters, <run>_eval.json.
A session ends after 12 hours: rerun the kernel; training resumes and finished evaluations
are skipped. Push with `kaggle kernels push -p kaggle/gate2` after adding a
kernel-metadata.json (id, enable_gpu, enable_internet, machine_shape).
"""

import os
import subprocess
import sys
import threading
from pathlib import Path

import torch  # preinstalled on Kaggle; used only to count GPUs

REF = os.environ.get("BITCYCLE_REF", "main")
STEPS = 2000
SEEDS = [0]
OUT = "/kaggle/working/results/gate2"

if not Path("bitcycle").exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REF, "https://github.com/Harukadia/bitcycle"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "bitcycle[eval,sft]"], check=True)
sys.path.insert(0, "bitcycle")
from scripts.sft import Config  # noqa: E402

# Arm (a), the untouched model, is evaluated once; the trained arms run per seed.
RUNS = [(Config(arm="base").name(), ["--arm", "base", "--eval-only"])] + [
    (Config(arm=arm, steps=STEPS, seed=seed).name(), ["--arm", arm, "--steps", str(STEPS), "--seed", str(seed)])
    for seed in SEEDS for arm in ("lora", "adapter", "adapter_lora")]


def queue(gpu: int, runs) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    for name, args in runs:
        if not Path(OUT, f"{name}_eval.json").exists():
            subprocess.run([sys.executable, "scripts/sft.py", *args, "--out", OUT], cwd="bitcycle", env=env, check=True)


gpus = max(1, torch.cuda.device_count())
threads = [threading.Thread(target=queue, args=(g, RUNS[g::gpus])) for g in range(gpus)]
for t in threads:
    t.start()
for t in threads:
    t.join()
