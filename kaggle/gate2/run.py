"""Kaggle entry point for Gate 2: retrofit arms of BitNet b1.58 2B4T, then lm-eval.

Clones bitcycle and runs scripts/sft.py for every arm, one queue per GPU. Results land in
/kaggle/working/results/gate2: training logs, the trainable parameters, <run>_eval.json.
A session ends after 12 hours: push again. Each version starts with an empty /kaggle/working,
so kernel-metadata.json lists this kernel as its own source and the results of the last
version are copied back first; training resumes and finished evaluations are skipped.
Push with `kaggle kernels push -p kaggle/gate2`.
"""

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import torch  # preinstalled on Kaggle; used only to count GPUs

REF = os.environ.get("BITCYCLE_REF", "main")
# Measured on a T4 (float32): batch 8 runs out of 16 GB with LoRA, batch 4 fits every arm at
# 5.4 (lora) to 9.8 (adapter_lora) s/step. One version trains the arms listed here; push again with
# the rest, so that each version ends well inside the session limit.
ARMS = ("lora",)
STEPS = 2000
SEEDS = [0]
COMMON = ["--batch", "4", "--eval-limit", "1319", "--eval-batch", "16"]  # 1319 = all of gsm8k test
OUT = "/kaggle/working/results/gate2"

last = next(Path("/kaggle/input").glob("**/results/gate2"), None)  # the last version's output
if last:
    shutil.copytree(last, OUT, dirs_exist_ok=True)
if not Path("bitcycle").exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REF, "https://github.com/Harukadia/bitcycle"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "bitcycle[eval,sft]"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=True)  # Kaggle ships 0.10; peft refuses it
sys.path.insert(0, "bitcycle")
from scripts.sft import Config  # noqa: E402

# Arm (a), the untouched model, is evaluated once; the trained arms run per seed.
RUNS = [(Config(arm="base").name(), ["--arm", "base", "--eval-only"])] + [
    (Config(arm=arm, steps=STEPS, seed=seed).name(), ["--arm", arm, "--steps", str(STEPS), "--seed", str(seed)])
    for seed in SEEDS for arm in ARMS]


def queue(gpu: int, runs) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    for name, args in runs:
        if not Path(OUT, f"{name}_eval.json").exists():
            subprocess.run([sys.executable, "scripts/sft.py", *args, *COMMON, "--out", OUT], cwd="bitcycle", env=env, check=True)


gpus = max(1, torch.cuda.device_count())
threads = [threading.Thread(target=queue, args=(g, RUNS[g::gpus])) for g in range(gpus)]
for t in threads:
    t.start()
for t in threads:
    t.join()
