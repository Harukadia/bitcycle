"""Kaggle entry point for Gate 2: retrofit arms of BitNet b1.58 2B4T, then lm-eval.

Clones bitcycle and runs scripts/sft.py for every arm, each GPU taking the next one. Results land
in /kaggle/working/results/gate2: training logs, the trainable parameters, <run>_eval.json.
A session ends after 12 hours, so runs are stopped after 11: push again. Each version starts with an empty /kaggle/working,
so kernel-metadata.json lists this kernel as its own source and the results of the last
version are copied back first; training resumes and finished evaluations are skipped.
Push with `kaggle kernels push -p kaggle/gate2`.
"""

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch  # preinstalled on Kaggle; used only to count GPUs

REF = os.environ.get("BITCYCLE_REF", "main")
# Measured on a T4 (float32): batch 8 runs out of 16 GB with LoRA, batch 4 fits every arm at
# 5.4 (lora) to 9.8 (adapter_lora) s/step. Evaluation starts at batch 16 and halves when it does
# not fit (gsm8k's 5-shot prompts do not).
ARMS = ("lora", "adapter", "adapter_lora")
STEPS = 2000
SEEDS = [0]
COMMON = ["--batch", "4", "--eval-limit", "1319", "--eval-batch", "16"]  # 1319 = all of gsm8k test
OUT = "/kaggle/working/results/gate2"
STOP = time.time() + 11 * 3600  # training is checkpointed; an evaluation cut here starts over

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


runs = iter(RUNS)  # shared: each GPU takes the next run


def queue(gpu: int) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    for name, args in runs:  # a failed run does not stop the ones after it
        if not Path(OUT, f"{name}_eval.json").exists():
            try:
                subprocess.run([sys.executable, "scripts/sft.py", *args, *COMMON, "--out", OUT], cwd="bitcycle",
                               env=env, timeout=max(1, STOP - time.time()))
            except subprocess.TimeoutExpired:
                return


threads = [threading.Thread(target=queue, args=(g,)) for g in range(max(1, torch.cuda.device_count()))]
for t in threads:
    t.start()
for t in threads:
    t.join()
