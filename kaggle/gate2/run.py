"""Kaggle entry point for Gate 2: retrofit arms of BitNet b1.58 2B4T, then lm-eval.

Clones bitcycle and runs scripts/sft.py for every arm, each GPU taking the next one. Results land
in /kaggle/working/results/gate2: training logs, the trainable parameters, the evaluation's
response cache, <run>_eval.json.

A version can end early (session limit, network, a failed run) and the next one starts with an
empty /kaggle/working. So results live in the private dataset harukadia/bitcycle-gate2-results:
`kaggle/sync.ps1 gate2` copies a version's output there and pushes the next version, which
copies them back; training and evaluation resume, finished runs are skipped.
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
STOP = time.time() + 11 * 3600  # a session ends after 12 hours; training and evaluation resume


def retry(cmd, **kw) -> None:
    """Setup steps use the network, which fails now and then: try again before giving up."""
    for wait in (30, 120, 600):
        if subprocess.run(cmd, **kw).returncode == 0:
            return
        time.sleep(wait)
    subprocess.run(cmd, check=True, **kw)


os.makedirs(OUT, exist_ok=True)
for f in Path("/kaggle/input").glob("**/*"):  # the results dataset
    if f.suffix in (".jsonl", ".pt", ".json", ".db"):
        shutil.copy(f, OUT)
retry(["git", "clone", "--depth", "1", "--branch", REF, "https://github.com/Harukadia/bitcycle"])
retry([sys.executable, "-m", "pip", "install", "-q", "-e", "bitcycle[eval,sft]"])
retry([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])  # Kaggle ships 0.10; peft refuses it
sys.path.insert(0, "bitcycle")
from scripts.sft import Config  # noqa: E402

# Arm (a), the untouched model, is evaluated once; the trained arms run per seed.
RUNS = [(Config(arm="base").name(), ["--arm", "base", "--eval-only"])] + [
    (Config(arm=arm, steps=STEPS, seed=seed).name(), ["--arm", arm, "--steps", str(STEPS), "--seed", str(seed)])
    for seed in SEEDS for arm in ARMS]
runs = iter(RUNS)  # shared: each GPU takes the next run


def queue(gpu: int) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    for name, args in runs:
        for _ in range(3):  # a run resumes where it stopped, so a failed one is tried again
            if Path(OUT, f"{name}_eval.json").exists():
                break
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
