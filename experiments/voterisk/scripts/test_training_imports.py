#!/usr/bin/env python3
"""Fast preflight for the vendored verl modules and Hydra config required by DAPO training.

This script deliberately prioritizes the VoteRisk repository's vendored ``verl``
over any editable/system ``verl`` installation in the active conda environment.
It also composes the DAPO Hydra config without starting Ray, so missing config
groups are caught before a smoke/full training run.
"""

import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import verl
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn, get_dataset_class
from verl.workers.utils.losses import ppo_loss
from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler, run_ppo

verl_path = Path(verl.__file__).resolve()
expected_root = (REPO_ROOT / "verl").resolve()
if expected_root not in verl_path.parents and verl_path != expected_root:
    raise RuntimeError(
        f"Wrong verl imported: {verl_path}. Expected vendored verl under {expected_root}. "
        "Check PYTHONPATH/editable installs before training."
    )

legacy_data_cfg = REPO_ROOT / "verl" / "trainer" / "config" / "data" / "legacy_data.yaml"
if not legacy_data_cfg.is_file():
    raise FileNotFoundError(
        f"Missing Hydra config: {legacy_data_cfg}. "
        "The DAPO ppo_trainer defaults require data/legacy_data.yaml."
    )

# Compose the exact DAPO config in a subprocess.  `--cfg job` asks Hydra to
# compose/print the job config and exit, so this does not start Ray or training.
env = os.environ.copy()
existing_pythonpath = env.get("PYTHONPATH", "")
env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
compose = subprocess.run(
    [sys.executable, "-m", "recipe.dapo.main_dapo", "--cfg", "job"],
    cwd=REPO_ROOT,
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
if compose.returncode != 0:
    print(compose.stdout)
    raise RuntimeError(f"DAPO Hydra config composition failed with exit code {compose.returncode}")

print("repo root:", REPO_ROOT)
print("verl imported from:", verl_path)
print("legacy data config:", legacy_data_cfg)
print("DatasetPadMode:", DatasetPadMode.NO_PADDING)
print("RLHFDataset:", RLHFDataset.__name__)
print("collate_fn:", collate_fn.__name__)
print("get_dataset_class:", get_dataset_class.__name__)
print("ppo_loss:", ppo_loss.__name__)
print("TaskRunner:", TaskRunner.__name__)
print("DAPO Hydra config composition: OK")
print("TRAINING IMPORT PREFLIGHT PASSED")
