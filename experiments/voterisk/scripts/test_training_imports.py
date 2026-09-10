#!/usr/bin/env python3
"""Fast preflight for the vendored verl modules required by DAPO training.

This script deliberately prioritizes the VoteRisk repository's vendored ``verl``
over any editable/system ``verl`` installation in the active conda environment.
"""

from pathlib import Path
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

print("repo root:", REPO_ROOT)
print("verl imported from:", verl_path)
print("DatasetPadMode:", DatasetPadMode.NO_PADDING)
print("RLHFDataset:", RLHFDataset.__name__)
print("collate_fn:", collate_fn.__name__)
print("get_dataset_class:", get_dataset_class.__name__)
print("ppo_loss:", ppo_loss.__name__)
print("TaskRunner:", TaskRunner.__name__)
print("TRAINING IMPORT PREFLIGHT PASSED")
