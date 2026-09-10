#!/usr/bin/env python3
"""Fast preflight for the vendored verl modules required by DAPO training."""

from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn, get_dataset_class
from verl.workers.utils.losses import ppo_loss
from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler, run_ppo

print("DatasetPadMode:", DatasetPadMode.NO_PADDING)
print("RLHFDataset:", RLHFDataset.__name__)
print("collate_fn:", collate_fn.__name__)
print("get_dataset_class:", get_dataset_class.__name__)
print("ppo_loss:", ppo_loss.__name__)
print("TaskRunner:", TaskRunner.__name__)
print("TRAINING IMPORT PREFLIGHT PASSED")
