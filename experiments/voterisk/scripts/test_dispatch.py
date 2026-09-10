#!/usr/bin/env python3
"""Dispatch test for EDAS vs reviewed VoteRisk."""

import os
import sys

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from experiments.voterisk.vote_risk_impl import _apply_vote_risk_advantage_adjustment
from verl.trainer.ppo import core_algos

# Mirror recipe/dapo/main_dapo.py: install reviewed implementation in-process.
core_algos._apply_vote_risk_advantage_adjustment = _apply_vote_risk_advantage_adjustment


def _make_batch():
    index = ["uid0"] * 10 + ["uid1"] * 10
    acc = [1.0] * 6 + [0.0] * 4 + [1.0] * 4 + [0.0] * 6
    answers = (
        ["c"] * 6
        + ["w1"] * 3 + ["w2"]
        + ["c"] * 4
        + ["w3"] * 3 + ["w4"] * 2 + ["w5"]
    )
    rewards = torch.tensor(acc, dtype=torch.float32).unsqueeze(-1)
    mask = torch.ones_like(rewards)
    return rewards, mask, np.array(index), acc, answers


def test_dispatch():
    rewards, mask, index, acc, answers = _make_batch()

    # Plain GRPO.
    plain, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=rewards.clone(), response_mask=mask, index=index,
        acc_list=acc, answer_list=answers, norm_adv_by_std_in_grpo=True,
    )

    # EDAS only.
    edas_metrics = {}
    edas, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=rewards.clone(), response_mask=mask, index=index,
        acc_list=acc, answer_list=answers, norm_adv_by_std_in_grpo=True,
        branch_adv_enabled=True, branch_adv_alpha=0.4,
        branch_adv_beta=0.3, branch_adv_kappa=2.0,
        branch_adv_out_metrics=edas_metrics,
    )
    assert edas_metrics.get("groups_total", 0) >= 1

    # VoteRisk only.
    vr_metrics = {}
    vr, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=rewards.clone(), response_mask=mask, index=index,
        acc_list=acc, answer_list=answers, norm_adv_by_std_in_grpo=True,
        vote_risk_enabled=True, vote_risk_G=10, vote_risk_K_target=16,
        vote_risk_alpha_smooth=0.5, vote_risk_lambda=0.4,
        branch_adv_out_metrics=vr_metrics,
    )
    assert vr_metrics.get("groups_processed", 0) >= 1
    assert "mean_risk" in vr_metrics

    # Both on defensively dispatches to VoteRisk, as core currently specifies.
    both_metrics = {}
    both, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=rewards.clone(), response_mask=mask, index=index,
        acc_list=acc, answer_list=answers, norm_adv_by_std_in_grpo=True,
        branch_adv_enabled=True, branch_adv_alpha=0.4,
        branch_adv_beta=0.3, branch_adv_kappa=2.0,
        vote_risk_enabled=True, vote_risk_G=10, vote_risk_K_target=16,
        vote_risk_alpha_smooth=0.5, vote_risk_lambda=0.4,
        branch_adv_out_metrics=both_metrics,
    )
    assert "groups_processed" in both_metrics
    assert "groups_b" not in both_metrics
    assert torch.allclose(vr, both)

    # All paths should remain finite.
    assert torch.isfinite(plain).all()
    assert torch.isfinite(edas).all()
    assert torch.isfinite(vr).all()
    print("plain finite; EDAS metrics:", edas_metrics)
    print("VoteRisk metrics:", vr_metrics)
    print("ALL DISPATCH TESTS PASSED")


if __name__ == "__main__":
    test_dispatch()
