#!/usr/bin/env python3
"""End-to-end dispatch test:
verifies compute_grpo_outcome_advantage correctly applies EDAS vs VoteRisk
when only branch_adv_enabled is set, only vote_risk_enabled is set, or both.

Also verifies the priority (VoteRisk wins when both flags are on).
"""
import os
import sys

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def _make_batch(seed=0):
    """Synthesise a small GRPO rollout batch with 2 groups (10 rollouts each).

    Group 0: 6 correct, 4 wrong (3 + 1 across 2 modes).
    Group 1: 4 correct, 6 wrong (3 + 2 + 1 across 3 modes).
    """
    rng = np.random.RandomState(seed)
    n = 20
    index = ["uid0"] * 10 + ["uid1"] * 10
    # Reward: 1 if correct, 0 if wrong.
    acc = [1.0] * 6 + [0.0] * 4 + [1.0] * 4 + [0.0] * 6
    answers = (
        ["c"] * 6
        + ["w1"] * 3
        + ["w2"] * 1
        + ["c"] * 4
        + ["w3"] * 3
        + ["w4"] * 2
        + ["w5"] * 1
    )
    # token-level rewards = per-sample scalar (no shaping).
    tok_rewards = torch.tensor(acc, dtype=torch.float32).unsqueeze(-1)
    response_mask = torch.ones_like(tok_rewards)
    return tok_rewards, response_mask, np.array(index), acc, answers


def _sum_abs_wrong(scores, acc_list):
    return sum(abs(s) for s, a in zip(scores, acc_list) if a < 1.0)


def test_dispatch():
    tok_rewards, mask, index, acc, answers = _make_batch()
    G, K = 10, 16

    # 1. Both flags off: pure GRPO (no shaping).
    scores_grpo = tok_rewards.clone()
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=scores_grpo,
        response_mask=mask,
        index=index,
        acc_list=acc,
        answer_list=answers,
        format_list=None,
        norm_adv_by_std_in_grpo=True,
    )
    grpo_neg = _sum_abs_wrong(advantages.squeeze(-1).tolist(), acc)

    # 2. EDAS on, VoteRisk off.
    scores_edas = tok_rewards.clone()
    out = {}
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=scores_edas,
        response_mask=mask,
        index=index,
        acc_list=acc,
        answer_list=answers,
        format_list=None,
        norm_adv_by_std_in_grpo=True,
        branch_adv_enabled=True,
        branch_adv_alpha=0.2,
        branch_adv_beta=0.2,
        branch_adv_kappa=2.0,
        branch_adv_out_metrics=out,
    )
    edas_neg = _sum_abs_wrong(advantages.squeeze(-1).tolist(), acc)
    print(f"GRPO |wrong| sum = {grpo_neg:.4f}")
    print(f"EDAS |wrong| sum = {edas_neg:.4f}  metrics={out}")
    assert out.get("groups_total", 0) >= 1

    # 3. VoteRisk on, EDAS off.
    scores_vr = tok_rewards.clone()
    out = {}
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=scores_vr,
        response_mask=mask,
        index=index,
        acc_list=acc,
        answer_list=answers,
        format_list=None,
        norm_adv_by_std_in_grpo=True,
        vote_risk_enabled=True,
        vote_risk_G=G,
        vote_risk_K_target=K,
        vote_risk_alpha_smooth=0.5,
        vote_risk_lambda=0.4,
        branch_adv_out_metrics=out,
    )
    vr_neg = _sum_abs_wrong(advantages.squeeze(-1).tolist(), acc)
    print(f"VR   |wrong| sum = {vr_neg:.4f}   metrics={out}")
    assert out.get("groups_processed", 0) >= 1

    # Both methods must be re-distributions around zero; sum |wrong| ≈
    # GRPO baseline (zero-mean preservation). Not strict equality (clip may
    # distort), but the order of magnitude should be close.
    assert 0.5 * grpo_neg <= edas_neg <= 2.0 * grpo_neg, (
        f"EDAS negative budget changed by >2x: {grpo_neg} -> {edas_neg}"
    )
    assert 0.5 * grpo_neg <= vr_neg <= 2.0 * grpo_neg, (
        f"VoteRisk negative budget changed by >2x: {grpo_neg} -> {vr_neg}"
    )

    # 4. Both flags on: VoteRisk takes precedence.
    scores_both = tok_rewards.clone()
    out = {}
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=scores_both,
        response_mask=mask,
        index=index,
        acc_list=acc,
        answer_list=answers,
        format_list=None,
        norm_adv_by_std_in_grpo=True,
        branch_adv_enabled=True,
        branch_adv_alpha=0.2,
        branch_adv_beta=0.2,
        branch_adv_kappa=2.0,
        vote_risk_enabled=True,
        vote_risk_G=G,
        vote_risk_K_target=K,
        vote_risk_alpha_smooth=0.5,
        vote_risk_lambda=0.4,
        branch_adv_out_metrics=out,
    )
    # VoteRisk metrics key (groups_processed, mean_risk, etc.) — not EDAS's
    # groups_b/groups_c — confirms VoteRisk path was taken.
    assert "groups_processed" in out
    assert "groups_b" not in out, "EDAS path was taken even though vote_risk_enabled=True"
    print(f"BOTH |wrong| sum = {_sum_abs_wrong(advantages.squeeze(-1).tolist(), acc):.4f}  metrics keys={list(out.keys())}")

    print("\nALL dispatch tests PASSED.")


if __name__ == "__main__":
    test_dispatch()