#!/usr/bin/env python3
"""Unit tests for the reviewed VoteRisk implementation."""

import math
import os
import sys

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from experiments.voterisk.vote_risk_impl import (
    apply_vote_risk_advantage_adjustment,
    vote_risk_single_mode,
)


def test_risk_monotonicity():
    r_low = vote_risk_single_mode(9, 1, 10, 16, 0.5)
    r_tie = vote_risk_single_mode(5, 5, 10, 16, 0.5)
    r_high = vote_risk_single_mode(1, 9, 10, 16, 0.5)
    print("risk monotonicity:", r_low, r_tie, r_high)
    assert 0 <= r_low < r_tie < r_high <= 1


def _run_group(n_correct: int, wrong_counts: list[int]):
    G = n_correct + sum(wrong_counts)
    acc = [1.0] * n_correct
    answers = ["correct"] * n_correct
    for mode, count in enumerate(wrong_counts):
        acc.extend([0.0] * count)
        answers.extend([f"wrong_{mode}"] * count)
    index = np.array(["uid"] * G)
    # Equal-magnitude negative advantages make the shaping direction transparent.
    scores = torch.tensor([1.0] * n_correct + [-1.0] * sum(wrong_counts), dtype=torch.float32)
    before = scores.clone()
    metrics = {}
    apply_vote_risk_advantage_adjustment(
        scores=scores,
        index=index,
        acc_list=acc,
        answer_list=answers,
        format_list=None,
        G=10,  # intentionally config value; implementation must use actual G
        K_target=16,
        alpha_smooth=0.5,
        lambda_vote=0.4,
        log_path=None,
        step=0,
        out_metrics=metrics,
        kappa=2.0,
    )
    return before, scores, metrics


def test_high_risk_is_penalised_more():
    # nc=3; wrong modes 4,2,1. The 4-count mode is the strongest vote threat.
    before, after, metrics = _run_group(3, [4, 2, 1])
    wrong_after = after[3:].tolist()
    mode0 = wrong_after[:4]
    mode1 = wrong_after[4:6]
    mode2 = wrong_after[6:]
    print("high-risk direction:", mode0[0], mode1[0], mode2[0], metrics)
    assert mode0[0] < -1.0, "highest-risk wrong mode must become MORE negative"
    assert mode2[0] > -1.0, "lowest-risk wrong mode should be relaxed"
    assert mode0[0] < mode1[0] < mode2[0]
    assert torch.allclose(after[:3], before[:3]), "correct trajectories must remain unchanged"


def test_absolute_risk_magnitude_is_preserved():
    # Same idea as the motivation: a dominant wrong mode far below the correct
    # mode should receive a smaller correction than a genuinely competitive
    # wrong mode. Max-normalising risk deviations would largely erase this.
    _, harmless, m_harmless = _run_group(6, [3, 1])
    _, dangerous, m_dangerous = _run_group(3, [4, 3])

    harmless_delta = abs(float(harmless[6].item()) + 1.0)  # first 3-count wrong
    dangerous_delta = abs(float(dangerous[3].item()) + 1.0)  # first 4-count wrong
    print("absolute magnitude:", harmless_delta, dangerous_delta)
    print("metrics harmless/dangerous:", m_harmless, m_dangerous)
    assert dangerous_delta > harmless_delta, (
        "a genuinely competitive wrong mode should get a larger correction "
        "than a harmless dominant-in-wrongs mode"
    )


def test_zero_sum_before_clip_behavior():
    # With equal |A| and small lambda no sample should hit kappa clip, so the
    # sample-weighted centred desired deltas should sum to ~0.
    before, after, metrics = _run_group(3, [4, 2, 1])
    wrong_delta_sum = float((after[3:] - before[3:]).sum().item())
    print("wrong delta sum:", wrong_delta_sum, metrics)
    assert abs(wrong_delta_sum) < 1e-6
    assert abs(metrics["mean_delta"]) < 1e-6


def test_skips_unidentifiable_groups():
    # No correct rollout => correct-vs-wrong competition cannot be estimated.
    G = 10
    scores = torch.full((G,), -1.0)
    before = scores.clone()
    metrics = {}
    apply_vote_risk_advantage_adjustment(
        scores=scores,
        index=np.array(["x"] * G),
        acc_list=[0.0] * G,
        answer_list=[f"w{i%3}" for i in range(G)],
        format_list=None,
        G=G,
        K_target=16,
        alpha_smooth=0.5,
        lambda_vote=0.4,
        log_path=None,
        step=0,
        out_metrics=metrics,
    )
    assert torch.allclose(scores, before)
    assert metrics["groups_processed"] == 0


def test_no_nan_random_batch():
    rng = np.random.RandomState(42)
    G = 10
    groups = 32
    scores = []
    acc = []
    ans = []
    ids = []
    for g in range(groups):
        # Force at least one correct and at least two wrong modes often enough.
        for i in range(G):
            c = i < 4 if g % 2 == 0 else rng.rand() < 0.5
            ids.append(f"g{g}")
            acc.append(1.0 if c else 0.0)
            ans.append("c" if c else f"w{rng.randint(0, 4)}")
            scores.append(0.7 if c else -0.7)
    t = torch.tensor(scores, dtype=torch.float32)
    metrics = {}
    apply_vote_risk_advantage_adjustment(
        scores=t,
        index=np.array(ids),
        acc_list=acc,
        answer_list=ans,
        format_list=None,
        G=G,
        K_target=16,
        alpha_smooth=0.5,
        lambda_vote=0.4,
        log_path=None,
        step=0,
        out_metrics=metrics,
    )
    assert torch.isfinite(t).all()
    assert 0.0 <= metrics["mean_risk"] <= 1.0
    assert 0.0 <= metrics["max_risk"] <= 1.0
    print("random metrics:", metrics)


if __name__ == "__main__":
    test_risk_monotonicity()
    test_high_risk_is_penalised_more()
    test_absolute_risk_magnitude_is_preserved()
    test_zero_sum_before_clip_behavior()
    test_skips_unidentifiable_groups()
    test_no_nan_random_batch()
    print("\nALL REVIEWED VoteRisk unit tests PASSED.")
