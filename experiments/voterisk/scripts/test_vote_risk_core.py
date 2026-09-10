#!/usr/bin/env python3
"""Unit tests for _apply_vote_risk_advantage_adjustment.

Tests cover:
  - p_c >> p_j → risk very low
  - p_c ≈ p_j  → risk moderate
  - p_j > p_c  → risk very high
  - correct trajectories not modified
  - all-correct / all-wrong groups skipped
  - kappa-clip preserves sign
  - numerical stability (no NaN)
"""
import os
import sys
import math

import numpy as np
import torch

# Allow imports from the verl package directly.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from verl.trainer.ppo.core_algos import (
    _vote_risk_single_mode,
    _apply_vote_risk_advantage_adjustment,
)


# --- 1. Per-mode risk monotonicity ---
def test_risk_monotonicity():
    G = 10
    K = 16
    alpha = 0.5
    # (n_c, n_j) tuples: n_c + n_j <= G
    cases = [
        (9, 1, "p_c >> p_j"),
        (5, 5, "p_c ≈ p_j"),
        (1, 9, "p_j > p_c"),
    ]
    print("=== test_risk_monotonicity ===")
    risks = []
    for n_c, n_j, desc in cases:
        r = _vote_risk_single_mode(n_c, n_j, G, K, alpha)
        print(f"  n_c={n_c} n_j={n_j} ({desc}): R={r:.6f}")
        assert math.isfinite(r), f"Risk not finite: {r}"
        assert 0.0 <= r <= 1.0, f"Risk out of bounds: {r}"
        risks.append(r)
    # Risk should increase as n_j overtakes n_c.
    assert risks[0] < risks[1] < risks[2], (
        f"Risk not monotonic in n_j when n_c fixed: {risks}"
    )
    print("  PASS: R_16 monotonically increases as n_j grows")


# --- 2. Edge cases (n_j=0) ---
def test_zero_wrong_mode_risk():
    G = 10
    K = 16
    r = _vote_risk_single_mode(n_correct=10, n_j=0, G=G, K_target=K, alpha_smooth=0.5)
    print(f"\n=== test_zero_wrong_mode_risk: R when n_j=0 = {r:.6f} ===")
    assert math.isfinite(r)
    # p_j is tiny (alpha/(G+3alpha)=0.5/11.5), so risk should be tiny.
    assert r < 0.05, f"Risk when n_j=0 should be tiny: {r}"
    print("  PASS: tiny risk when n_j=0")


# --- 3. End-to-end shaping: correct trajectories not touched ---
def test_correct_untouched():
    """A group with both correct and wrong; correct trajectories' advantages
    must be unchanged after applying VoteRisk."""
    n_correct = 4
    n_wrong = 6  # split across 3 modes: 3, 2, 1
    answers_norm = (["c"] * n_correct) + (["w1"] * 3) + (["w2"] * 2) + (["w3"] * 1)
    acc_list = [1.0] * n_correct + [0.0] * n_wrong
    assert len(acc_list) == len(answers_norm)
    index = ["uid0"] * (n_correct + n_wrong)

    # Initial advantages (after GRPO standardization): all wrong are negative.
    scores = torch.tensor(
        [0.6, 0.4, 0.5, 0.3, -0.5, -0.7, -0.9, -0.4, -0.6, -0.3],
        dtype=torch.float32,
    )
    correct_orig = scores[:n_correct].clone()
    scores_orig = scores.clone()

    out_metrics: dict = {}
    _apply_vote_risk_advantage_adjustment(
        scores=scores,
        index=np.array(index),
        acc_list=acc_list,
        answer_list=answers_norm,
        format_list=None,
        G=10,
        K_target=16,
        alpha_smooth=0.5,
        lambda_vote=0.4,
        log_path=None,
        step=0,
        out_metrics=out_metrics,
    )

    # Correct trajectories must not change.
    assert torch.allclose(scores[:n_correct], correct_orig), (
        f"Correct trajectories were modified! before={correct_orig.tolist()} "
        f"after={scores[:n_correct].tolist()}"
    )
    # Wrong trajectories: kappa-clip must preserve sign(A_final) == sign(A_orig).
    for i in range(n_correct, len(scores)):
        new_adv = scores[i].item()
        orig_adv = scores_orig[i].item()
        # If both are non-zero, sign must match. Zero is a degenerate edge case
        # allowed (rare; happens when orig_adv = 0).
        if abs(orig_adv) > 1e-9 and abs(new_adv) > 1e-9:
            assert math.copysign(1.0, new_adv) == math.copysign(1.0, orig_adv), (
                f"Sign flipped at i={i}: orig={orig_adv} new={new_adv}"
            )
        # kappa-clip: |delta| <= |orig|/kappa.
        delta = new_adv - orig_adv
        assert abs(delta) <= abs(orig_adv) / 2.0 + 1e-9, (
            f"delta exceeds kappa-clip at i={i}: |delta|={abs(delta)} "
            f"limit={abs(orig_adv) / 2.0}"
        )

    # Metrics reported.
    assert "groups_processed" in out_metrics
    assert out_metrics["groups_processed"] >= 1
    print(f"\n=== test_correct_untouched: out_metrics={out_metrics} ===")
    print("  PASS: correct trajectories untouched, sign preserved")


# --- 4. Skipped groups ---
def test_skipped_groups():
    """all-correct and all-wrong groups should be skipped (no shaping)."""
    # Group A: all correct
    idx_a = ["a"] * 5
    acc_a = [1.0] * 5
    ans_a = ["42"] * 5
    # Group B: all wrong with single mode (K=1 → skipped per algorithm)
    idx_b = ["b"] * 5
    acc_b = [0.0] * 5
    ans_b = ["x"] * 5
    # Group C: all wrong, no correct answer (n_c=0 → skipped)
    idx_c = ["c"] * 5
    acc_c = [0.0] * 5
    ans_c = ["x1", "x2", "x3", "x4", "x5"]

    scores = torch.tensor(
        [-0.2, -0.4, 0.1, 0.2, 0.3, -0.1, -0.5, -0.7, -0.9, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
        dtype=torch.float32,
    )
    scores_orig = scores.clone()
    index = np.array(idx_a + idx_b + idx_c)
    acc = acc_a + acc_b + acc_c
    ans = ans_a + ans_b + ans_c

    out_metrics: dict = {}
    _apply_vote_risk_advantage_adjustment(
        scores=scores,
        index=index,
        acc_list=acc,
        answer_list=ans,
        format_list=None,
        G=10,
        K_target=16,
        alpha_smooth=0.5,
        lambda_vote=0.4,
        log_path=None,
        step=0,
        out_metrics=out_metrics,
    )
    # All three groups should be skipped → no delta.
    assert torch.allclose(scores, scores_orig), "Skipped groups were modified"
    assert out_metrics["groups_processed"] == 0, (
        f"Expected 0 processed, got {out_metrics['groups_processed']}"
    )
    print(f"\n=== test_skipped_groups: processed={out_metrics['groups_processed']} ===")
    print("  PASS: skipped groups not modified")


# --- 5. No NaN ---
def test_no_nan():
    """Random rollout with 50% correct; check no NaN / inf introduced."""
    torch.manual_seed(42)
    np.random.seed(42)
    G = 10
    n_groups = 16
    acc = []
    ans = []
    index = []
    scores_list = []
    for g in range(n_groups):
        idx = f"uid_{g}"
        for _ in range(G):
            index.append(idx)
            is_correct = np.random.rand() < 0.6
            acc.append(1.0 if is_correct else 0.0)
            ans.append("c" if is_correct else f"w{np.random.randint(0, 4)}")
            scores_list.append(np.random.randn())
    scores = torch.tensor(scores_list, dtype=torch.float32)
    out_metrics: dict = {}
    _apply_vote_risk_advantage_adjustment(
        scores=scores,
        index=np.array(index),
        acc_list=acc,
        answer_list=ans,
        format_list=None,
        G=G,
        K_target=16,
        alpha_smooth=0.5,
        lambda_vote=0.4,
        log_path=None,
        step=0,
        out_metrics=out_metrics,
    )
    assert torch.isfinite(scores).all(), "NaN / inf in scores"
    print(f"\n=== test_no_nan: processed={out_metrics['groups_processed']} skip={out_metrics['groups_skip']} ===")
    print(f"  mean_risk={out_metrics.get('mean_risk', 0):.4f} max_risk={out_metrics.get('max_risk', 0):.4f}")
    print(f"  mean_delta_abs={out_metrics.get('mean_delta_abs', 0):.4f}")
    print("  PASS: no NaN / inf, finite deltas")


if __name__ == "__main__":
    test_risk_monotonicity()
    test_zero_wrong_mode_risk()
    test_correct_untouched()
    test_skipped_groups()
    test_no_nan()
    print("\nALL VoteRisk unit tests PASSED.")