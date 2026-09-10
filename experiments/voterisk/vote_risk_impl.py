"""Reviewed VoteRisk advantage shaping used by the DAPO experiments.

This module intentionally lives outside the vendored verl core so the EDAS
baseline remains untouched. ``recipe.dapo.main_dapo`` installs this function
at runtime when ``algorithm.vote_risk_enabled=True``.

Design
------
For a prompt group, let n_c be the number of verified-correct rollouts and
n_j the count of a canonical wrong-answer mode j.  For a target test-time
budget K, we compute exactly

    R_{K,j} = P(N_j >= N_c)

under a three-category multinomial model (correct, wrong mode j, all other
answers) with symmetric Dirichlet smoothing.

The shaping signal is the *absolute* vote-risk deviation from the
sample-weighted mean wrong risk,

    d_j = R_{K,j} - sum_l n_l R_{K,l} / N_w.

We deliberately do NOT divide by max|d_j|.  Such a normalisation would erase
the absolute threat magnitude and make a harmless dominant wrong mode look as
important as a genuinely competitive wrong mode.  Desired deltas are

    Delta_i = -lambda * mean(|A_wrong|) * d_j,

so high-risk wrong modes become more negative while low-risk modes become less
negative.  Because the mean is weighted by per-mode sample counts, desired
wrong-sample deltas sum to zero before clipping.  EDAS's kappa clipping is then
applied to preserve the sign of each original advantage.
"""

from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.ppo.core_algos import _grade_answer

_MULTINOM_CACHE: dict[int, np.ndarray] = {}


def _get_multinomial_table(K: int) -> np.ndarray:
    if K <= 0:
        raise ValueError(f"K_target must be positive, got {K}")
    if K in _MULTINOM_CACHE:
        return _MULTINOM_CACHE[K]
    tab = np.zeros((K + 1, K + 1), dtype=np.float64)
    for a in range(K + 1):
        for b in range(K + 1 - a):
            tab[a, b] = math.comb(K, a) * math.comb(K - a, b)
    _MULTINOM_CACHE[K] = tab
    return tab


def vote_risk_single_mode(
    n_correct: int,
    n_j: int,
    G: int,
    K_target: int,
    alpha_smooth: float = 0.5,
) -> float:
    """Exact P(N_j >= N_c) for K_target draws from a trinomial model."""
    if G <= 0:
        raise ValueError(f"G must be positive, got {G}")
    if not (0 <= n_correct <= G and 0 <= n_j <= G and n_correct + n_j <= G):
        raise ValueError(
            f"invalid counts: n_correct={n_correct}, n_j={n_j}, G={G}"
        )
    if alpha_smooth < 0:
        raise ValueError(f"alpha_smooth must be >= 0, got {alpha_smooth}")

    denom = G + 3.0 * alpha_smooth
    if denom <= 0:
        raise ValueError("invalid smoothing denominator")
    p_c = (n_correct + alpha_smooth) / denom
    p_j = (n_j + alpha_smooth) / denom
    p_o = (G - n_correct - n_j + alpha_smooth) / denom

    tab = _get_multinomial_table(K_target)
    risk = 0.0
    for a in range(K_target + 1):
        # a = correct votes, b = votes for wrong mode j.  Tie is a threat.
        for b in range(a, K_target - a + 1):
            coeff = tab[a, b]
            if coeff == 0.0:
                continue
            risk += coeff * (p_c**a) * (p_j**b) * (p_o ** (K_target - a - b))
    return float(min(max(risk, 0.0), 1.0))


def _canonicalise_wrong_answers(answers: list[str]) -> tuple[dict[int, str], Counter]:
    """Cluster answers using the same math-equivalence check as EDAS."""
    representatives: list[str] = []
    sample_to_rep: dict[int, str] = {}
    counts: Counter = Counter()

    for pos, raw in enumerate(answers):
        ans = str(raw).strip().lower()
        rep = None
        for candidate in representatives:
            try:
                if _grade_answer(ans, candidate):
                    rep = candidate
                    break
            except Exception:
                pass
        if rep is None:
            rep = ans
            representatives.append(rep)
        sample_to_rep[pos] = rep
        counts[rep] += 1
    return sample_to_rep, counts


def apply_vote_risk_advantage_adjustment(
    scores: torch.Tensor,
    index: np.ndarray,
    acc_list: list,
    answer_list: list,
    format_list: Optional[list],
    G: int,
    K_target: int,
    alpha_smooth: float,
    lambda_vote: float,
    log_path: Optional[str],
    step: int,
    out_metrics: dict,
    kappa: float = 2.0,
) -> None:
    """Apply corrected VoteRisk shaping in-place.

    ``G`` is retained in the signature for compatibility with the existing
    trainer configuration, but each prompt uses its actual group size.  This
    avoids silently using the wrong probability model if dynamic batching ever
    produces a group whose size differs from ``rollout.n``.
    """
    if kappa <= 1.0:
        raise ValueError(f"kappa must be > 1, got {kappa}")
    if lambda_vote < 0:
        raise ValueError(f"lambda_vote must be >= 0, got {lambda_vote}")

    id2indices: dict = defaultdict(list)
    for i in range(scores.shape[0]):
        id2indices[index[i]].append(i)

    n_groups_processed = 0
    n_groups_skip = 0
    n_group_size_mismatch = 0
    n_adjusted = 0
    n_risk_samples = 0
    sum_wrong = 0
    sum_risk_weighted = 0.0
    sum_group_max_risk = 0.0
    max_risk_global = 0.0
    sum_delta_abs = 0.0
    sum_delta = 0.0
    max_delta_abs = 0.0
    log_lines: list[str] = []
    eps = 1e-12

    for uid, indices in id2indices.items():
        G_group = len(indices)
        if G_group != G:
            n_group_size_mismatch += 1

        wrong_indices = [
            i
            for i in indices
            if float(acc_list[i]) == 0.0
            and (format_list is None or float(format_list[i]) >= 0.5)
        ]
        N_w = len(wrong_indices)
        n_c = sum(1 for i in indices if float(acc_list[i]) >= 1.0 - 1e-9)

        # No verified correct mode -> vote competition against the correct mode
        # cannot be estimated from this group.  All-correct and <=1 wrong are
        # also uninformative for within-wrong redistribution.
        if n_c == 0 or n_c == G_group or N_w <= 1:
            n_groups_skip += 1
            continue

        wrong_answers = [str(answer_list[i]) for i in wrong_indices]
        sample_to_rep, canonical_counts = _canonicalise_wrong_answers(wrong_answers)
        if len(canonical_counts) <= 1:
            n_groups_skip += 1
            continue

        per_mode_risk: dict[str, float] = {}
        for rep, n_j in canonical_counts.items():
            per_mode_risk[rep] = vote_risk_single_mode(
                n_correct=n_c,
                n_j=n_j,
                G=G_group,
                K_target=K_target,
                alpha_smooth=alpha_smooth,
            )

        # IMPORTANT: sample-weighted centring gives sum_i d_i = 0 over wrong
        # samples.  A mode-level unweighted mean would not preserve that budget.
        mean_risk = sum(
            canonical_counts[rep] * risk for rep, risk in per_mode_risk.items()
        ) / N_w

        deviations = {rep: risk - mean_risk for rep, risk in per_mode_risk.items()}
        if max(abs(v) for v in deviations.values()) < eps:
            n_groups_skip += 1
            continue

        orig_advs = [float(scores[i].item()) for i in wrong_indices]
        base = sum(abs(a) for a in orig_advs) / N_w

        n_groups_processed += 1
        sum_wrong += N_w
        group_max_risk = max(per_mode_risk.values())
        sum_group_max_risk += group_max_risk
        max_risk_global = max(max_risk_global, group_max_risk)

        if log_path:
            log_lines.append(
                f"\n[group {uid}] G_actual={G_group} G_cfg={G} n_c={n_c} "
                f"N_w={N_w} modes={len(canonical_counts)} base={base:.6f} "
                f"weighted_mean_risk={mean_risk:.6f}"
            )
            log_lines.append(f"  counts={dict(canonical_counts)}")
            log_lines.append(
                "  risks=" + str({k: round(v, 6) for k, v in per_mode_risk.items()})
            )

        # Desired delta is NEGATIVE for above-average risk, making a negative
        # wrong advantage more negative.  Below-average risk is relaxed.
        for pos, i in enumerate(wrong_indices):
            rep = sample_to_rep[pos]
            risk_i = per_mode_risk[rep]
            dev_i = deviations[rep]
            orig_adv = orig_advs[pos]
            delta_desired = -lambda_vote * base * dev_i

            max_delta = abs(orig_adv) / kappa
            if abs(delta_desired) <= max_delta:
                delta_final = delta_desired
            else:
                delta_final = math.copysign(max_delta, delta_desired)

            new_adv = orig_adv + delta_final
            scores[i] = scores[i].new_tensor(new_adv)

            n_adjusted += 1
            n_risk_samples += 1
            sum_risk_weighted += risk_i
            sum_delta += delta_final
            sum_delta_abs += abs(delta_final)
            max_delta_abs = max(max_delta_abs, abs(delta_final))

            if log_path:
                log_lines.append(
                    f"    idx={i} rep={rep[:24]!r} risk={risk_i:.6f} "
                    f"dev={dev_i:+.6f} orig={orig_adv:+.6f} "
                    f"delta={delta_final:+.6f} final={new_adv:+.6f}"
                )

    if log_path and n_groups_processed:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        header = [
            "=" * 72,
            f"=== step {step} (VoteRisk reviewed) ===",
            f"groups_processed={n_groups_processed} skip={n_groups_skip} "
            f"G_mismatch={n_group_size_mismatch}",
            f"K_target={K_target} alpha={alpha_smooth} lambda={lambda_vote} kappa={kappa}",
            "=" * 72,
        ]
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(header + log_lines) + "\n")

    out_metrics["groups_processed"] = n_groups_processed
    out_metrics["groups_skip"] = n_groups_skip
    out_metrics["groups_total"] = n_groups_processed + n_groups_skip
    out_metrics["group_size_mismatch"] = n_group_size_mismatch
    out_metrics["mean_wrong_per_group"] = (
        sum_wrong / n_groups_processed if n_groups_processed else 0.0
    )
    out_metrics["mean_risk"] = (
        sum_risk_weighted / n_risk_samples if n_risk_samples else 0.0
    )
    out_metrics["mean_group_max_risk"] = (
        sum_group_max_risk / n_groups_processed if n_groups_processed else 0.0
    )
    out_metrics["max_risk"] = max_risk_global
    out_metrics["mean_delta_abs"] = (
        sum_delta_abs / n_adjusted if n_adjusted else 0.0
    )
    out_metrics["max_delta_abs"] = max_delta_abs
    out_metrics["mean_delta"] = sum_delta / n_adjusted if n_adjusted else 0.0


# Alias matching the legacy core name, useful for a one-line monkeypatch.
_apply_vote_risk_advantage_adjustment = apply_vote_risk_advantage_adjustment
