#!/usr/bin/env python3
"""eval_model.py — VLLM-based Maj@K / Pass@K evaluator for voterisk checkpoints.

Designed to share the EXACT same decoding parameters / prompt template across
EDAS and VoteRisk so the two are directly comparable.

Per dataset (default: MATH-500, AMC23, AIME24, AIME25):
  - Load N=64 rollouts per problem (configurable; same N for both methods).
  - For each problem:
      * Pass@K = 1 if any of K samples is correct.
      * Maj@K  = 1 if the most-voted (canonical-equivalent) answer is correct,
                 breaking ties as failure (conservative).
      * p_correct   = mean accuracy over the N samples.
      * p_wrong_max = mean (max wrong-mode count / N).
      * vote_margin = mean (p_correct - p_wrong_max).
      * vote_risk_16 = empirical P(wrong-mode votes >= correct-mode votes
                        in K=16 i.i.d. draws) — single number per problem,
                        averaged.
  - Per-problem JSONL + per-dataset CSV + aggregate metrics.

Outputs land in:
    experiments/voterisk/eval_rollouts/{method}/{dataset}.jsonl
    experiments/voterisk/results/{method}_per_dataset.csv
    experiments/voterisk/results/{method}_per_problem.jsonl

Usage:
    python eval_model.py \
        --model_path <ckpt_or_base> \
        --method edas|voterisk|base \
        --datasets MATH-500,AMC23,AIME24,AIME25 \
        --n 64 \
        --max_response_length 8192 \
        --temperature 1.0 \
        --top_p 1.0 \
        --gpus 2 \
        --out_root /home/luorongchuan/workspace_135/VoteRisk/experiments/voterisk
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# Reduce vLLM startup chatter unless the caller sets VERBOSE=1.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

# mathruler is the canonical answer-grading path (matches EDAS).
try:
    from mathruler.grader import extract_boxed_content, grade_answer
except Exception:  # pragma: no cover
    extract_boxed_content = None
    grade_answer = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEST_DATA_ROOT = os.environ.get("VR_TEST_DATA_ROOT", "/home/luorongchuan/workspace_135/datasets/test_data")

DEFAULT_DATASETS = ["MATH-500", "AMC23", "AIME24", "AIME25"]

# Decoding params (must be identical across methods).
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_RESPONSE_LENGTH = 8192
DEFAULT_N = 64
DEFAULT_SEED = 42

# Maj@K / Pass@K K values to report (canonical set).
MAJ_KS = [4, 8, 16, 32]
PASS_KS = [1, 4, 8, 16, 32]


# ---------------------------------------------------------------------------
# Answer extraction / grading (faithful to EDAS reward)
# ---------------------------------------------------------------------------

def extract_answer(response: str) -> str:
    if not response:
        return ""
    if extract_boxed_content is not None:
        try:
            ans = extract_boxed_content(response)
            if ans:
                return ans.strip()
        except Exception:
            pass
    # Fallback: regex over \\boxed{...} (handles nested braces loosely).
    import re
    m = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", response)
    return (m.group(1).strip() if m else "")


def is_correct(answer: str, ground_truth: str) -> bool:
    if not answer or not ground_truth:
        return False
    if grade_answer is not None:
        try:
            return bool(grade_answer(answer, ground_truth))
        except Exception:
            return False
    return answer.strip().lower() == ground_truth.strip().lower()


# ---------------------------------------------------------------------------
# Maj@K / Pass@K computation
# ---------------------------------------------------------------------------

def canonical_answer(ans: str) -> str:
    """Lowercase + strip so identical answers cluster. Math equivalence is
    delegated to mathruler.grade_answer at comparison time."""
    return str(ans).strip().lower()


def _canonical_dedup(answers: list[str]) -> list[str]:
    """Return one representative per canonical (grade-equivalent) cluster."""
    canonicals: list[str] = []
    for ans in answers:
        a = canonical_answer(ans)
        matched = False
        for c in canonicals:
            try:
                if grade_answer is not None and grade_answer(a, c):
                    canonicals[-1] = c  # no-op
                    matched = True
                    break
            except Exception:
                pass
        if not matched:
            canonicals.append(a)
    return canonicals


def compute_pass_at_k(correct_flags: list[bool], Ks: list[int]) -> dict[int, float]:
    """pass@K = 1 if any sample in the first K is correct; mean over problems."""
    out: dict[int, float] = {}
    for K in Ks:
        if K > len(correct_flags):
            out[K] = float("nan")
            continue
        out[K] = float(any(correct_flags[:K]))
    return out


def compute_maj_at_k(answers: list[str], correct_flags: list[bool], Ks: list[int]) -> dict[int, float]:
    """maj@K = 1 if the majority answer among first K samples is correct.

    Conservative tie rule: a tie at the top (correct tie with a wrong answer)
    is treated as failure, i.e. 0. We also report tie_rate.
    """
    out: dict[int, float] = {}
    out_tie: dict[int, float] = {}
    for K in Ks:
        if K > len(answers):
            out[K] = float("nan")
            out_tie[K] = float("nan")
            continue
        sub_answers = answers[:K]
        # Canonicalise + dedup via grade_answer.
        canonicals: list[str] = []
        canon_counts: list[int] = []
        for a in sub_answers:
            ca = canonical_answer(a)
            matched = False
            for idx, c in enumerate(canonicals):
                try:
                    if grade_answer is not None and grade_answer(ca, c):
                        canon_counts[idx] += 1
                        matched = True
                        break
                except Exception:
                    pass
            if not matched:
                canonicals.append(ca)
                canon_counts.append(1)
        # Find top count.
        max_count = max(canon_counts) if canon_counts else 0
        top_indices = [i for i, c in enumerate(canon_counts) if c == max_count]
        if len(top_indices) > 1:
            out[K] = 0.0  # tie -> failure
            out_tie[K] = 1.0
        else:
            top_canonical = canonicals[top_indices[0]]
            # Is this canonical the correct answer?
            # Find the right GT in correct_flags — we just check if the
            # top-canonical matches the (first) correct answer's canonical.
            correct_canonicals = [
                canonical_answer(a)
                for a, c in zip(sub_answers, correct_flags[:K])
                if c
            ]
            is_correct_top = False
            for cc in correct_canonicals:
                try:
                    if grade_answer is not None and grade_answer(top_canonical, cc):
                        is_correct_top = True
                        break
                except Exception:
                    if top_canonical == cc:
                        is_correct_top = True
                        break
            out[K] = 1.0 if is_correct_top else 0.0
            out_tie[K] = 0.0
    return out, out_tie


def compute_mechanism_metrics(
    answers: list[str], correct_flags: list[bool], N: int
) -> dict[str, float]:
    """p_correct, p_wrong_max, vote_margin, vote_risk_16, error_entropy,
    unique_wrong_answers, no_answer_rate, truncation_rate.

    Caller passes the full N=64 answers / correct_flags for one problem.
    """
    n_correct = sum(1 for c in correct_flags if c)
    p_correct = n_correct / N

    no_answer = sum(1 for a in answers if not a.strip()) / N

    # Canonical dedup of wrong answers.
    wrong_answers = [
        canonical_answer(a)
        for a, c in zip(answers, correct_flags)
        if not c and a.strip()
    ]
    wrong_counts = Counter(wrong_answers)
    K_modes = len(wrong_counts)

    # error entropy (over wrong pool, base e).
    if wrong_counts:
        total = sum(wrong_counts.values())
        H = -sum((c / total) * math.log(c / total) for c in wrong_counts.values())
    else:
        H = 0.0

    # p_wrong_max = max wrong-mode count / N (so a fair comparison vs p_correct).
    max_wrong = max(wrong_counts.values()) if wrong_counts else 0
    p_wrong_max = max_wrong / N

    vote_margin = p_correct - p_wrong_max

    # Empirical vote_risk_16: sample 16 i.i.d. draws from the empirical
    # distribution over (correct, wrong_mode_with_max_count, other) and count
    # fraction of times max_wrong >= n_correct.
    # We use the (p_c, p_j, p_o) triplet directly with the same exact
    # multinomial used in training (alpha=0.5 smoothing).
    K = 16
    alpha = 0.5
    p_c = (n_correct + alpha) / (N + 3 * alpha)
    p_j = (max_wrong + alpha) / (N + 3 * alpha)
    p_o = (N - n_correct - max_wrong + alpha) / (N + 3 * alpha)

    # Pre-compute multinomial coefficients for K=16.
    @lru_cache(maxsize=None)
    def _coeff(a: int, b: int) -> float:
        return math.comb(K, a) * math.comb(K - a, b)

    from functools import lru_cache

    risk = 0.0
    for a in range(K + 1):
        for b in range(a, K - a + 1):
            risk += _coeff(a, b) * (p_c**a) * (p_j**b) * (p_o ** (K - a - b))

    return {
        "p_correct": p_correct,
        "p_wrong_max": p_wrong_max,
        "vote_margin": vote_margin,
        "vote_risk_16": float(risk),
        "error_entropy": H,
        "n_unique_wrong_answers": K_modes,
        "no_answer_rate": no_answer,
        "n_correct": n_correct,
        "n_max_wrong": max_wrong,
    }


# ---------------------------------------------------------------------------
# VLLM generation
# ---------------------------------------------------------------------------

def load_vllm(model_path: str, gpu_mem: float = 0.85, max_response_length: int = 8192,
              tp_size: int = 1, dtype: str = "bfloat16") -> Any:
    from vllm import LLM
    return LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_response_length + 4096,  # leave headroom for prompt
        dtype=dtype,
        trust_remote_code=True,
        enforce_eager=False,
        disable_log_stats=True,
    )


def generate_rollouts(
    llm: Any, prompts_chat: list[list[dict]], n: int, temperature: float, top_p: float,
    max_response_length: int, seed: int, batch_size: int = 256,
) -> list[list[str]]:
    """Generate N rollouts per prompt. Returns [num_prompts][N] strings."""
    from vllm import SamplingParams

    sp = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_response_length,
        seed=seed,
    )

    # Flatten: each prompt -> N SamplingParams-equivalent entries via n=.
    outputs = llm.chat(prompts_chat, sampling_params=sp, use_tqdm=True)
    out: list[list[str]] = []
    for o in outputs:
        out.append([c.text for c in o.outputs])
    return out


# ---------------------------------------------------------------------------
# Per-problem evaluator
# ---------------------------------------------------------------------------

def evaluate_problem(
    problem_idx: int,
    answers: list[str],
    correct_flags: list[bool],
) -> dict[str, Any]:
    metrics = compute_mechanism_metrics(answers, correct_flags, N=len(answers))
    pass_at_k = compute_pass_at_k(correct_flags, PASS_KS)
    maj_at_k, tie_rates = compute_maj_at_k(answers, correct_flags, MAJ_KS)

    record = {
        "problem_idx": problem_idx,
        "n_samples": len(answers),
        "n_correct": int(sum(correct_flags)),
        "n_no_answer": int(sum(1 for a in answers if not a.strip())),
        **metrics,
        "tie_rate": sum(tie_rates.values()) / len(tie_rates) if tie_rates else 0.0,
    }
    for K, v in pass_at_k.items():
        record[f"pass_at_{K}"] = v
    for K, v in maj_at_k.items():
        record[f"maj_at_{K}"] = v
    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--method", required=True, choices=["edas", "voterisk", "base"])
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--max_response_length", type=int, default=DEFAULT_MAX_RESPONSE_LENGTH)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--gpu_mem", type=float, default=0.85)
    parser.add_argument("--out_root", default="/home/luorongchuan/workspace_135/VoteRisk/experiments/voterisk")
    parser.add_argument("--limit", type=int, default=-1, help="Limit problems per dataset (-1 = all)")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    rng = np.random.RandomState(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_root = Path(args.out_root)
    rollout_dir = out_root / "eval_rollouts" / args.method
    rollout_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] model_path={args.model_path} method={args.method}")
    print(f"[eval] datasets={datasets} n={args.n} T={args.temperature} top_p={args.top_p}")
    print(f"[eval] seed={args.seed} max_response_length={args.max_response_length}")

    # Lazy-import heavy modules here (after seed is set, before vLLM init).
    print("[eval] loading vLLM ...")
    t_load = time.time()
    llm = load_vllm(
        model_path=args.model_path,
        gpu_mem=args.gpu_mem,
        max_response_length=args.max_response_length,
        tp_size=args.tp_size,
        dtype=args.dtype,
    )
    print(f"[eval] vLLM loaded in {time.time() - t_load:.1f}s")

    all_records: list[dict] = []
    per_dataset_records: dict[str, list[dict]] = {}

    for ds in datasets:
        ds_path = os.path.join(TEST_DATA_ROOT, ds, "test.parquet")
        if not os.path.exists(ds_path):
            print(f"[eval] missing {ds_path}, skipping")
            continue

        df = pd.read_parquet(ds_path)
        if args.limit > 0:
            df = df.iloc[: args.limit].reset_index(drop=True)

        prompts = df["prompt"].tolist()
        # Convert numpy arrays to lists of dicts.
        prompts_chat = []
        for p in prompts:
            if hasattr(p, "tolist"):
                prompts_chat.append(p.tolist())
            else:
                prompts_chat.append(list(p))

        gt_col = "reward_model"
        ground_truths = [
            str(df.iloc[i][gt_col].get("ground_truth", ""))
            if isinstance(df.iloc[i][gt_col], dict)
            else str(df.iloc[i][gt_col])
            for i in range(len(df))
        ]

        print(f"[eval] [{ds}] {len(prompts_chat)} problems — generating {args.n} rollouts each")
        t0 = time.time()
        answers_per_problem = generate_rollouts(
            llm=llm,
            prompts_chat=prompts_chat,
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_response_length=args.max_response_length,
            seed=args.seed,
        )
        print(f"[eval] [{ds}] generation took {time.time() - t0:.1f}s")

        # Save raw rollouts first.
        raw_path = rollout_dir / f"{ds}.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for i, (ans_list, gt) in enumerate(zip(answers_per_problem, ground_truths)):
                f.write(json.dumps({
                    "problem_idx": i,
                    "ground_truth": gt,
                    "responses": ans_list,
                }, ensure_ascii=False) + "\n")

        # Score.
        ds_records = []
        for i, (ans_list, gt) in enumerate(zip(answers_per_problem, ground_truths)):
            extracted = [extract_answer(a) for a in ans_list]
            correct_flags = [is_correct(a, gt) for a in extracted]
            rec = evaluate_problem(i, extracted, correct_flags)
            rec["dataset"] = ds
            ds_records.append(rec)
            all_records.append(rec)
        per_dataset_records[ds] = ds_records

        # Per-problem JSONL.
        per_problem_path = out_root / "results" / f"{args.method}_per_problem.jsonl"
        per_problem_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if per_problem_path.exists() else "w"
        with open(per_problem_path, mode, encoding="utf-8") as f:
            for rec in ds_records:
                rec_to_write = {"dataset": ds, **rec}
                f.write(json.dumps(rec_to_write, ensure_ascii=False) + "\n")

    # Aggregate metrics per dataset.
    rows = []
    for ds, recs in per_dataset_records.items():
        if not recs:
            continue
        row = {"method": args.method, "dataset": ds, "n_problems": len(recs)}
        # Pass@K, Maj@K mean over problems.
        for K in PASS_KS:
            col = f"pass_at_{K}"
            row[col] = float(np.mean([r[col] for r in recs]))
        for K in MAJ_KS:
            col = f"maj_at_{K}"
            row[col] = float(np.mean([r[col] for r in recs]))
        # Mechanism metrics.
        for col in [
            "p_correct", "p_wrong_max", "vote_margin", "vote_risk_16",
            "error_entropy", "n_unique_wrong_answers", "no_answer_rate",
            "n_correct", "n_max_wrong", "tie_rate",
        ]:
            row[col] = float(np.mean([r[col] for r in recs]))
        rows.append(row)

    df_per_dataset = pd.DataFrame(rows)
    csv_path = out_root / "results" / f"{args.method}_per_dataset.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    df_per_dataset.to_csv(csv_path, mode="a" if csv_path.exists() else "w", header=write_header, index=False)
    print(f"[eval] per-dataset CSV -> {csv_path}")
    print(df_per_dataset.to_string(index=False))

    # Print summary.
    print(f"\n[eval] DONE — method={args.method} total_problems={len(all_records)}")


if __name__ == "__main__":
    # Late import so module-level seed does not affect vLLM import paths.
    from functools import lru_cache
    main()