#!/usr/bin/env python3
"""Fair evaluator for EDAS vs VoteRisk.

Key points:
- Generate one fixed pool of N=64 i.i.d. samples per problem.
- Pass@k uses the standard unbiased estimator from N samples and c successes:
    1 - C(N-c, k) / C(N, k).
- Maj@k estimates expected self-consistency accuracy by repeated subsampling
  k responses from the fixed N-sample pool.  It is NOT the result of only the
  first k samples.
- Mathematical answer modes use ``mathruler.grade_answer`` equivalence.
- Unparseable/no-answer responses are abstentions for voting, not one shared
  artificial wrong mode.  If all selected samples abstain, the vote fails.
- A tie for top vote is conservatively counted as failure and reported.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

try:
    from mathruler.grader import extract_boxed_content, grade_answer
except Exception:  # pragma: no cover
    extract_boxed_content = None
    grade_answer = None

TEST_DATA_ROOT = os.environ.get(
    "VR_TEST_DATA_ROOT", "/home/luorongchuan/workspace_135/datasets/test_data"
)
DEFAULT_DATASETS = ["MATH-500", "AMC23", "AIME24", "AIME25"]
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_RESPONSE_LENGTH = 8192
DEFAULT_N = 64
DEFAULT_SEED = 42
DEFAULT_MAJ_RESAMPLES = 2000
PASS_KS = [1, 4, 8, 16, 32]
MAJ_KS = [4, 8, 16, 32]


def extract_answer(response: str) -> str:
    if not response:
        return ""
    if extract_boxed_content is not None:
        try:
            ans = extract_boxed_content(response)
            if ans:
                return str(ans).strip()
        except Exception:
            pass
    import re

    m = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", response)
    return m.group(1).strip() if m else ""


def is_correct(answer: str, ground_truth: str) -> bool:
    if not answer or not ground_truth:
        return False
    if grade_answer is not None:
        try:
            return bool(grade_answer(answer, ground_truth))
        except Exception:
            return False
    return answer.strip().lower() == ground_truth.strip().lower()


def math_equiv(a: str, b: str) -> bool:
    a = str(a).strip()
    b = str(b).strip()
    if not a or not b:
        return False
    if grade_answer is not None:
        try:
            return bool(grade_answer(a, b))
        except Exception:
            pass
    return a.lower() == b.lower()


def build_answer_clusters(
    answers: list[str], correct_flags: list[bool]
) -> tuple[list[int | None], list[str], list[bool]]:
    """Cluster valid extracted answers by mathematical equivalence.

    Returns per-sample cluster id (None for no-answer), representatives, and a
    boolean saying whether each cluster is the verified-correct cluster.
    """
    reps: list[str] = []
    cluster_correct: list[bool] = []
    sample_cluster: list[int | None] = []

    for ans, corr in zip(answers, correct_flags):
        ans = str(ans).strip()
        if not ans:
            sample_cluster.append(None)
            continue
        found = None
        for idx, rep in enumerate(reps):
            if math_equiv(ans, rep):
                found = idx
                break
        if found is None:
            found = len(reps)
            reps.append(ans)
            cluster_correct.append(bool(corr))
        else:
            cluster_correct[found] = cluster_correct[found] or bool(corr)
        sample_cluster.append(found)
    return sample_cluster, reps, cluster_correct


def pass_at_k_estimate(n: int, c: int, k: int) -> float:
    """Standard unbiased pass@k estimator used for N sampled completions."""
    if k > n:
        return float("nan")
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    # Numerically stable product form of 1 - comb(n-c,k)/comb(n,k).
    fail = 1.0
    for i in range(k):
        fail *= (n - c - i) / (n - i)
    return float(1.0 - fail)


def compute_pass_at_k(correct_flags: list[bool], Ks: list[int]) -> dict[int, float]:
    n = len(correct_flags)
    c = int(sum(correct_flags))
    return {k: pass_at_k_estimate(n, c, k) for k in Ks}


def compute_maj_at_k(
    sample_cluster: list[int | None],
    cluster_correct: list[bool],
    Ks: list[int],
    seed: int,
    n_resamples: int,
) -> tuple[dict[int, float], dict[int, float]]:
    """Estimate expected plurality/self-consistency accuracy from a fixed pool."""
    n = len(sample_cluster)
    out: dict[int, float] = {}
    ties: dict[int, float] = {}

    for K in Ks:
        if K > n:
            out[K] = float("nan")
            ties[K] = float("nan")
            continue
        rng = np.random.default_rng(seed + 1009 * K)
        successes = 0
        tie_count = 0
        for _ in range(n_resamples):
            idxs = rng.choice(n, size=K, replace=False)
            counts: Counter = Counter(
                sample_cluster[int(i)]
                for i in idxs
                if sample_cluster[int(i)] is not None
            )
            if not counts:
                continue  # all abstentions -> failure
            max_count = max(counts.values())
            winners = [cid for cid, count in counts.items() if count == max_count]
            if len(winners) != 1:
                tie_count += 1
                continue  # conservative tie -> failure
            winner = winners[0]
            if cluster_correct[int(winner)]:
                successes += 1
        out[K] = successes / n_resamples
        ties[K] = tie_count / n_resamples
    return out, ties


@lru_cache(maxsize=None)
def _multinom_coeff(K: int, a: int, b: int) -> float:
    return float(math.comb(K, a) * math.comb(K - a, b))


def exact_vote_risk(n_correct: int, n_wrong: int, N: int, K: int = 16, alpha: float = 0.5) -> float:
    """P(wrong mode ties/beats correct) under the empirical trinomial model."""
    if n_wrong <= 0:
        return 0.0
    denom = N + 3.0 * alpha
    p_c = (n_correct + alpha) / denom
    p_j = (n_wrong + alpha) / denom
    p_o = (N - n_correct - n_wrong + alpha) / denom
    risk = 0.0
    for a in range(K + 1):
        for b in range(a, K - a + 1):
            risk += (
                _multinom_coeff(K, a, b)
                * (p_c**a)
                * (p_j**b)
                * (p_o ** (K - a - b))
            )
    return float(min(max(risk, 0.0), 1.0))


def compute_mechanism_metrics(
    answers: list[str],
    correct_flags: list[bool],
    sample_cluster: list[int | None],
    cluster_correct: list[bool],
    finish_reasons: list[str],
    output_token_counts: list[int],
) -> dict[str, float]:
    N = len(answers)
    n_correct = int(sum(correct_flags))
    p_correct = n_correct / N if N else 0.0

    cluster_counts = Counter(cid for cid in sample_cluster if cid is not None)
    wrong_counts = {
        cid: count
        for cid, count in cluster_counts.items()
        if not cluster_correct[int(cid)]
    }
    max_wrong = max(wrong_counts.values()) if wrong_counts else 0
    p_wrong_max = max_wrong / N if N else 0.0
    vote_margin = p_correct - p_wrong_max

    total_wrong_valid = sum(wrong_counts.values())
    if total_wrong_valid > 0:
        error_entropy = -sum(
            (count / total_wrong_valid) * math.log(count / total_wrong_valid)
            for count in wrong_counts.values()
        )
    else:
        error_entropy = 0.0

    no_answer_rate = sum(1 for a in answers if not str(a).strip()) / N if N else 0.0
    truncation_rate = (
        sum(1 for r in finish_reasons if str(r).lower() in {"length", "max_tokens"}) / N
        if N
        else 0.0
    )
    mean_output_tokens = float(np.mean(output_token_counts)) if output_token_counts else 0.0

    return {
        "p_correct": p_correct,
        "p_wrong_max": p_wrong_max,
        "vote_margin": vote_margin,
        "vote_risk_16": exact_vote_risk(n_correct, max_wrong, N, K=16, alpha=0.5),
        "error_entropy": float(error_entropy),
        "n_unique_wrong_answers": float(len(wrong_counts)),
        "no_answer_rate": float(no_answer_rate),
        "truncation_rate": float(truncation_rate),
        "mean_output_tokens": mean_output_tokens,
        "n_correct": float(n_correct),
        "n_max_wrong": float(max_wrong),
    }


def load_vllm(
    model_path: str,
    gpu_mem: float,
    max_response_length: int,
    tp_size: int,
    dtype: str,
) -> Any:
    from vllm import LLM

    return LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_response_length + 4096,
        dtype=dtype,
        trust_remote_code=True,
        enforce_eager=False,
        disable_log_stats=True,
    )


def generate_rollouts(
    llm: Any,
    prompts_chat: list[list[dict]],
    n: int,
    temperature: float,
    top_p: float,
    max_response_length: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    from vllm import SamplingParams

    sp = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_response_length,
        seed=seed,
    )
    outputs = llm.chat(prompts_chat, sampling_params=sp, use_tqdm=True)
    result: list[list[dict[str, Any]]] = []
    for req in outputs:
        row = []
        for c in req.outputs:
            token_ids = getattr(c, "token_ids", None)
            row.append(
                {
                    "text": c.text,
                    "finish_reason": str(getattr(c, "finish_reason", "") or ""),
                    "output_tokens": len(token_ids) if token_ids is not None else 0,
                }
            )
        result.append(row)
    return result


def evaluate_problem(
    problem_idx: int,
    ground_truth: str,
    completions: list[dict[str, Any]],
    base_seed: int,
    n_resamples: int,
) -> dict[str, Any]:
    responses = [str(x["text"]) for x in completions]
    answers = [extract_answer(x) for x in responses]
    correct_flags = [is_correct(a, ground_truth) for a in answers]
    sample_cluster, _reps, cluster_correct = build_answer_clusters(answers, correct_flags)

    pass_at_k = compute_pass_at_k(correct_flags, PASS_KS)
    maj_at_k, tie_rates = compute_maj_at_k(
        sample_cluster=sample_cluster,
        cluster_correct=cluster_correct,
        Ks=MAJ_KS,
        seed=base_seed + problem_idx * 10007,
        n_resamples=n_resamples,
    )
    metrics = compute_mechanism_metrics(
        answers=answers,
        correct_flags=correct_flags,
        sample_cluster=sample_cluster,
        cluster_correct=cluster_correct,
        finish_reasons=[x["finish_reason"] for x in completions],
        output_token_counts=[int(x["output_tokens"]) for x in completions],
    )

    rec: dict[str, Any] = {
        "problem_idx": problem_idx,
        "n_samples": len(completions),
        **metrics,
        "tie_rate": float(np.mean(list(tie_rates.values()))) if tie_rates else 0.0,
    }
    for k, v in pass_at_k.items():
        rec[f"pass_at_{k}"] = v
    for k, v in maj_at_k.items():
        rec[f"maj_at_{k}"] = v
    for k, v in tie_rates.items():
        rec[f"tie_at_{k}"] = v
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--method", required=True, choices=["edas", "voterisk", "base"])
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--maj_resamples", type=int, default=DEFAULT_MAJ_RESAMPLES)
    parser.add_argument("--max_response_length", type=int, default=DEFAULT_MAX_RESPONSE_LENGTH)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--gpu_mem", type=float, default=0.85)
    parser.add_argument(
        "--out_root",
        default="/home/luorongchuan/workspace_135/VoteRisk/experiments/voterisk",
    )
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    if args.n < max(PASS_KS + MAJ_KS):
        raise ValueError(f"--n must be >= {max(PASS_KS + MAJ_KS)} for the canonical metrics")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    out_root = Path(args.out_root)
    results_dir = out_root / "results"
    rollout_dir = out_root / "eval_rollouts" / args.method
    results_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[eval] method={args.method} N={args.n} Maj-resamples={args.maj_resamples} "
        f"T={args.temperature} top_p={args.top_p} seed={args.seed}"
    )
    t0 = time.time()
    llm = load_vllm(
        args.model_path,
        gpu_mem=args.gpu_mem,
        max_response_length=args.max_response_length,
        tp_size=args.tp_size,
        dtype=args.dtype,
    )
    print(f"[eval] vLLM loaded in {time.time() - t0:.1f}s")

    all_records: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []

    for ds_idx, ds in enumerate(datasets):
        path = Path(TEST_DATA_ROOT) / ds / "test.parquet"
        if not path.exists():
            print(f"[eval][WARN] missing {path}; skipping")
            continue
        df = pd.read_parquet(path)
        if args.limit > 0:
            df = df.iloc[: args.limit].reset_index(drop=True)

        prompts_chat = []
        for p in df["prompt"].tolist():
            prompts_chat.append(p.tolist() if hasattr(p, "tolist") else list(p))
        ground_truths = []
        for i in range(len(df)):
            obj = df.iloc[i]["reward_model"]
            if isinstance(obj, dict):
                ground_truths.append(str(obj.get("ground_truth", "")))
            else:
                ground_truths.append(str(obj))

        print(f"[eval] {ds}: {len(df)} problems × {args.n} samples")
        completions_per_problem = generate_rollouts(
            llm,
            prompts_chat,
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_response_length=args.max_response_length,
            seed=args.seed + ds_idx * 1000003,
        )

        ds_records: list[dict[str, Any]] = []
        raw_path = rollout_dir / f"{ds}.jsonl"
        with raw_path.open("w", encoding="utf-8") as raw_f:
            for i, (completions, gt) in enumerate(zip(completions_per_problem, ground_truths)):
                answers = [extract_answer(str(x["text"])) for x in completions]
                correct = [is_correct(a, gt) for a in answers]
                raw_f.write(
                    json.dumps(
                        {
                            "problem_idx": i,
                            "ground_truth": gt,
                            "completions": [
                                {
                                    **x,
                                    "extracted_answer": a,
                                    "is_correct": bool(c),
                                }
                                for x, a, c in zip(completions, answers, correct)
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                rec = evaluate_problem(
                    problem_idx=i,
                    ground_truth=gt,
                    completions=completions,
                    base_seed=args.seed + ds_idx * 1000003,
                    n_resamples=args.maj_resamples,
                )
                rec["dataset"] = ds
                rec["method"] = args.method
                ds_records.append(rec)
                all_records.append(rec)

        if not ds_records:
            continue
        row: dict[str, Any] = {
            "method": args.method,
            "dataset": ds,
            "n_problems": len(ds_records),
        }
        aggregate_cols = (
            [f"pass_at_{k}" for k in PASS_KS]
            + [f"maj_at_{k}" for k in MAJ_KS]
            + [f"tie_at_{k}" for k in MAJ_KS]
            + [
                "p_correct",
                "p_wrong_max",
                "vote_margin",
                "vote_risk_16",
                "error_entropy",
                "n_unique_wrong_answers",
                "no_answer_rate",
                "truncation_rate",
                "mean_output_tokens",
                "tie_rate",
            ]
        )
        for col in aggregate_cols:
            row[col] = float(np.mean([float(r[col]) for r in ds_records]))
        dataset_rows.append(row)

    # Always overwrite this method's outputs.  Appending on rerun silently
    # duplicated rows in the previous evaluator and corrupted comparisons.
    pd.DataFrame(dataset_rows).to_csv(
        results_dir / f"{args.method}_per_dataset.csv", index=False
    )
    with (results_dir / f"{args.method}_per_problem.jsonl").open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[eval] DONE method={args.method}, problems={len(all_records)}")
    if dataset_rows:
        print(pd.DataFrame(dataset_rows).to_string(index=False))


if __name__ == "__main__":
    main()
