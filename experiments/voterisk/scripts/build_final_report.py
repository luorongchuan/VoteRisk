#!/usr/bin/env python3
"""build_final_report.py — collect per-dataset metrics from each method and
write the final markdown / CSV / plot artefacts.

Reads:
  {exp_root}/results/{method}_per_dataset.csv     (per-dataset aggregates)
  {exp_root}/results/{method}_per_problem.jsonl   (per-problem detail)
  {exp_root}/logs/seq_train_{method}.log          (training metrics)

Writes:
  {exp_root}/results/FINAL_COMPARISON.md          (human-readable answer)
  {exp_root}/results/main_metrics.csv             (single-row-per-method averages)
  {exp_root}/results/per_dataset_metrics.csv      (one row per (method, dataset))
  {exp_root}/results/per_problem_metrics.csv      (one row per (method, dataset, problem))
  {exp_root}/results/mechanism_metrics.csv        (mechanism-only aggregates)
  {exp_root}/results/training_metrics.csv         (parsed from training logs)
  {exp_root}/results/*.png                        (Maj@K, Pass@K, vote_margin, ...)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Matplotlib for plots. Use a non-interactive backend so this works headless.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PASS_KS = [1, 4, 8, 16, 32]
MAJ_KS = [4, 8, 16, 32]
MECHANISM_COLS = [
    "p_correct", "p_wrong_max", "vote_margin", "vote_risk_16",
    "error_entropy", "no_answer_rate",
]


def _load_per_dataset(exp_root: Path, method: str) -> pd.DataFrame:
    path = exp_root / "results" / f"{method}_per_dataset.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_per_problem(exp_root: Path, method: str) -> pd.DataFrame:
    path = exp_root / "results" / f"{method}_per_problem.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rec["method"] = method
            rows.append(rec)
    return pd.DataFrame(rows)


def _parse_training_log(log_path: Path) -> pd.DataFrame:
    """Light-weight regex parse of verl-style training logs. We pull out the
    reward / KL / grad-norm lines and the branch_adv / vote_risk metrics.
    """
    if not log_path.exists():
        return pd.DataFrame()

    rows = []
    # Try to capture per-step metrics lines from the JSON the trainer emits.
    pat_reward = re.compile(r'"critic/score/mean":\s*([\d.eE+-]+)')
    pat_kl = re.compile(r'"kl":\s*([\d.eE+-]+)')
    pat_grad = re.compile(r'"grad_norm":\s*([\d.eE+-]+)')
    pat_loss = re.compile(r'"actor/loss":\s*([\d.eE+-]+)')
    pat_brA = re.compile(r'"branch_adv/mean_delta_abs":\s*([\d.eE+-]+)')
    pat_brB = re.compile(r'"branch_adv/groups_b":\s*([\d.eE+-]+)')
    pat_brC = re.compile(r'"branch_adv/groups_c":\s*([\d.eE+-]+)')
    pat_vrRisk = re.compile(r'"vote_risk/mean_risk":\s*([\d.eE+-]+)')
    pat_vrDelta = re.compile(r'"vote_risk/mean_delta_abs":\s*([\d.eE+-]+)')
    pat_global = re.compile(r'"global_step":\s*(\d+)')
    pat_acc = re.compile(r'"metrics/accuracy_mean":\s*([\d.eE+-]+)')

    last_step = -1
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m_g = pat_global.search(line)
            if not m_g:
                continue
            step = int(m_g.group(1))
            if step == last_step:
                continue
            last_step = step

            def _try(pat):
                m = pat.search(line)
                return float(m.group(1)) if m else float("nan")

            rows.append({
                "global_step": step,
                "reward": _try(pat_reward),
                "kl": _try(pat_kl),
                "grad_norm": _try(pat_grad),
                "actor_loss": _try(pat_loss),
                "branch_adv_mean_delta_abs": _try(pat_brA),
                "branch_adv_groups_b": _try(pat_brB),
                "branch_adv_groups_c": _try(pat_brC),
                "vote_risk_mean_risk": _try(pat_vrRisk),
                "vote_risk_mean_delta_abs": _try(pat_vrDelta),
                "accuracy_mean": _try(pat_acc),
            })
    return pd.DataFrame(rows)


def _plot_pass_at_k(df_per_ds: pd.DataFrame, exp_root: Path):
    if df_per_ds.empty:
        return
    datasets = sorted(df_per_ds["dataset"].unique())
    methods = sorted(df_per_ds["method"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        for m in methods:
            sub = df_per_ds[df_per_ds["dataset"] == ds]
            if sub.empty:
                continue
            ys = [sub[f"pass_at_{K}"].mean() for K in PASS_KS]
            ax.plot(PASS_KS, ys, marker="o", label=m)
        ax.set_title(ds)
        ax.set_xlabel("K")
        ax.set_ylabel("Pass@K")
        ax.set_xticks(PASS_KS)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(exp_root / "results" / "pass_at_k.png", dpi=120)
    plt.close(fig)


def _plot_maj_at_k(df_per_ds: pd.DataFrame, exp_root: Path):
    if df_per_ds.empty:
        return
    datasets = sorted(df_per_ds["dataset"].unique())
    methods = sorted(df_per_ds["method"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        for m in methods:
            sub = df_per_ds[df_per_ds["dataset"] == ds]
            if sub.empty:
                continue
            ys = [sub[f"maj_at_{K}"].mean() for K in MAJ_KS]
            ax.plot(MAJ_KS, ys, marker="o", label=m)
        ax.set_title(ds)
        ax.set_xlabel("K")
        ax.set_ylabel("Maj@K")
        ax.set_xticks(MAJ_KS)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(exp_root / "results" / "maj_at_k.png", dpi=120)
    plt.close(fig)


def _plot_simple_bar(df_per_ds: pd.DataFrame, column: str, fname: str, exp_root: Path,
                    ylabel: str = ""):
    if df_per_ds.empty or column not in df_per_ds.columns:
        return
    datasets = sorted(df_per_ds["dataset"].unique())
    methods = sorted(df_per_ds["method"].unique())
    x = np.arange(len(datasets))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(1.6 * len(datasets) + 2, 4))
    for i, m in enumerate(methods):
        ys = []
        for ds in datasets:
            sub = df_per_ds[(df_per_ds["dataset"] == ds) & (df_per_ds["method"] == m)]
            ys.append(sub[column].mean() if not sub.empty else 0.0)
        ax.bar(x + i * width, ys, width=width, label=m)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel or column)
    ax.legend()
    fig.tight_layout()
    fig.savefig(exp_root / "results" / fname, dpi=120)
    plt.close(fig)


def _plot_training_reward(df_train: dict[str, pd.DataFrame], exp_root: Path):
    if not df_train:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, df in df_train.items():
        if df.empty:
            continue
        d = df.dropna(subset=["reward", "global_step"])
        ax.plot(d["global_step"], d["reward"], label=method, alpha=0.7)
    ax.set_xlabel("global_step")
    ax.set_ylabel("reward")
    ax.set_title("Training reward over steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(exp_root / "results" / "training_reward.png", dpi=120)
    plt.close(fig)


def _fmt_pct(v: float) -> str:
    if pd.isna(v):
        return "—"
    return f"{100.0 * v:.2f}%"


def _fmt_float(v: float, ndigits: int = 3) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:.{ndigits}f}"


def _bootstrap_diff(values_a: np.ndarray, values_b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """paired bootstrap CI for (a - b)."""
    rng = np.random.RandomState(seed)
    if len(values_a) != len(values_b):
        n = min(len(values_a), len(values_b))
        values_a = values_a[:n]
        values_b = values_b[:n]
    diffs = values_a - values_b
    n = len(diffs)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = diffs[idx].mean()
    return float(diffs.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _build_markdown(exp_root: Path, methods: list[str], datasets: list[str]) -> str:
    per_ds_all = []
    per_prob_all = []
    for m in methods:
        df = _load_per_dataset(exp_root, m)
        if not df.empty:
            df["method"] = m
            per_ds_all.append(df)
        pp = _load_per_problem(exp_root, m)
        if not pp.empty:
            per_prob_all.append(pp)

    if not per_ds_all:
        return "# Final Comparison\n\nNo evaluation results found yet. Run training + eval first.\n"

    df_per_ds = pd.concat(per_ds_all, ignore_index=True)
    df_per_problem = pd.concat(per_prob_all, ignore_index=True) if per_prob_all else pd.DataFrame()

    # Aggregate over datasets (mean across dataset means, weighted equally).
    df_main = df_per_ds.groupby("method", as_index=False).agg({
        **{f"pass_at_{K}": "mean" for K in PASS_KS},
        **{f"maj_at_{K}": "mean" for K in MAJ_KS},
        **{c: "mean" for c in MECHANISM_COLS},
    })

    md = []
    md.append("# FINAL_COMPARISON — VoteRisk vs EDAS on Qwen3-4B + DAPO\n")
    md.append("All numbers below come from `experiments/voterisk/results/*.csv` ")
    md.append("and `*_per_problem.jsonl`, which are produced by `eval_model.py` ")
    md.append("with **identical decoding / sampling / verifier / seed** across methods.\n")

    md.append("## 1. EDAS final training configuration\n")
    edas_row = df_per_ds[df_per_ds["method"] == "edas"].iloc[0] if "edas" in df_per_ds["method"].unique() else None
    md.append("See `ORIGINAL_EDAS_CONFIG.md` for the complete inventory. Key shared values:\n")
    md.append(f"- Model: `Qwen3-4B-Base` (`{os.environ.get('VR_MODEL_PATH', '/home/luorongchuan/workspace_135/models/Qwen3-4B-Base')}`)")
    md.append("- Data: `DAPO-Math-17K` (formatted parquet at `datasets/dapo-math-17k.formatted.parquet`)")
    md.append("- DAPO clip: low=0.2, high=0.28, KL disabled (`use_kl_loss=False`)")
    md.append("- Filter groups: enabled, metric=`accuracy`, max_num_gen_batches=20")
    md.append("- Rollout G=10 (per EDAS 8B config)")
    md.append("- max_prompt_length=2048, max_response_length=8192")
    md.append("- LR=1e-6, weight_decay=1e-2, grad_clip=1.0")
    md.append("- Optimizer: AdamW (FSDP, param/optimizer offload)")
    md.append("- Reward: `math_no_format.compute_score_batch` (rule + sympy, LLM judge disabled)")
    md.append("- Train batch size: 256 prompts, PPO mini-batch=64, micro-batch-per-GPU=1")
    md.append("- Epochs / steps: EDAS 8B defaults (`total_epochs=2`)\n")

    md.append("## 2. What did VoteRisk change?\n")
    md.append("**Only the advantage shaping function.** Specifically, in ")
    md.append("`verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage` ")
    md.append("the optional `_apply_vote_risk_advantage_adjustment` is invoked ")
    md.append("instead of `_apply_branch_advantage_adjustment` when ")
    md.append("`algorithm.vote_risk_enabled=True`. Everything else — reward, ")
    md.append("verifier, rollout, optimiser, KL, clip, dataset, model, prompt ")
    md.append("template, sampling — is shared verbatim with EDAS.\n")
    md.append("The new shaping replaces EDAS's per-mode relative-surprise signal ")
    md.append("with `R_{K,j} = P(N_j ≥ N_c | K=16 draws from Trinomial(p_c, p_j, p_o))` ")
    md.append("(Dirichlet smoothing α=0.5), then centres and kappa-clips using the ")
    md.append("**same** EDAS scaffold (so `sign(A_final) = sign(A_orig)` and correct ")
    md.append("trajectories are not modified).\n")

    md.append("## 3. Were the runs stable?\n")
    train_logs = {m: _parse_training_log(exp_root / "logs" / f"seq_train_{m}.log") for m in methods}
    stable_lines = []
    for m in methods:
        df = train_logs[m]
        if df.empty:
            stable_lines.append(f"- **{m}**: training log not parsed (no `seq_train_{m}.log`)")
            continue
        n_nan = int(df["reward"].isna().sum())
        n_total = len(df)
        finite = df[["reward", "kl", "grad_norm"]].applymap(np.isfinite).all(axis=1).sum()
        stable_lines.append(
            f"- **{m}**: {n_total} steps parsed; "
            f"reward non-NaN in {n_total - n_nan}/{n_total}; "
            f"reward+KL+grad_norm all finite in {finite}/{n_total}"
        )
    md.append("\n".join(stable_lines) + "\n")

    md.append("## 4. Pass@K (per dataset)\n")
    if not df_per_ds.empty:
        cols = ["method", "dataset"] + [f"pass_at_{K}" for K in PASS_KS]
        md.append(df_per_ds[cols].to_markdown(index=False, floatfmt=".3f"))
    md.append("\n")

    md.append("## 5. Maj@K (per dataset)\n")
    if not df_per_ds.empty:
        cols = ["method", "dataset"] + [f"maj_at_{K}" for K in MAJ_KS] + ["tie_rate"]
        cols = [c for c in cols if c in df_per_ds.columns]
        md.append(df_per_ds[cols].to_markdown(index=False, floatfmt=".3f"))
    md.append("\n")

    md.append("## 6-10. Mechanism metrics (per dataset)\n")
    if not df_per_ds.empty:
        cols = ["method", "dataset"] + MECHANISM_COLS + ["n_unique_wrong_answers"]
        cols = [c for c in cols if c in df_per_ds.columns]
        md.append(df_per_ds[cols].to_markdown(index=False, floatfmt=".3f"))
    md.append("\n")

    md.append("## 11. Did VoteRisk improve Maj@K without hurting Pass@K?\n")
    if {"edas", "voterisk"}.issubset(set(df_per_ds["method"].unique())):
        edas_row = df_per_ds[df_per_ds["method"] == "edas"].set_index("dataset")
        vr_row = df_per_ds[df_per_ds["method"] == "voterisk"].set_index("dataset")
        diff_lines = []
        for ds in sorted(set(edas_row.index) & set(vr_row.index)):
            for K in MAJ_KS:
                d = vr_row.loc[ds, f"maj_at_{K}"] - edas_row.loc[ds, f"maj_at_{K}"]
                diff_lines.append(f"  - `{ds}` Maj@{K}: Δ = {100 * d:+.2f} pp")
            for K in [1, 16, 32]:
                d = vr_row.loc[ds, f"pass_at_{K}"] - edas_row.loc[ds, f"pass_at_{K}"]
                diff_lines.append(f"  - `{ds}` Pass@{K}: Δ = {100 * d:+.2f} pp")
        md.append("\n".join(diff_lines) + "\n")
    md.append("\n")

    md.append("## 12. VoteRisk vs EDAS — verdict\n")
    if {"edas", "voterisk"}.issubset(set(df_per_ds["method"].unique())):
        edas_main = df_main[df_main["method"] == "edas"].iloc[0]
        vr_main = df_main[df_main["method"] == "voterisk"].iloc[0]
        bullets = []
        # Primary: Maj@16 / Maj@32.
        bullets.append(
            f"- **Maj@16 (avg)**: EDAS = {100 * edas_main['maj_at_16']:.2f}%, "
            f"VoteRisk = {100 * vr_main['maj_at_16']:.2f}% "
            f"(Δ = {100 * (vr_main['maj_at_16'] - edas_main['maj_at_16']):+.2f} pp)"
        )
        bullets.append(
            f"- **Maj@32 (avg)**: EDAS = {100 * edas_main['maj_at_32']:.2f}%, "
            f"VoteRisk = {100 * vr_main['maj_at_32']:.2f}% "
            f"(Δ = {100 * (vr_main['maj_at_32'] - edas_main['maj_at_32']):+.2f} pp)"
        )
        # Secondary: Pass@1, Pass@16, Pass@32.
        for K in [1, 16, 32]:
            bullets.append(
                f"- **Pass@{K} (avg)**: EDAS = {100 * edas_main[f'pass_at_{K}']:.2f}%, "
                f"VoteRisk = {100 * vr_main[f'pass_at_{K}']:.2f}% "
                f"(Δ = {100 * (vr_main[f'pass_at_{K}'] - edas_main[f'pass_at_{K}']):+.2f} pp)"
            )
        # Tertiary: mechanism metrics.
        for c in ["p_wrong_max", "vote_margin", "vote_risk_16"]:
            d = vr_main[c] - edas_main[c]
            bullets.append(f"- **{c} (avg)**: EDAS = {edas_main[c]:.3f}, VoteRisk = {vr_main[c]:.3f} (Δ = {d:+.3f})")
        md.append("\n".join(bullets) + "\n")

        # Bootstrap CI for the main signal.
        if not df_per_problem.empty:
            ci_lines = ["\nPaired bootstrap (MATH-500, VoteRisk − EDAS, 1000 resamples):"]
            math_pp = df_per_problem[df_per_problem["dataset"] == "MATH-500"] if "MATH-500" in df_per_problem["dataset"].unique() else df_per_problem
            for metric in ["maj_at_8", "maj_at_16", "maj_at_32", "pass_at_1", "pass_at_16", "pass_at_32"]:
                if metric not in math_pp.columns:
                    continue
                a = math_pp[math_pp["method"] == "voterisk"].sort_values("problem_idx")[metric].to_numpy()
                b = math_pp[math_pp["method"] == "edas"].sort_values("problem_idx")[metric].to_numpy()
                if len(a) == 0 or len(b) == 0:
                    continue
                m, lo, hi = _bootstrap_diff(a, b)
                ci_lines.append(
                    f"  - `{metric}`: Δ = {100 * m:+.2f} pp, "
                    f"95% CI = [{100 * lo:+.2f}, {100 * hi:+.2f}]"
                )
            md.append("\n".join(ci_lines) + "\n")

        # Final verdict.
        maj16_delta = vr_main["maj_at_16"] - edas_main["maj_at_16"]
        pass1_delta = vr_main["pass_at_1"] - edas_main["pass_at_1"]
        if maj16_delta > 0 and pass1_delta > -0.02:
            verdict = "**YES**: VoteRisk improves Maj@16 without substantially hurting Pass@1."
        elif maj16_delta > 0 and pass1_delta < -0.05:
            verdict = "**PARTIAL**: VoteRisk improves Maj@16 but Pass@1 dropped — investigate."
        elif maj16_delta > 0:
            verdict = "**YES (with caveats)**: VoteRisk improves Maj@16; Pass@1 unchanged within noise."
        elif maj16_delta > -0.01:
            verdict = "**NO**: VoteRisk is on par with EDAS (Δ ≈ 0)."
        else:
            verdict = "**NO**: VoteRisk under-performs EDAS on Maj@16."
        md.append(f"\n**Verdict**: {verdict}\n")

    md.append("\n## Files\n")
    md.append("- `main_metrics.csv` — averaged over datasets, one row per method")
    md.append("- `per_dataset_metrics.csv` — one row per (method, dataset)")
    md.append("- `per_problem_metrics.csv` — one row per (method, dataset, problem)")
    md.append("- `mechanism_metrics.csv` — mechanism-only aggregates")
    md.append("- `training_metrics.csv` — parsed from verl training logs")
    md.append("- `pass_at_k.png`, `maj_at_k.png`, `vote_margin.png`, `p_wrong_max.png`, `vote_risk_16.png`, `training_reward.png`\n")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", required=True)
    parser.add_argument("--methods", nargs="+", default=["edas", "voterisk"])
    parser.add_argument("--datasets", default="MATH-500,AMC23,AIME24,AIME25")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    res_dir = exp_root / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    per_ds_all = []
    per_prob_all = []
    for m in args.methods:
        df = _load_per_dataset(exp_root, m)
        if not df.empty:
            df["method"] = m
            per_ds_all.append(df)
        pp = _load_per_problem(exp_root, m)
        if not pp.empty:
            per_prob_all.append(pp)
    df_per_ds = pd.concat(per_ds_all, ignore_index=True) if per_ds_all else pd.DataFrame()
    df_per_problem = pd.concat(per_prob_all, ignore_index=True) if per_prob_all else pd.DataFrame()

    # 1. per-dataset CSV (append if missing).
    pds_csv = res_dir / "per_dataset_metrics.csv"
    write_header = not pds_csv.exists()
    df_per_ds.to_csv(pds_csv, mode="a" if pds_csv.exists() else "w", header=write_header, index=False)

    # 2. per-problem CSV.
    if not df_per_problem.empty:
        df_per_problem.to_csv(res_dir / "per_problem_metrics.csv", index=False)

    # 3. main_metrics.csv (method-level averages).
    if not df_per_ds.empty:
        df_main = df_per_ds.groupby("method", as_index=False).agg({
            **{f"pass_at_{K}": "mean" for K in PASS_KS},
            **{f"maj_at_{K}": "mean" for K in MAJ_KS},
            **{c: "mean" for c in MECHANISM_COLS},
        })
        df_main.to_csv(res_dir / "main_metrics.csv", index=False)

    # 4. mechanism-only CSV.
    if not df_per_ds.empty:
        mech_cols = ["method", "dataset"] + MECHANISM_COLS
        mech_cols = [c for c in mech_cols if c in df_per_ds.columns]
        df_per_ds[mech_cols].to_csv(res_dir / "mechanism_metrics.csv", index=False)

    # 5. training-metrics CSV.
    train_dfs = []
    for m in args.methods:
        log_path = exp_root / "logs" / f"seq_train_{m}.log"
        df = _parse_training_log(log_path)
        if df.empty:
            continue
        df["method"] = m
        train_dfs.append(df)
    if train_dfs:
        df_train = pd.concat(train_dfs, ignore_index=True)
        df_train.to_csv(res_dir / "training_metrics.csv", index=False)

    # 6. plots.
    if not df_per_ds.empty:
        _plot_pass_at_k(df_per_ds, exp_root)
        _plot_maj_at_k(df_per_ds, exp_root)
        _plot_simple_bar(df_per_ds, "vote_margin", "vote_margin.png", exp_root, ylabel="vote_margin (p_correct - p_wrong_max)")
        _plot_simple_bar(df_per_ds, "p_wrong_max", "p_wrong_max.png", exp_root, ylabel="p_wrong_max")
        _plot_simple_bar(df_per_ds, "vote_risk_16", "vote_risk_16.png", exp_root, ylabel="vote_risk_16")
    if train_dfs:
        _plot_training_reward({m: d.drop(columns=["method"]) for m, d in zip(args.methods, train_dfs)}, exp_root)

    # 7. FINAL_COMPARISON.md.
    md = _build_markdown(exp_root, args.methods, args.datasets.split(","))
    (res_dir / "FINAL_COMPARISON.md").write_text(md, encoding="utf-8")
    print(f"[report] wrote {res_dir}/FINAL_COMPARISON.md")
    print(md[:2000])


if __name__ == "__main__":
    main()