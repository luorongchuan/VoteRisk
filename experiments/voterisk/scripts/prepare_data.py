#!/usr/bin/env python3
"""Prepare DAPO-Math-17k parquet for voterisk training.

The original parquet (datasets/dapo-math-17k.parquet) already stores the
prompt as a chat-list. EDAS / branch_adv reward function reads `accuracy`,
`answer_pred`, `format`, and `split` from `extra_info` / per-sample metadata.

This script writes a *formatted* parquet with:
  - extra_info = {"split": "train"}  (so reward path is the train path)
  - data_source = "dapo-math"
  - everything else preserved

The formatted parquet lives next to the original (datasets/dapo-math-17k.formatted.parquet).
"""
import os
import sys

import pandas as pd

# Paths are env-overridable so this repo stays git-safe (no absolute paths
# in source). Defaults match the local /home/luorongchuan/workspace_135 layout.
SRC = os.environ.get("VR_DAPO_SRC", "/home/luorongchuan/workspace_135/datasets/dapo-math-17k.parquet")
DST = os.environ.get("VR_DAPO_DST", "/home/luorongchuan/workspace_135/datasets/dapo-math-17k.formatted.parquet")


def main():
    df = pd.read_parquet(SRC)
    print(f"[info] read {len(df)} rows from {SRC}")

    # Inject split='train' into extra_info.
    def _merge(ei):
        if ei is None:
            base = {}
        elif isinstance(ei, dict):
            base = dict(ei)
        else:
            try:
                base = dict(ei.tolist())
            except Exception:
                base = {}
        base["split"] = "train"
        return base

    df["extra_info"] = df["extra_info"].map(_merge)

    # Stable data_source (used by reward manager bucketing).
    df["data_source"] = df.get("data_source", pd.Series(["dapo-math"] * len(df)))
    df["data_source"] = df["data_source"].fillna("dapo-math")

    out_dir = os.path.dirname(os.path.abspath(DST))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(DST, index=False)
    print(f"[ok] wrote {DST} ({len(df)} rows)")


if __name__ == "__main__":
    main()