"""Audit raw evaluation integrity before generating journal text."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .experiments import METRICS
from .regime_study import HEURISTICS, LEARNED, REGIMES, summarize


def verify(directory: str = "results_regime2026") -> dict:
    base = Path(directory)
    meta = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
    with (base / "raw_evaluation.csv").open(newline="", encoding="utf-8") as file:
        raw = list(csv.DictReader(file))
    with (base / "summary.csv").open(newline="", encoding="utf-8") as file:
        summary = list(csv.DictReader(file))
    train_seeds = meta["training_seeds"]
    eval_seeds = meta["held_out_seeds"]
    checkpoints = meta["checkpoints"]
    regimes = {x.name for x in REGIMES}
    expected = (len(LEARNED) * len(train_seeds) * len(checkpoints)
                + len(HEURISTICS)) * len(eval_seeds) * len(regimes)
    if len(raw) != expected:
        raise AssertionError(f"Expected {expected} evaluation rows, found {len(raw)}")
    expected_keys = {
        (method, seed, checkpoint, regime, e_seed)
        for method in LEARNED
        for seed in train_seeds
        for checkpoint in checkpoints
        for regime in regimes
        for e_seed in eval_seeds
    } | {
        (method, -1, 0, regime, e_seed)
        for method in HEURISTICS
        for regime in regimes
        for e_seed in eval_seeds
    }
    actual_keys = {
        (r["method"], int(r["train_seed"]), int(r["checkpoint"]), r["regime"], int(r["eval_seed"]))
        for r in raw
    }
    if actual_keys != expected_keys:
        raise AssertionError(f"Missing {len(expected_keys - actual_keys)} keys; excess {len(actual_keys - expected_keys)}")
    if len(raw) != len(actual_keys):
        raise AssertionError("Duplicate raw evaluation keys")
    # A paired comparison requires the same workload at each evaluation seed.
    generated = defaultdict(set)
    for r in raw:
        generated[r["regime"], int(r["eval_seed"])].add(round(float(r["generated"]), 4))
        if not (0 <= float(r["dcr"]) <= 100 and 0 <= float(r["trr"]) <= 100):
            raise AssertionError("Out-of-range percentage")
        if float(r["unfinished"]) < 0:
            raise AssertionError("Negative unfinished count")
    mismatches = [key for key, values in generated.items() if len(values) != 1]
    if mismatches:
        raise AssertionError(f"Nonidentical task counts in paired streams: {mismatches[:5]}")
    expected_summaries = (len(LEARNED) * len(checkpoints) + len(HEURISTICS)) * len(regimes)
    if len(summary) != expected_summaries:
        raise AssertionError(f"Expected {expected_summaries} summary rows, found {len(summary)}")
    expected_summary = {
        (row["method"], int(row["checkpoint"]), row["regime"]): row
        for row in summarize(raw)
    }
    for row in summary:
        key = row["method"], int(row["checkpoint"]), row["regime"]
        expected_row = expected_summary.get(key)
        if expected_row is None:
            raise AssertionError(f"Unknown summary group {key}")
        for metric in METRICS:
            for stat in ("mean", "sd"):
                field = f"{metric}_{stat}"
                if not np.isclose(float(row[field]), float(expected_row[field]),
                                  rtol=1e-8, atol=1e-8, equal_nan=True):
                    raise AssertionError(f"Summary not reproducible from raw data: {key} {field}")
    for method in LEARNED:
        for seed in train_seeds:
            for checkpoint in checkpoints:
                model = base / f"model_{method}_seed{seed}_ep{checkpoint}.pt"
                if not model.is_file():
                    raise AssertionError(f"Missing trained checkpoint {model}")
    return {
        "evaluation_records": len(raw),
        "summary_groups": len(summary),
        "identical_task_counts_for_paired_streams": True,
        "training_replicates": len(train_seeds),
        "held_out_streams_per_regime": len(eval_seeds),
        "regime_count": len(regimes),
        "learned_checkpoints": len(LEARNED) * len(train_seeds) * len(checkpoints),
    }


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results_regime2026")
    a = p.parse_args()
    print(json.dumps(verify(a.results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
