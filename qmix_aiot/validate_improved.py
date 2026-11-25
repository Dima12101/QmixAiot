"""Fail-fast integrity checks for the complete independent-test study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .improved_study import NEW_METHODS, REFERENCE_METHODS
from .regime_study import HEURISTICS, _read_csv, summarize


def validate(path: str = "results_improved2026", reference_dir: str = "results_regime2026"):
    directory = Path(path)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    rows = _read_csv(directory / "raw_evaluation.csv")
    summary = _read_csv(directory / "summary.csv")
    training_seeds = set(metadata["train_seeds"])
    eval_seeds = set(metadata["evaluation_seeds"])
    checkpoints = set(metadata["checkpoints"])
    regimes = set(r["name"] for r in metadata["regimes"])
    methods = set(NEW_METHODS + REFERENCE_METHODS + HEURISTICS)
    assert training_seeds and eval_seeds.isdisjoint(training_seeds)
    assert len(training_seeds) == 5 and len(eval_seeds) == 8
    assert len(regimes) == 10 and methods == set(r["method"] for r in rows)
    expected_rows = len(regimes) * len(eval_seeds) * (
        (len(NEW_METHODS) + len(REFERENCE_METHODS)) * len(training_seeds) * len(checkpoints)
        + len(HEURISTICS)
    )
    assert len(rows) == expected_rows, (len(rows), expected_rows)
    keys = set()
    for r in rows:
        method = r["method"]
        checkpoint = int(r["checkpoint"])
        seed = int(r["train_seed"])
        eval_seed = int(r["eval_seed"])
        key = method, checkpoint, seed, eval_seed, r["regime"]
        assert key not in keys
        keys.add(key)
        assert eval_seed in eval_seeds and r["regime"] in regimes
        assert checkpoint == 0 and seed == -1 if method in HEURISTICS else checkpoint in checkpoints and seed in training_seeds
        assert float(r["generated"]) > 0
        assert 0 <= float(r["dcr"]) <= 100
        assert 0 <= float(r["trr"]) <= 100
        assert 0 <= float(r["jain"]) <= 1 + 1e-7
        assert np.isclose(float(r["on_time"]) / float(r["generated"]) * 100, float(r["dcr"]))
        rejected = float(r["generated"]) * float(r["trr"]) / 100
        assert np.isclose(rejected + float(r["on_time"]) + float(r["late"]) + float(r["unfinished"]), float(r["generated"]))
    computed = summarize(rows)
    assert len(computed) == len(summary)
    look = {(r["method"], int(r["checkpoint"]), r["regime"]): r for r in summary}
    for record in computed:
        key = record["method"], int(record["checkpoint"]), record["regime"]
        other = look[key]
        assert int(record["evaluation_records"]) == int(other["evaluation_records"])
        assert int(record["independent_units"]) == int(other["independent_units"])
        for metric in ("dcr", "trr", "mrt", "jain", "energy_proxy"):
            assert np.isclose(float(record[f"{metric}_mean"]), float(other[f"{metric}_mean"]), equal_nan=True)
            assert np.isclose(float(record[f"{metric}_sd"]), float(other[f"{metric}_sd"]), equal_nan=True)
    for method in NEW_METHODS:
        for seed in training_seeds:
            for checkpoint in checkpoints:
                assert (directory / f"model_{method}_seed{seed}_ep{checkpoint}.pt").is_file()
    for method in REFERENCE_METHODS:
        for seed in training_seeds:
            for checkpoint in checkpoints:
                assert (Path(reference_dir) / f"model_{method}_seed{seed}_ep{checkpoint}.pt").is_file()
    return {
        "valid": True,
        "raw_records": len(rows), "summary_groups": len(summary),
        "train_seeds": len(training_seeds), "evaluation_seeds": len(eval_seeds),
        "regimes": len(regimes), "learned_variants": len(NEW_METHODS) + len(REFERENCE_METHODS),
        "heuristics": len(HEURISTICS),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results_improved2026")
    parser.add_argument("--reference-dir", default="results_regime2026")
    args = parser.parse_args()
    print(json.dumps(validate(args.results, args.reference_dir), ensure_ascii=False))
