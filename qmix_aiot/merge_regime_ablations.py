"""Merge independently trained ablation batches into the controlled study.

The main batch and auxiliary component ablations may be trained in separate
processes, but all use the identical simulator, training seeds and test streams.
This command rejects duplicate method runs and incomplete input batches.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch

from .experiments import write_csv
from .regime_study import LEARNED, HEURISTICS, REGIMES, _read_csv, cpu_identification, summarize


def merge(primary: str, auxiliary: str) -> None:
    dest, source = Path(primary), Path(auxiliary)
    main_meta = json.loads((dest / "metadata.json").read_text(encoding="utf-8"))
    aux_meta = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    for key in ("train_config", "checkpoints", "train_episodes", "training_seeds",
                "held_out_seeds", "regimes", "eval_horizon", "training"):
        if main_meta[key] != aux_meta[key]:
            raise ValueError(f"Incompatible study input: {key}")
    main_methods = set(main_meta["model_methods"])
    aux_methods = set(aux_meta["model_methods"])
    if main_methods & aux_methods:
        raise ValueError(f"Duplicate learned methods: {main_methods & aux_methods}")
    if main_methods | aux_methods != set(LEARNED):
        raise ValueError(f"Missing or unregistered methods: {(main_methods | aux_methods) ^ set(LEARNED)}")
    if set(main_meta["heuristics"]) != set(HEURISTICS) or aux_meta["heuristics"]:
        raise ValueError("Heuristics must occur only in the primary batch")

    main_raw = _read_csv(dest / "raw_evaluation.csv")
    aux_raw = _read_csv(source / "raw_evaluation.csv")
    rows = main_raw + aux_raw
    raw_keys = [(r["method"], r["checkpoint"], r["train_seed"],
                 r["eval_seed"], r["regime"]) for r in rows]
    if len(raw_keys) != len(set(raw_keys)):
        raise ValueError("Duplicate raw evaluations; aborting merge")
    expected = (len(LEARNED) * len(main_meta["training_seeds"]) *
                len(main_meta["checkpoints"]) * len(REGIMES) *
                len(main_meta["held_out_seeds"]) +
                len(HEURISTICS) * len(REGIMES) * len(main_meta["held_out_seeds"]))
    if len(rows) != expected:
        raise ValueError(f"Incomplete raw records: actual {len(rows)} vs expected {expected}")

    histories = _read_csv(dest / "training_history.csv") + _read_csv(source / "training_history.csv")
    history_keys = [(r["method"], r["train_seed"], r["episode"]) for r in histories]
    expected_hist = len(LEARNED) * len(main_meta["training_seeds"]) * int(main_meta["train_episodes"])
    if len(histories) != expected_hist or len(history_keys) != len(set(history_keys)):
        raise ValueError("Incomplete/duplicate training history; aborting merge")

    for checkpoint in main_meta["checkpoints"]:
        for method in aux_methods:
            for seed in main_meta["training_seeds"]:
                name = f"model_{method}_seed{seed}_ep{checkpoint}.pt"
                if not (source / name).is_file():
                    raise FileNotFoundError(source / name)
    for file in source.glob("model_*.pt"):
        target = dest / file.name
        if target.exists():
            raise FileExistsError(f"Model would be overwritten: {target}")
        shutil.copy2(file, target)

    main_meta["model_methods"] = list(LEARNED)
    main_meta["design"] = "Load-response and component ablations with independently trained models"
    main_meta["scientific_controls"].append(
        "Recurrence, fairness penalty and energy proxy penalty are removed one at a time"
    )
    main_meta["parallel_training_batches"] = [str(dest), str(source)]
    main_meta["cpu_model"] = cpu_identification()
    main_meta["logical_cpus_visible"] = os.cpu_count()
    main_meta["torch_num_threads"] = min(2, torch.get_num_threads())
    main_meta["limitations"] = [
        f"{len(main_meta['training_seeds'])} training seeds are descriptive and do not establish small-effect significance"
        if line.startswith("Three training seeds") else line
        for line in main_meta["limitations"]
    ]
    main_meta["elapsed_wall_seconds_batches_sum"] = (
        main_meta["elapsed_wall_seconds"] + aux_meta["elapsed_wall_seconds"]
    )
    write_csv(dest / "raw_evaluation.csv", rows)
    write_csv(dest / "summary.csv", summarize(rows))
    write_csv(dest / "training_history.csv", histories)
    (dest / "metadata.json").write_text(
        json.dumps(main_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print({"raw_evaluations": len(rows), "trained_methods": len(LEARNED),
           "models": len(LEARNED) * len(main_meta["training_seeds"])})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", default="results_regime2026")
    parser.add_argument("--auxiliary", default="results_regime_ablations")
    args = parser.parse_args()
    merge(args.primary, args.auxiliary)
