"""Evaluate work-conserving and targetwise QMIX against matched baselines.

The test streams (default seeds 4000-4007) are distinct from the exploratory
streams and were fixed before training the new candidate methods.  Existing
baseline checkpoints are read-only and evaluated on those same test streams.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .environment import EnvironmentConfig
from .experiments import benchmark_inference, evaluate_policy, write_csv
from .learning import Learner, generate_episode
from .regime_study import HEURISTICS, REGIMES, _read_csv, cpu_identification, summarize


NEW_METHODS = (
    "work_conserving_qmix", "structured_qmix", "structured_no_public",
    "structured_optional_reject",
)
REFERENCE_METHODS = (
    "adapted_qmix", "no_public", "vanilla_qmix", "iql",
    "no_gru", "no_fairness", "no_energy",
)


def evaluate_regimes(method, checkpoint, train_seed, learner, *, eval_seed_start,
                     eval_seeds, nodes, devices, eval_horizon, regimes):
    rows = []
    for regime in regimes:
        config = EnvironmentConfig(nodes=nodes, devices=devices, horizon=eval_horizon,
                                   arrival_per_device=regime.arrival, scenario=regime.scenario)
        for evaluation_seed in range(eval_seed_start, eval_seed_start + eval_seeds):
            metrics = evaluate_policy(method, config, seed=evaluation_seed, learner=learner)
            rows.append({
                "method": method, "checkpoint": checkpoint,
                "train_seed": train_seed, "eval_seed": evaluation_seed,
                "regime": regime.name, "scenario": regime.scenario,
                "arrival_per_device": regime.arrival, **metrics,
            })
    return rows


def run(output: str, *, methods: tuple[str, ...] = NEW_METHODS,
        episodes: int = 110, checkpoints: tuple[int, ...] = (55, 110),
        train_seeds: int = 5, eval_seeds: int = 8, eval_seed_start: int = 4000,
        nodes: int = 25, devices: int = 50, train_horizon: int = 90,
        eval_horizon: int = 120, include_reference: bool = True,
        reference_dir: str = "results_regime2026", all_regimes: bool = True):
    if not checkpoints or max(checkpoints) != episodes or min(checkpoints) < 3:
        raise ValueError("Checkpoints must be positive and end at total episodes")
    if eval_seed_start < 3000:
        raise ValueError("Experimental test streams must not reuse the old seed interval")
    outdir = Path(output)
    outdir.mkdir(parents=True, exist_ok=True)
    raw = outdir / "raw_evaluation.csv"
    history_path = outdir / "training_history.csv"
    rows, history = _read_csv(raw), _read_csv(history_path)
    start = time.perf_counter()
    train_cfg = EnvironmentConfig(nodes=nodes, devices=devices, horizon=train_horizon)
    regimes = REGIMES if all_regimes else tuple(
        regime for regime in REGIMES if regime.name in {"steady_024", "steady_036", "failure_024"}
    )
    for method in methods:
        if method not in NEW_METHODS:
            raise ValueError(f"Not a new experimental method: {method}")
        for idx in range(train_seeds):
            seed = 100 + idx
            expected = len(regimes) * eval_seeds * len(checkpoints)
            complete = sum(r["method"] == method and int(r["train_seed"]) == seed for r in rows) == expected
            if complete and (outdir / f"model_{method}_seed{seed}_ep{episodes}.pt").is_file():
                print(f"Already complete: {method}/{seed}", flush=True)
                continue
            rows = [r for r in rows if not (r["method"] == method and int(r["train_seed"]) == seed)]
            history = [r for r in history if not (r["method"] == method and int(r["train_seed"]) == seed)]
            learner = Learner(train_cfg, seed=seed, method=method)
            for episode_idx in range(episodes):
                epsilon = max(.05, .85 * (1 - episode_idx / max(1.0, episodes * .8)))
                from .environment import IoTEdgeEnvironment
                env = IoTEdgeEnvironment(train_cfg, seed=seed * 100000 + episode_idx)
                trial, metrics = generate_episode(env, learner, epsilon)
                learner.replay.append(trial)
                losses = [learner.train_step() for _ in range(3)]
                history.append({
                    "method": method, "train_seed": seed, "episode": episode_idx + 1,
                    "epsilon": epsilon, "episode_reward": float(trial.rewards.sum()),
                    "last_loss": next((x for x in reversed(losses) if x is not None), float("nan")),
                    **metrics,
                })
                if episode_idx + 1 in checkpoints:
                    checkpoint = episode_idx + 1
                    learner.save(str(outdir / f"model_{method}_seed{seed}_ep{checkpoint}.pt"))
                    rows.extend(evaluate_regimes(
                        method, checkpoint, seed, learner,
                        eval_seed_start=eval_seed_start, eval_seeds=eval_seeds,
                        nodes=nodes, devices=devices, eval_horizon=eval_horizon, regimes=regimes,
                    ))
                    write_csv(raw, rows)
                    write_csv(outdir / "summary.csv", summarize(rows))
                    write_csv(history_path, history)
                    print(f"New {method} seed={seed} ep={checkpoint}: {len(rows)} test records, "
                          f"{time.perf_counter() - start:.0f}s", flush=True)
    if include_reference:
        for method in REFERENCE_METHODS:
            for idx in range(train_seeds):
                seed = 100 + idx
                expected = len(regimes) * eval_seeds * len(checkpoints)
                if sum(r["method"] == method and int(r["train_seed"]) == seed for r in rows) == expected:
                    continue
                rows = [r for r in rows if not (r["method"] == method and int(r["train_seed"]) == seed)]
                learner = Learner(train_cfg, seed=seed, method=method)
                for checkpoint in checkpoints:
                    checkpoint_path = Path(reference_dir) / f"model_{method}_seed{seed}_ep{checkpoint}.pt"
                    if not checkpoint_path.is_file():
                        raise FileNotFoundError(f"Matched reference checkpoint missing: {checkpoint_path}")
                    learner.load(str(checkpoint_path))
                    rows.extend(evaluate_regimes(
                        method, checkpoint, seed, learner,
                        eval_seed_start=eval_seed_start, eval_seeds=eval_seeds,
                        nodes=nodes, devices=devices, eval_horizon=eval_horizon, regimes=regimes,
                    ))
                    write_csv(raw, rows)
                    write_csv(outdir / "summary.csv", summarize(rows))
                print(f"Reference {method} seed={seed} tested on new streams", flush=True)
        for method in HEURISTICS:
            if sum(r["method"] == method for r in rows) == len(regimes) * eval_seeds:
                continue
            rows = [r for r in rows if r["method"] != method]
            rows.extend(evaluate_regimes(
                method, 0, -1, None, eval_seed_start=eval_seed_start,
                eval_seeds=eval_seeds, nodes=nodes, devices=devices,
                eval_horizon=eval_horizon, regimes=regimes,
            ))
            write_csv(raw, rows)
            write_csv(outdir / "summary.csv", summarize(rows))
    metadata = {
        "study": "Structured and work-conserving QMIX in synthetic edge-task scheduling",
        "train_config": asdict(train_cfg), "new_methods": methods,
        "reference_methods": REFERENCE_METHODS if include_reference else (),
        "reference_checkpoints": reference_dir if include_reference else None,
        "heuristics": HEURISTICS if include_reference else (),
        "train_seeds": [100 + i for i in range(train_seeds)],
        "evaluation_seeds": list(range(eval_seed_start, eval_seed_start + eval_seeds)),
        "train_episodes": episodes, "checkpoints": checkpoints,
        "train_horizon": train_horizon, "eval_horizon": eval_horizon,
        "regimes": [asdict(r) for r in regimes],
        "training": {"optimizer": "Adam", "learning_rate": .0004, "gamma": .98,
                     "replay_capacity_episodes": 80, "updates_per_episode": 3,
                     "batch_size_episodes": 3, "target_sync_updates": 20,
                     "epsilon": "0.85 to 0.05 by 80 percent of training"},
        "scientific_controls": [
            "All methods evaluated on identical task streams using new seeds fixed before training candidates",
            "Matched baseline checkpoints trained on same original training seeds and training protocol",
            "Work-conserving dispatch removes optional rejection only when a legal destination exists",
            "Structured model changes target score parameterization, keeping QMIX mixer and reward",
            "Structured No-public retains feasibility mask and outgoing local links; excludes public numeric advertisements",
            "Independent statistical unit for learned policies is a separately initialized trained model",
            "No post-selection or tuning on evaluation streams",
        ],
        "platform": platform.platform(), "processor": cpu_identification(),
        "cpu_visible": os.cpu_count(), "torch_num_threads": torch.get_num_threads(),
        "torch": torch.__version__, "numpy": np.__version__,
        "wall_seconds_current_invocation": time.perf_counter() - start,
        "limitations": [
            "Synthetic simulator only, not empirical calibration or field measurements",
            "Reference baselines use checkpoint files from the identically parameterized controlled training study",
            "Feasibility mask already uses an approximate response-time prediction",
            "Energy proxy is not measured electrical energy",
        ],
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        write_csv(raw, rows)
        write_csv(outdir / "summary.csv", summarize(rows))
    if history:
        write_csv(history_path, history)
    print(f"Study completed: {len(rows)} evaluation records, {time.perf_counter()-start:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results_improved2026")
    parser.add_argument("--methods", nargs="+", choices=NEW_METHODS, default=list(NEW_METHODS))
    parser.add_argument("--episodes", type=int, default=110)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[55, 110])
    parser.add_argument("--train-seeds", type=int, default=5)
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--eval-seed-start", type=int, default=4000)
    parser.add_argument("--nodes", type=int, default=25)
    parser.add_argument("--devices", type=int, default=50)
    parser.add_argument("--train-horizon", type=int, default=90)
    parser.add_argument("--eval-horizon", type=int, default=120)
    parser.add_argument("--reference-dir", default="results_regime2026")
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--pilot-regimes", action="store_true")
    args = parser.parse_args()
    run(args.output, methods=tuple(args.methods), episodes=args.episodes,
        checkpoints=tuple(args.checkpoints), train_seeds=args.train_seeds,
        eval_seeds=args.eval_seeds, eval_seed_start=args.eval_seed_start,
        nodes=args.nodes, devices=args.devices, train_horizon=args.train_horizon,
        eval_horizon=args.eval_horizon, include_reference=not args.no_reference,
        reference_dir=args.reference_dir, all_regimes=not args.pilot_regimes)


if __name__ == "__main__":
    main()
