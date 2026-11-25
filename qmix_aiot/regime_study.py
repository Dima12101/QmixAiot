"""Controlled follow-up study: load regimes, information ablation, training budget.

Results are *synthetic simulation outcomes*, not deployment measurements.
The same held-out seeds and workload parameters are used for each method.
Every statistical unit for a learned method is an independently trained model;
held-out episodes from a model are averaged before calculating dispersion.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .environment import EnvironmentConfig, IoTEdgeEnvironment
from .experiments import METRICS, benchmark_inference, evaluate_policy, write_csv
from .learning import Learner, generate_episode


LEARNED = (
    "adapted_qmix", "no_public", "vanilla_qmix", "iql",
    "no_gru", "no_fairness", "no_energy",
)
HEURISTICS = ("greedy", "round_robin")


def cpu_identification() -> str:
    model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    return model or "not provided by OS"


@dataclass(frozen=True)
class Regime:
    name: str
    scenario: str
    arrival: float


REGIMES = (
    Regime("steady_006", "stable", 0.06),
    Regime("steady_012", "stable", 0.12),
    Regime("steady_024", "stable", 0.24),
    Regime("steady_036", "stable", 0.36),
    Regime("peaks_012", "peaks", 0.12),
    Regime("peaks_024", "peaks", 0.24),
    Regime("failure_012", "failures", 0.12),
    Regime("failure_024", "failures", 0.24),
    Regime("hetero_012", "heterogeneous", 0.12),
    Regime("hetero_024", "heterogeneous", 0.24),
)


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], int(row["checkpoint"]), row["regime"])].append(row)
    result: list[dict] = []
    for (method, checkpoint, regime), group in sorted(groups.items()):
        units: dict[int, list[dict]] = defaultdict(list)
        for record in group:
            unit = int(record["eval_seed"] if method in HEURISTICS else record["train_seed"])
            units[unit].append(record)
        record = {
            "method": method,
            "checkpoint": checkpoint,
            "regime": regime,
            "scenario": group[0]["scenario"],
            "arrival_per_device": float(group[0]["arrival_per_device"]),
            "evaluation_records": len(group),
            "independent_units": len(units),
            "unit_kind": "held_out_stream" if method in HEURISTICS else "trained_model",
        }
        for metric in METRICS:
            values = np.array([
                np.mean([float(r[metric]) for r in samples]) for samples in units.values()
            ], dtype=float)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_sd"] = (
                float(np.std(values, ddof=1)) if len(values) >= 2 else float("nan")
            )
        result.append(record)
    return result


def _evaluation_rows(
    *, method: str, checkpoint: int, train_seed: int,
    learner: Learner | None, nodes: int, devices: int, horizon: int,
    eval_seeds: int,
) -> list[dict]:
    rows: list[dict] = []
    for regime in REGIMES:
        cfg = EnvironmentConfig(
            nodes=nodes, devices=devices, horizon=horizon,
            arrival_per_device=regime.arrival, scenario=regime.scenario,
        )
        for offset in range(eval_seeds):
            evaluation_seed = 2000 + offset
            metrics = evaluate_policy(method, cfg, seed=evaluation_seed, learner=learner)
            rows.append({
                "method": method, "checkpoint": checkpoint, "train_seed": train_seed,
                "eval_seed": evaluation_seed, "regime": regime.name,
                "scenario": regime.scenario, "arrival_per_device": regime.arrival,
                **metrics,
            })
    return rows


def run_study(
    output: str = "results_regime2026", *, episodes: int = 110,
    checkpoints: tuple[int, ...] = (55, 110), train_seeds: int = 5,
    eval_seeds: int = 8, nodes: int = 25, devices: int = 50,
    train_horizon: int = 90, eval_horizon: int = 120,
) -> None:
    if train_seeds < 2 or eval_seeds < 2 or min(checkpoints) < 3 or max(checkpoints) != episodes:
        raise ValueError("Need multiple seeds and increasing checkpoints ending at episodes")
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    raw_path = dest / "raw_evaluation.csv"
    history_path = dest / "training_history.csv"
    rows = _read_csv(raw_path)
    history = _read_csv(history_path)
    start = time.perf_counter()
    train_cfg = EnvironmentConfig(nodes=nodes, devices=devices, horizon=train_horizon)
    for method in LEARNED:
        for i in range(train_seeds):
            seed = 100 + i
            final_path = dest / f"model_{method}_seed{seed}_ep{episodes}.pt"
            expected_rows = len(REGIMES) * eval_seeds * len(checkpoints)
            existing_rows = sum(x["method"] == method and int(x["train_seed"]) == seed for x in rows)
            if final_path.is_file() and existing_rows == expected_rows:
                print(f"Resume: complete {method} seed {seed}; skipping", flush=True)
                continue
            rows = [x for x in rows if not (x["method"] == method and int(x["train_seed"]) == seed)]
            history = [x for x in history if not (x["method"] == method and int(x["train_seed"]) == seed)]
            learner = Learner(train_cfg, seed=seed, method=method)
            for episode_idx in range(episodes):
                epsilon = max(.05, .85 * (1 - episode_idx / max(1.0, episodes * .8)))
                env = IoTEdgeEnvironment(train_cfg, seed=seed * 100000 + episode_idx)
                batch, metrics = generate_episode(env, learner, epsilon)
                learner.replay.append(batch)
                losses = [learner.train_step() for _ in range(3)]
                history.append({
                    "method": method, "train_seed": seed, "episode": episode_idx + 1,
                    "epsilon": epsilon, "episode_reward": float(batch.rewards.sum()),
                    "last_loss": next((x for x in reversed(losses) if x is not None), float("nan")),
                    **metrics,
                })
                if (episode_idx + 1) in checkpoints:
                    checkpoint = episode_idx + 1
                    learner.save(str(dest / f"model_{method}_seed{seed}_ep{checkpoint}.pt"))
                    rows.extend(_evaluation_rows(
                        method=method, checkpoint=checkpoint, train_seed=seed,
                        learner=learner, nodes=nodes, devices=devices,
                        horizon=eval_horizon, eval_seeds=eval_seeds,
                    ))
                    write_csv(raw_path, rows)
                    write_csv(dest / "summary.csv", summarize(rows))
                    write_csv(history_path, history)
                    print(f"Completed {method} seed={seed} ep={checkpoint}; {len(rows)} evaluations; "
                          f"elapsed={time.perf_counter() - start:.0f}s", flush=True)
    for method in HEURISTICS:
        expected = len(REGIMES) * eval_seeds
        if sum(x["method"] == method for x in rows) == expected:
            continue
        rows = [x for x in rows if x["method"] != method]
        rows.extend(_evaluation_rows(
            method=method, checkpoint=0, train_seed=-1, learner=None,
            nodes=nodes, devices=devices, horizon=eval_horizon, eval_seeds=eval_seeds,
        ))
        write_csv(raw_path, rows)
        write_csv(dest / "summary.csv", summarize(rows))
    benchmark_inference(dest, seed=273, repeats=250)
    metadata = {
        "design": "Load-response and component ablations, independent seeds, synthetic IoT edge",
        "train_config": asdict(train_cfg),
        "model_methods": LEARNED, "heuristics": HEURISTICS,
        "checkpoints": checkpoints, "train_episodes": episodes,
        "training_seeds": [100 + i for i in range(train_seeds)],
        "held_out_seeds": [2000 + i for i in range(eval_seeds)],
        "regimes": [asdict(r) for r in REGIMES],
        "eval_horizon": eval_horizon,
        "algorithm": "Same QMIX/PyTorch implementation as first revision; observations now include origin-local network estimates",
        "scientific_controls": [
            "Adapted and no_public differ only in neighboring public announcements; reward and masks match",
            "No-GRU differs only in recurrence; no_fairness and no_energy remove individual reward penalties",
            "All policies see the same local link estimates; dispatch heuristic uses shared mask and advertised queue counts",
            "Independent unit is a trained model (learned) or task-stream seed (heuristics)",
            "Held-out seeds reused across methods, not independent model replicates",
            "No scenario-specific re-training or hyperparameter selection on evaluation seeds",
        ],
        "training": {"optimizer": "Adam", "learning_rate": 0.0004, "gamma": 0.98,
                     "replay_capacity_episodes": 80, "updates_per_episode": 3,
                     "batch_size_episodes": 3, "target_sync_updates": 20,
                     "epsilon": "linear 0.85 to 0.05 by 80% of episodes"},
        "torch": torch.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "cpu_model": cpu_identification(),
        "logical_cpus_visible": os.cpu_count(), "torch_num_threads": torch.get_num_threads(),
        "elapsed_wall_seconds": time.perf_counter() - start,
        "limitations": [
            "Synthetic model; no empirical calibration or field measurements",
            "Greedy is a feasibility-aware local policy, not a global optimal bound",
            f"{train_seeds} training seeds are descriptive and do not establish small-effect significance",
            "Observation exchange overhead is modeled as announcements, not measured network packets",
            "Changing workload rate changes total generated tasks and resource saturation",
        ],
    }
    (dest / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(raw_path, rows)
    write_csv(dest / "summary.csv", summarize(rows))
    write_csv(history_path, history)
    print(f"Study complete: {len(rows)} evaluation records, "
          f"{time.perf_counter() - start:.1f}s", flush=True)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results_regime2026")
    parser.add_argument("--episodes", type=int, default=110)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[55, 110])
    parser.add_argument("--train-seeds", type=int, default=5)
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=25)
    parser.add_argument("--devices", type=int, default=50)
    parser.add_argument("--train-horizon", type=int, default=90)
    parser.add_argument("--eval-horizon", type=int, default=120)
    args = parser.parse_args()
    run_study(args.output, episodes=args.episodes, checkpoints=tuple(args.checkpoints),
              train_seeds=args.train_seeds, eval_seeds=args.eval_seeds,
              nodes=args.nodes, devices=args.devices,
              train_horizon=args.train_horizon, eval_horizon=args.eval_horizon)


if __name__ == "__main__":
    cli()
