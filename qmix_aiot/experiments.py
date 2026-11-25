"""Execute seeded comparisons and save the complete underlying observations.

All charts and tables must be generated from CSV records produced here.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .environment import EnvironmentConfig, IoTEdgeEnvironment, heuristic_actions
from .learning import Learner, generate_episode, train


MAIN_METHODS = ["adapted_qmix", "vanilla_qmix", "iql", "greedy", "round_robin"]
ABLATIONS = ["no_gru", "no_masks", "no_fairness", "no_energy"]
SCENARIOS = ["stable", "peaks", "failures", "heterogeneous"]
METRICS = ["dcr", "mrt", "jain", "trr", "energy_proxy", "unfinished", "generated"]


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("No rows")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_summary(rows: list[dict]):
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["method"], row["scenario"]].append(row)
    summary = []
    for (method, scenario), values in sorted(groups.items()):
        # For RL, first average evaluation episodes within each trained model.
        # Heuristics have no trained model: their independent units are the
        # separately seeded evaluation task streams themselves.
        per_training_seed: dict[int, list[dict]] = defaultdict(list)
        for value in values:
            independent_id = (
                int(value["eval_seed"])
                if method in {"greedy", "round_robin"}
                else int(value["train_seed"])
            )
            per_training_seed[independent_id].append(value)
        per_seed = [
            {metric: float(np.nanmean([v[metric] for v in samples])) for metric in METRICS}
            for samples in per_training_seed.values()
        ]
        record = {
            "method": method,
            "scenario": scenario,
            "runs": len(values),
            "independent_models": len(per_seed) if method not in {"greedy", "round_robin"} else 0,
            "independent_units": len(per_seed),
        }
        for metric in METRICS:
            vals = np.array([x[metric] for x in per_seed], dtype=float)
            record[f"{metric}_mean"] = float(np.nanmean(vals))
            record[f"{metric}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float("nan")
        summary.append(record)
    return summary


def evaluate_policy(
    method: str, cfg: EnvironmentConfig, seed: int,
    learner: Learner | None = None,
) -> dict:
    env = IoTEdgeEnvironment(cfg, seed=seed)
    obs = env.observations()
    if learner is not None:
        learner.reset_hidden()
    while True:
        if learner is None:
            action = heuristic_actions(env, method, env.t)
        else:
            action = learner.choose(obs, learner.action_mask(env), epsilon=0)
        obs, _, done, _ = env.step(action)
        if done:
            break
    return env.metrics()


def benchmark_inference(outdir: Path, seed: int, repeats: int = 200):
    result = []
    for n in (10, 25, 50, 100):
        cfg = EnvironmentConfig(nodes=n, devices=n * 2, horizon=60)
        model = Learner(cfg, seed=seed)
        env = IoTEdgeEnvironment(cfg, seed=seed)
        obs = env.observations()
        masks = env.action_masks()
        for _ in range(25):
            model.reset_hidden()
            model.choose(obs, masks)
        durations = []
        for _ in range(repeats):
            model.reset_hidden()
            before = time.perf_counter_ns()
            model.choose(obs, masks)
            durations.append((time.perf_counter_ns() - before) / 1e6)
        result.append({
            "nodes": n,
            "actions_per_agent": n + 1,
            "obs_dimension": model.agents.input_layer[0].in_features,
            "median_ms": float(np.median(durations)),
            "p95_ms": float(np.percentile(durations, 95)),
            "repeats": repeats,
            "note": "Untrained weights; end-to-end simultaneous local decisions on CPU",
        })
    write_csv(outdir / "inference_scaling.csv", result)
    return result


def run_experiments(
    output_dir: str, train_episodes: int = 55, train_seeds: int = 3,
    eval_seeds: int = 5, horizon: int = 90, eval_horizon: int = 120,
    nodes: int = 25, devices: int = 50, include_ablations: bool = True,
    inference_repeats: int = 200,
):
    if train_seeds < 2 or eval_seeds < 2:
        raise ValueError("At least two independent seeds required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = EnvironmentConfig(nodes=nodes, devices=devices, horizon=horizon)
    rows, histories = [], []
    methods = MAIN_METHODS + (ABLATIONS if include_ablations else [])
    start_wall = time.perf_counter()
    for method in methods:
        training_seeds = range(train_seeds) if method not in {"greedy", "round_robin"} else [-1]
        for train_seed in training_seeds:
            learner = None
            if train_seed >= 0:
                print(f"Training {method} seed={train_seed} ({train_episodes} episodes)", flush=True)
                learner, history = train(cfg, method, seed=100 + train_seed, episodes=train_episodes)
                for item in history:
                    histories.append({"method": method, "train_seed": train_seed, **item})
                learner.save(str(output / f"model_{method}_seed{train_seed}.pt"))
            for scenario in SCENARIOS:
                e_cfg = EnvironmentConfig(nodes=nodes, devices=devices, horizon=eval_horizon, scenario=scenario)
                for eval_seed in range(eval_seeds):
                    record = evaluate_policy(method, e_cfg, seed=2000 + eval_seed, learner=learner)
                    rows.append({"method": method, "train_seed": train_seed, "eval_seed": eval_seed,
                                 "scenario": scenario, **record})
            print(f"Completed method={method} seed={train_seed}", flush=True)
        # Save incremental checkpoints after each method.
        write_csv(output / "raw_evaluation.csv", rows)
        write_csv(output / "summary.csv", make_summary(rows))
        if histories:
            write_csv(output / "training_history.csv", histories)
    scaling = benchmark_inference(output, seed=273, repeats=inference_repeats)
    write_csv(output / "raw_evaluation.csv", rows)
    summary = make_summary(rows)
    write_csv(output / "summary.csv", summary)
    if histories:
        write_csv(output / "training_history.csv", histories)
    metadata = {
        "experiment": "QmixAiot revision, synthetic simulator",
        "base_source_commit": "532b142a029590a3f635c36052f1866cea994a40",
        "config": asdict(cfg),
        "train_episodes": train_episodes,
        "training_seeds": [100 + x for x in range(train_seeds)],
        "evaluation_seeds": [2000 + x for x in range(eval_seeds)],
        "eval_horizon": eval_horizon,
        "evaluation_scenarios": SCENARIOS,
        "algorithms": methods,
        "replay_length_episodes": 80,
        "gradient_updates_per_episode": 3,
        "gradient_batch_size_episodes": 3,
        "optimizer": "Adam lr=0.0004",
        "gamma": .98,
        "epsilon": "linear 0.85 -> 0.05 over 80% of episodes",
        "target_update": "hard synchronization every 20 optimizer updates",
        "simulated_step_s": 1,
        "measurement": "time.perf_counter_ns, CPU untrained inference, median and p95",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "elapsed_wall_seconds": time.perf_counter() - start_wall,
        "inference_scaling": scaling,
        "limitations": [
            "Synthetic resource and workload ranges; not measured warehouse traffic",
            "Energy is dimensionless processed-work proxy, not watts or joules",
            "Trained policy sizes fixed to node count; scaling benchmark uses untrained networks",
            "Training seeds, not repeated validation episodes, determine independent model count",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"finished": True, "seconds": metadata["elapsed_wall_seconds"], "methods": methods}, ensure_ascii=False), flush=True)
    return summary


def cli():
    parser = argparse.ArgumentParser(description="Reproduce QMIX-AIoT article experiments")
    parser.add_argument("--output", default="results")
    parser.add_argument("--train-episodes", type=int, default=55)
    parser.add_argument("--train-seeds", type=int, default=3)
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--eval-horizon", type=int, default=120)
    parser.add_argument("--nodes", type=int, default=25)
    parser.add_argument("--devices", type=int, default=50)
    parser.add_argument("--without-ablations", action="store_true")
    parser.add_argument("--inference-repeats", type=int, default=200)
    args = parser.parse_args()
    run_experiments(
        args.output, args.train_episodes, args.train_seeds, args.eval_seeds,
        args.horizon, args.eval_horizon, args.nodes, args.devices,
        include_ablations=not args.without_ablations,
        inference_repeats=args.inference_repeats,
    )


if __name__ == "__main__":
    cli()
