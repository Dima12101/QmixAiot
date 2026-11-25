"""CPU inference timing of the two QMIX dispatcher architectures.

These are untrained weights; only neural policy inference with a precomputed
mask is measured. Mask construction, environment, network transmission,
and training time are excluded.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .environment import EnvironmentConfig, IoTEdgeEnvironment
from .experiments import write_csv
from .learning import Learner


def run(output: str, repeats: int = 150, seed: int = 273):
    rows = []
    for nodes in (10, 25, 50, 100):
        cfg = EnvironmentConfig(nodes=nodes, devices=2 * nodes, horizon=60)
        env = IoTEdgeEnvironment(cfg, seed=seed)
        for method in ("adapted_qmix", "work_conserving_qmix", "structured_qmix"):
            learner = Learner(cfg, seed=seed, method=method)
            obs = env.observations()
            mask = learner.action_mask(env)
            for _ in range(25):
                learner.reset_hidden()
                learner.choose(obs, mask)
            durations = []
            for _ in range(repeats):
                learner.reset_hidden()
                start = time.perf_counter_ns()
                learner.choose(obs, mask)
                durations.append((time.perf_counter_ns() - start) / 1e6)
            rows.append({
                "nodes": nodes, "method": method, "obs_dim": env.obs_dim,
                "actions_per_agent": env.action_dim,
                "median_ms": float(np.median(durations)),
                "p95_ms": float(np.percentile(durations, 95)),
                "repeats": repeats,
                "note": "Untrained weights, CPU; excludes mask construction, communications, environment, training",
            })
            print(f"{nodes} {method}: median {rows[-1]['median_ms']:.3f} ms", flush=True)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_improved2026/inference_scaling.csv")
    parser.add_argument("--repeats", type=int, default=150)
    args = parser.parse_args()
    run(args.output, args.repeats)
