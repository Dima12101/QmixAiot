"""Record observable decision patterns without changing simulator behavior."""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from .environment import EnvironmentConfig, IoTEdgeEnvironment, heuristic_actions
from .learning import Learner


def diagnose(method: str, cfg: EnvironmentConfig, seed: int, checkpoint: str | None,
             force_dispatch: bool = False):
    env = IoTEdgeEnvironment(cfg, seed=seed)
    learner = None
    if checkpoint:
        learner = Learner(cfg, method=method, seed=0)
        learner.load(checkpoint)
        learner.reset_hidden()
    obs = env.observations()
    count = Counter()
    destinations = Counter()
    mask_sizes = []
    while True:
        mask = env.action_masks()
        if force_dispatch:
            mask[mask[:, 1:].any(axis=1), 0] = False
        if learner is None:
            action = heuristic_actions(env, method, env.t)
        else:
            action = learner.choose(obs, mask)
        for origin in range(cfg.nodes):
            if env.pending[origin]:
                count["arrivals_seen"] += 1
                mask_sizes.append(int(mask[origin, 1:].sum()))
                if action[origin] == 0:
                    count["rejected_by_choice"] += 1
                    if mask[origin, 1:].any():
                        count["avoidable_choice_reject"] += 1
                else:
                    destinations[int(action[origin]) - 1] += 1
            elif action[origin] != 0:
                count["action_when_idle"] += 1
        obs, _, done, _ = env.step(action)
        if done:
            break
    return {
        "method": method, "seed": seed, "checkpoint": checkpoint,
        "decision_counts": dict(count), "mean_feasible_destinations": float(np.mean(mask_sizes)),
        "destination_total": sum(destinations.values()),
        "destination_unique": len(destinations),
        "destination_top5": destinations.most_common(5),
        "rejection_reasons": dict(env.reasons), "metrics": env.metrics(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method")
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int, default=4000)
    parser.add_argument("--arrival", type=float, default=.24)
    parser.add_argument("--force-dispatch", action="store_true")
    args = parser.parse_args()
    cfg = EnvironmentConfig(arrival_per_device=args.arrival, horizon=120)
    print(json.dumps(diagnose(args.method, cfg, args.seed, args.checkpoint, args.force_dispatch), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
