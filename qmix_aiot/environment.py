"""Discrete-event-inspired, one-second-step edge scheduling simulator.

Units: simulated seconds, giga-operations, GOp/s, GB and MB. Energy is a
dimensionless workload proxy, not measured electrical power. This is a
synthetic experimental environment, not a calibrated real deployment.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Task:
    identifier: int
    origin: int
    created: float
    work: float
    memory: float
    input_mb: float
    deadline: float
    priority: float
    destination: Optional[int] = None
    ready_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    outcome: str = "pending"


@dataclass(frozen=True)
class EnvironmentConfig:
    nodes: int = 25
    devices: int = 50
    horizon: int = 90
    arrival_per_device: float = 0.12
    scenario: str = "stable"
    max_pending: int = 12
    max_queue: int = 8

    def __post_init__(self):
        if self.nodes < 2 or self.devices < 1 or self.horizon < 5:
            raise ValueError("Need at least 2 nodes, 1 device and 5 steps")
        if self.scenario not in {"stable", "peaks", "failures", "heterogeneous"}:
            raise ValueError("Unrecognized scenario")


class IoTEdgeEnvironment:
    """One shared task stream, one dispatch decision per edge agent and step.

    A deterministic, priority-ordered arbitration resolves simultaneous
    assignments. Nodes process a single task at once, and tasks may queue.
    An arriving task must be dispatched by its ingress node. A decision may
    reject it (action 0) or send it to destination action-1.
    """

    def __init__(self, config: EnvironmentConfig, seed: int = 0):
        self.config = config
        self.seed = int(seed)
        self.num_agents = config.nodes
        self.action_dim = config.nodes + 1
        # Four public node attributes plus origin-local link estimates. Both
        # the learned dispatcher and greedy baseline can access these values.
        self.obs_dim = 10 + 6 * config.nodes
        self.state_dim = 5 * config.nodes + 2
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        cfg = self.config
        if cfg.scenario == "heterogeneous":
            self.cpu = self.rng.uniform(15.0, 60.0, cfg.nodes)
            self.memory = self.rng.uniform(12.0, 64.0, cfg.nodes)
        else:
            self.cpu = self.rng.uniform(25.0, 40.0, cfg.nodes)
            self.memory = self.rng.uniform(24.0, 48.0, cfg.nodes)
        self.link_latency = self.rng.uniform(0.03, 0.30, (cfg.nodes, cfg.nodes))
        self.link_bandwidth = self.rng.uniform(8.0, 50.0, (cfg.nodes, cfg.nodes))
        np.fill_diagonal(self.link_latency, 0.0)
        self.active = np.ones(cfg.nodes, dtype=np.bool_)
        self.failed_nodes = set(self.rng.choice(cfg.nodes, size=max(1, cfg.nodes // 5), replace=False).tolist())
        self.pending = [deque() for _ in range(cfg.nodes)]
        self.queues = [deque() for _ in range(cfg.nodes)]
        self.running: list[Optional[Task]] = [None for _ in range(cfg.nodes)]
        self.running_finish = np.zeros(cfg.nodes)
        self.tasks: dict[int, Task] = {}
        self.next_task_id = 0
        self.t = 0
        self.work_by_node = np.zeros(cfg.nodes, dtype=np.float64)
        self.energy_proxy = 0.0
        self.generated = 0
        self.on_time = 0
        self.late = 0
        self.rejected = 0
        self.reasons: dict[str, int] = {}
        self.response_times: list[float] = []
        self._generate_arrivals()
        return self.observations()

    def _rate_multiplier(self) -> float:
        if self.config.scenario == "peaks" and 25 <= self.t % 75 < 45:
            return 2.6
        return 1.0

    def _generate_arrivals(self):
        cfg = self.config
        arrivals = self.rng.poisson(
            cfg.arrival_per_device * cfg.devices * self._rate_multiplier() / cfg.nodes,
            size=cfg.nodes,
        )
        for ingress, n in enumerate(arrivals):
            for _ in range(int(n)):
                task = Task(
                    identifier=self.next_task_id,
                    origin=ingress,
                    created=float(self.t),
                    work=float(self.rng.uniform(14, 75)),
                    memory=float(self.rng.uniform(1, 11)),
                    input_mb=float(self.rng.uniform(.1, 3.0)),
                    deadline=float(self.t + self.rng.uniform(3, 12)),
                    priority=float(self.rng.uniform(.5, 1)),
                )
                self.tasks[task.identifier] = task
                self.next_task_id += 1
                self.generated += 1
                if len(self.pending[ingress]) >= cfg.max_pending:
                    self._reject(task, "ingress_overflow")
                else:
                    self.pending[ingress].append(task)

    def _reject(self, task: Task, reason: str):
        if task.outcome not in {"pending", "queued", "running"}:
            return
        task.outcome = "rejected"
        self.rejected += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def _failure_transition(self):
        if self.config.scenario != "failures":
            return
        start = self.config.horizon // 3
        end = self.config.horizon * 2 // 3
        if self.t == start:
            for node in self.failed_nodes:
                self.active[node] = False
                if self.running[node] is not None:
                    self._reject(self.running[node], "node_failure")
                    self.running[node] = None
                while self.queues[node]:
                    self._reject(self.queues[node].popleft(), "node_failure")
        elif self.t == end:
            for node in self.failed_nodes:
                self.active[node] = True

    def network_delay(self, origin: int, target: int, task: Task) -> float:
        if origin == target:
            return 0.0
        return float(self.link_latency[origin, target] + task.input_mb / self.link_bandwidth[origin, target])

    def expected_finish(self, origin: int, target: int, task: Task) -> float:
        if not self.active[target] or task.memory > self.memory[target]:
            return float("inf")
        # Only public node advertisements are used. Private queue job sizes and
        # future completion times are NOT exposed to the acting agent.
        published_backlog = len(self.queues[target]) + int(self.running[target] is not None)
        typical_job_work = (14.0 + 75.0) / 2.0
        return (self.t + self.network_delay(origin, target, task)
                + (published_backlog * typical_job_work + task.work) / self.cpu[target])

    def action_masks(self, constrained: bool = True) -> np.ndarray:
        masks = np.zeros((self.num_agents, self.action_dim), dtype=np.bool_)
        masks[:, 0] = True
        for origin in range(self.num_agents):
            if not self.pending[origin]:
                continue
            task = self.pending[origin][0]
            for target in range(self.num_agents):
                if not constrained:
                    masks[origin, 1 + target] = True
                else:
                    masks[origin, 1 + target] = (
                        self.active[target]
                        and len(self.queues[target]) < self.config.max_queue
                        and task.memory <= self.memory[target]
                        and self.expected_finish(origin, target, task) <= task.deadline
                    )
        return masks

    def observations(self) -> np.ndarray:
        n = self.num_agents
        public = np.stack([
            self.cpu / 60.0,
            self.memory / 64.0,
            np.array([len(q) + int(x is not None) for q, x in zip(self.queues, self.running)]) / 9.0,
            self.active.astype(np.float32),
        ], axis=1).astype(np.float32).ravel()
        result = np.zeros((n, self.obs_dim), dtype=np.float32)
        for i in range(n):
            task = self.pending[i][0] if self.pending[i] else None
            result[i, :10] = [
                self.cpu[i] / 60.0,
                self.memory[i] / 64.0,
                (len(self.queues[i]) + int(self.running[i] is not None)) / 9.0,
                min(len(self.pending[i]) / self.config.max_pending, 1),
                task.work / 75.0 if task else 0,
                task.memory / 12.0 if task else 0,
                (task.deadline - self.t) / 12.0 if task else 0,
                task.priority if task else 0,
                i / max(1, n - 1),
                self.t / self.config.horizon,
            ]
            result[i, 10:10 + 4 * n] = public
            # Each origin knows its own outgoing-link estimates. These are
            # also used by the public-information greedy reference policy.
            result[i, 10 + 4 * n:10 + 5 * n] = self.link_latency[i] / 0.30
            result[i, 10 + 5 * n:10 + 6 * n] = self.link_bandwidth[i] / 50.0
        return result

    def state(self) -> np.ndarray:
        n = self.num_agents
        data = np.stack([
            self.cpu / 60.0,
            self.memory / 64.0,
            np.array([len(x) for x in self.queues]) / max(1, self.config.max_queue),
            np.array([len(x) for x in self.pending]) / max(1, self.config.max_pending),
            self.active.astype(np.float32),
        ], axis=1).astype(np.float32).ravel()
        return np.concatenate([data, np.array([self.t / self.config.horizon, self.generated / max(1, self.t + 1) / n], dtype=np.float32)])

    def _dispatch(self, actions: np.ndarray):
        candidates: list[tuple[float, int, Task, int]] = []
        for origin, action in enumerate(actions):
            if not self.pending[origin]:
                continue
            task = self.pending[origin].popleft()
            if action == 0:
                self._reject(task, "chosen_reject")
            elif not (1 <= action <= self.num_agents):
                self._reject(task, "invalid_action")
            else:
                candidates.append((-task.priority, origin, task, int(action) - 1))
        for _, origin, task, target in sorted(candidates):
            if (not self.active[target] or task.memory > self.memory[target]
                    or len(self.queues[target]) >= self.config.max_queue):
                self._reject(task, "infeasible_assignment")
                continue
            task.destination = target
            task.ready_at = self.t + self.network_delay(origin, target, task)
            task.outcome = "queued"
            self.queues[target].append(task)

    def _process_nodes(self) -> tuple[int, int, float]:
        ontime_before, late_before = self.on_time, self.late
        energy_before = self.energy_proxy
        for node in range(self.num_agents):
            if not self.active[node]:
                continue
            now = float(self.t)
            end = now + 1.0
            while now < end:
                if self.running[node] is None:
                    if not self.queues[node]:
                        break
                    first = self.queues[node][0]
                    if first.ready_at is None or first.ready_at >= end:
                        break
                    now = max(now, first.ready_at)
                    task = self.queues[node].popleft()
                    task.started_at = now
                    task.outcome = "running"
                    self.running[node] = task
                    self.running_finish[node] = now + task.work / self.cpu[node]
                if self.running_finish[node] > end:
                    break
                now = self.running_finish[node]
                task = self.running[node]
                assert task is not None
                task.finished_at = now
                task.outcome = "completed"
                self.work_by_node[node] += task.work
                self.energy_proxy += task.work * (0.7 + 0.3 * self.cpu[node] / 60.0)
                self.response_times.append(now - task.created)
                if now <= task.deadline:
                    self.on_time += 1
                else:
                    self.late += 1
                self.running[node] = None
        return self.on_time - ontime_before, self.late - late_before, self.energy_proxy - energy_before

    def _expire_waiting(self):
        for collection in self.pending + self.queues:
            keep = deque()
            while collection:
                task = collection.popleft()
                if task.deadline <= self.t + 1.0:
                    self._reject(task, "deadline_expired")
                else:
                    keep.append(task)
            collection.extend(keep)

    @staticmethod
    def _jain(values: np.ndarray) -> float:
        if not np.any(values > 0):
            return 1.0
        return float(values.sum() ** 2 / (len(values) * np.square(values).sum()))

    def fairness(self) -> float:
        # Capacity-normalized service time, not equal raw job counts.
        return self._jain(self.work_by_node / self.cpu)

    def step(self, actions: np.ndarray, reward_component: str = "full"):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_agents,):
            raise ValueError("Expected one action per edge node")
        self._failure_transition()
        old_rejected = self.rejected
        self._dispatch(actions)
        good, late, spent = self._process_nodes()
        self._expire_waiting()
        new_rejected = self.rejected - old_rejected
        inequality = 1.0 - self.fairness()
        reward = 1.0 * good - 1.0 * late - 0.55 * new_rejected
        if reward_component not in {"no_fairness", "baseline"}:
            reward -= 0.10 * inequality
        if reward_component not in {"no_energy", "baseline"}:
            reward -= 0.005 * spent
        reward /= max(1, self.num_agents)
        self.t += 1
        done = self.t >= self.config.horizon
        if not done:
            self._generate_arrivals()
        return self.observations(), float(reward), done, self.metrics()

    def metrics(self) -> dict[str, float]:
        unfinished = self.generated - self.on_time - self.late - self.rejected
        assert unfinished >= 0, "Task accounting violation"
        return {
            "generated": float(self.generated),
            "dcr": 100.0 * self.on_time / max(1, self.generated),
            "mrt": float(np.mean(self.response_times)) if self.response_times else float("nan"),
            "jain": self.fairness(),
            "trr": 100.0 * self.rejected / max(1, self.generated),
            "unfinished": float(unfinished),
            "on_time": float(self.on_time),
            "late": float(self.late),
            "energy_proxy": self.energy_proxy,
        }


def heuristic_actions(env: IoTEdgeEnvironment, method: str, tick: int) -> np.ndarray:
    """Deterministic rule baselines using the same feasibility restrictions."""
    actions = np.zeros(env.num_agents, dtype=np.int64)
    mask = env.action_masks(constrained=True)
    for origin in range(env.num_agents):
        if not env.pending[origin]:
            continue
        possible = np.flatnonzero(mask[origin, 1:])
        if len(possible) == 0:
            continue
        if method == "greedy":
            job = env.pending[origin][0]
            best = min(possible, key=lambda j: (env.expected_finish(origin, int(j), job), int(j)))
        elif method == "round_robin":
            best = int(possible[np.searchsorted(possible, (tick + origin) % env.num_agents) % len(possible)])
        else:
            raise ValueError(method)
        actions[origin] = best + 1
    return actions
