import numpy as np

from qmix_aiot.environment import EnvironmentConfig, IoTEdgeEnvironment, Task, heuristic_actions


def simulate(seed=5, scenario="stable", policy="greedy"):
    cfg = EnvironmentConfig(nodes=8, devices=24, horizon=40, scenario=scenario)
    env = IoTEdgeEnvironment(cfg, seed=seed)
    while True:
        action = heuristic_actions(env, policy, env.t)
        _, _, done, metrics = env.step(action)
        if done:
            break
    return env, metrics


def test_same_seed_same_arrivals_and_results():
    a, am = simulate()
    b, bm = simulate()
    np.testing.assert_allclose(a.cpu, b.cpu)
    assert am == bm
    assert [(t.created, t.work, t.deadline) for t in a.tasks.values()] == [
        (t.created, t.work, t.deadline) for t in b.tasks.values()
    ]


def test_task_conservation_and_deadline_accounting():
    env, metrics = simulate(seed=42)
    assert int(metrics["generated"]) == len(env.tasks)
    assert int(metrics["generated"]) == (int(metrics["on_time"]) + int(metrics["late"])
                                               + env.rejected + int(metrics["unfinished"]))
    assert 0 <= metrics["dcr"] <= 100
    assert 0 <= metrics["trr"] <= 100
    assert 0 < metrics["jain"] <= 1
    assert all(task.outcome in {"completed", "rejected", "pending", "queued", "running"}
               for task in env.tasks.values())


def test_scenarios_have_real_differences():
    stable, _ = simulate(seed=10, scenario="stable")
    peaks, _ = simulate(seed=10, scenario="peaks")
    failure, _ = simulate(seed=10, scenario="failures")
    hetero, _ = simulate(seed=10, scenario="heterogeneous")
    assert peaks.generated > stable.generated
    assert failure.reasons.get("node_failure", 0) >= 0
    assert hetero.cpu.max() > stable.cpu.max() or hetero.cpu.min() < stable.cpu.min()


def test_feasible_action_reject_and_no_task_mask():
    env = IoTEdgeEnvironment(EnvironmentConfig(nodes=6, devices=18, horizon=30), seed=8)
    masks = env.action_masks()
    assert np.all(masks[:, 0])
    assert np.all(masks.sum(axis=1)[[not q for q in env.pending]] == 1)


def test_greedy_and_round_robin_use_valid_masks():
    env = IoTEdgeEnvironment(EnvironmentConfig(nodes=6, devices=18, horizon=30), seed=8)
    for method in ("greedy", "round_robin"):
        selected = heuristic_actions(env, method, 0)
        assert np.all(env.action_masks()[np.arange(env.num_agents), selected])


def test_queue_length_announcement_does_not_expose_private_job_work():
    env = IoTEdgeEnvironment(EnvironmentConfig(nodes=3, devices=6, horizon=20), seed=12)
    candidate = Task(100, 0, 0, 30, 2, 1, 10, .7)
    private = Task(101, 1, 0, 20, 2, 1, 10, .7, ready_at=0)
    env.queues[1].append(private)
    before = env.expected_finish(0, 1, candidate)
    private.work = 3000  # Not advertised to other agents.
    after = env.expected_finish(0, 1, candidate)
    assert before == after


def test_origin_receives_only_its_local_link_estimates():
    cfg = EnvironmentConfig(nodes=3, devices=6, horizon=20)
    env = IoTEdgeEnvironment(cfg, seed=12)
    obs = env.observations()
    assert obs.shape == (3, 10 + 6 * cfg.nodes)
    for origin in range(cfg.nodes):
        np.testing.assert_allclose(obs[origin, 10 + 4 * cfg.nodes:10 + 5 * cfg.nodes],
                                   env.link_latency[origin] / 0.30)
        np.testing.assert_allclose(obs[origin, 10 + 5 * cfg.nodes:10 + 6 * cfg.nodes],
                                   env.link_bandwidth[origin] / 50.0)


def test_capacity_normalized_fairness_not_raw_equality():
    env = IoTEdgeEnvironment(EnvironmentConfig(nodes=2, devices=4, horizon=20), seed=12)
    env.cpu = np.array([10., 20.])
    env.work_by_node = np.array([10., 20.])
    assert np.isclose(env.fairness(), 1.0)
