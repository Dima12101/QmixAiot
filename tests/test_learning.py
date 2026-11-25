import numpy as np
import torch

from qmix_aiot.environment import EnvironmentConfig, IoTEdgeEnvironment
from qmix_aiot.learning import Learner, QMIXMixer, generate_episode


def test_mixer_is_monotonic_in_agent_q():
    mixer = QMIXMixer(n_agents=3, state_dim=17)
    q = torch.randn(2, 5, 3)
    state = torch.randn(2, 5, 17)
    baseline = mixer(q, state)
    perturbed = mixer(q + 0.1, state)
    assert torch.all(perturbed >= baseline - 1e-6)


def test_recurrent_hidden_and_full_episode_update():
    cfg = EnvironmentConfig(nodes=3, devices=6, horizon=8)
    learner = Learner(cfg, seed=4)
    history = []
    for seed in range(3):
        env = IoTEdgeEnvironment(cfg, seed=seed)
        episode, _ = generate_episode(env, learner, epsilon=0.5)
        learner.replay.append(episode)
        history.append(episode)
    assert history[0].observations.shape == (9, 3, 28)
    assert history[0].masks.shape == (9, 3, 4)
    loss = learner.train_step()
    assert loss is not None and np.isfinite(loss)


def test_iql_and_feed_forward_train():
    cfg = EnvironmentConfig(nodes=3, devices=6, horizon=8)
    for method in ("iql", "no_gru", "no_fairness", "no_energy", "no_public"):
        learner = Learner(cfg, seed=19, method=method)
        for seed in range(3):
            ep, _ = generate_episode(IoTEdgeEnvironment(cfg, seed=seed), learner, epsilon=.25)
            learner.replay.append(ep)
        assert np.isfinite(learner.train_step())


def test_work_conserving_mask_forbids_only_avoidable_rejection():
    cfg = EnvironmentConfig(nodes=4, devices=12, horizon=8)
    env = IoTEdgeEnvironment(cfg, seed=8)
    model = Learner(cfg, seed=8, method="work_conserving_qmix")
    mask = model.action_mask(env)
    assert np.array_equal(mask[:, 0], ~mask[:, 1:].any(axis=1))
    model.reset_hidden()
    actions = model.choose(env.observations(), mask, epsilon=1)
    assert np.all(mask[np.arange(cfg.nodes), actions])


def test_targetwise_qmix_replays_and_backpropagates():
    cfg = EnvironmentConfig(nodes=4, devices=12, horizon=8)
    for method in ("structured_qmix", "structured_no_public", "structured_optional_reject"):
        learner = Learner(cfg, seed=19, method=method)
        for seed in range(3):
            ep, _ = generate_episode(IoTEdgeEnvironment(cfg, seed=seed), learner, epsilon=.25)
            assert ep.observations.shape == (9, 4, 34)
            assert ep.masks.shape == (9, 4, 5)
            learner.replay.append(ep)
        assert np.isfinite(learner.train_step())


def test_candidatewise_parameter_count_is_smaller_than_flat_output_network():
    cfg = EnvironmentConfig(nodes=25, devices=50, horizon=8)
    flat = Learner(cfg, seed=0, method="work_conserving_qmix")
    structured = Learner(cfg, seed=0, method="structured_qmix")
    assert sum(p.numel() for p in flat.agents.parameters()) == 41114
    assert sum(p.numel() for p in structured.agents.parameters()) == 19825
    assert sum(p.numel() for p in flat.mixer.parameters()) == 114721
    assert sum(p.numel() for p in structured.mixer.parameters()) == 114721
