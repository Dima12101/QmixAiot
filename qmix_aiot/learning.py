"""Episode-replay recurrent QMIX, feed-forward ablation and independent DQN.

This implementation deliberately does not claim an optimization guarantee.
Monotonicity constrains the mixing function, not training convergence.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .environment import EnvironmentConfig, IoTEdgeEnvironment


class AgentNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64, recurrent: bool = True):
        super().__init__()
        self.recurrent = recurrent
        self.hidden_dim = hidden_dim
        self.input_layer = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
        if recurrent:
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        else:
            self.gru = None
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_actions))

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor | None = None):
        """Accept B,K,T,F; return B,K,T,A and B,K,H hidden state."""
        b, k, t, f = obs.shape
        x = self.input_layer(obs.reshape(b * k, t, f))
        if self.gru is not None:
            h0 = None if hidden is None else hidden.reshape(1, b * k, self.hidden_dim)
            x, h = self.gru(x, h0)
            hidden_next = h.reshape(b, k, self.hidden_dim)
        else:
            hidden_next = x[:, -1, :].reshape(b, k, self.hidden_dim)
        q = self.output_layer(x).reshape(b, k, t, -1)
        return q, hidden_next


class TargetwiseAgentNetwork(nn.Module):
    """Parameter-shared target scorer with task history and resource pairs."""

    def __init__(self, n_nodes: int, hidden_dim: int = 48):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.local_encoder = nn.Sequential(nn.Linear(10, hidden_dim), nn.ReLU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.target_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.reject_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor | None = None):
        b, k, t, features = obs.shape
        n = self.n_nodes
        if k != n or features != 10 + 6 * n:
            raise ValueError("Expected one agent and advertised feature block per node")
        local = obs[..., :10]
        embedded = self.local_encoder(local.reshape(b * k, t, 10))
        first = None if hidden is None else hidden.reshape(1, b * k, self.hidden_dim)
        context, next_hidden = self.gru(embedded, first)
        context = context.reshape(b, k, t, self.hidden_dim)
        public = obs[..., 10:10 + 4 * n].reshape(b, k, t, n, 4)
        latency = obs[..., 10 + 4 * n:10 + 5 * n].unsqueeze(-1)
        bandwidth = obs[..., 10 + 5 * n:10 + 6 * n].unsqueeze(-1)
        destination = torch.arange(n, device=obs.device, dtype=obs.dtype).view(1, 1, 1, n, 1)
        source = (local[..., 8:9] * (n - 1)).round().unsqueeze(-1)
        phase = torch.remainder(destination - source, n) / n
        destination_scaled = destination.expand(b, k, t, n, 1) / max(1, n - 1)
        target = torch.cat([public, latency, bandwidth, destination_scaled, phase], dim=-1)
        target_embed = self.target_encoder(target)
        agent_query = self.query(context).unsqueeze(-2)
        scores = (agent_query * target_embed).sum(-1) / (self.hidden_dim ** 0.5)
        reject = self.reject_head(context)
        return torch.cat([reject, scores], dim=-1), next_hidden.reshape(b, k, self.hidden_dim)


class QMIXMixer(nn.Module):
    """Two-layer state-conditioned nonnegative mixer of Rashid et al."""

    def __init__(self, n_agents: int, state_dim: int, mix_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.mix_dim = mix_dim
        self.hyper_w1 = nn.Linear(state_dim, n_agents * mix_dim)
        self.hyper_b1 = nn.Linear(state_dim, mix_dim)
        self.hyper_w2 = nn.Linear(state_dim, mix_dim)
        self.state_value = nn.Sequential(nn.Linear(state_dim, mix_dim), nn.ReLU(), nn.Linear(mix_dim, 1))

    def forward(self, agent_q: torch.Tensor, state: torch.Tensor):
        """agent_q B,T,K; state B,T,S -> joint values B,T."""
        b, t, k = agent_q.shape
        if k != self.n_agents:
            raise ValueError("Agent dimension mismatch")
        s = state.reshape(b * t, -1)
        first = self.hyper_w1(s).abs().reshape(b * t, k, self.mix_dim)
        bias = self.hyper_b1(s).reshape(b * t, 1, self.mix_dim)
        hidden = F.elu(torch.bmm(agent_q.reshape(b * t, 1, k), first) + bias)
        second = self.hyper_w2(s).abs().reshape(b * t, self.mix_dim, 1)
        combined = torch.bmm(hidden, second).reshape(b, t)
        return combined + self.state_value(s).reshape(b, t)


@dataclass
class Episode:
    observations: np.ndarray
    states: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray


class Learner:
    def __init__(
        self,
        config: EnvironmentConfig,
        seed: int = 0,
        method: str = "adapted_qmix",
        lr: float = .0004,
        gamma: float = .98,
    ):
        if method not in {"adapted_qmix", "vanilla_qmix", "iql", "no_gru", "no_masks", "no_fairness", "no_energy", "no_public", "work_conserving_qmix", "structured_qmix", "structured_no_public", "structured_optional_reject"}:
            raise ValueError(method)
        self.method = method
        self.config = config
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(int(seed))
        torch.set_num_threads(min(2, torch.get_num_threads()))
        n, a = config.nodes, config.nodes + 1
        dims = IoTEdgeEnvironment(config, seed).obs_dim
        state_dim = 5 * n + 2
        self.agents = (
            TargetwiseAgentNetwork(n) if method.startswith("structured_")
            else AgentNetwork(dims, a, recurrent=method != "no_gru")
        )
        self.target_agents = deepcopy(self.agents)
        self.mixer = None if method == "iql" else QMIXMixer(n, state_dim)
        self.target_mixer = None if self.mixer is None else deepcopy(self.mixer)
        params = list(self.agents.parameters())
        if self.mixer is not None:
            params.extend(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.replay: deque[Episode] = deque(maxlen=80)
        self.hidden: torch.Tensor | None = None
        self.updates = 0

    @property
    def uses_masks(self) -> bool:
        # Legal-action enforcement is part of the common environment for all
        # main baselines. Mask removal is examined ONLY in the named ablation.
        return self.method != "no_masks"

    @property
    def work_conserving(self) -> bool:
        return self.method in {"work_conserving_qmix", "structured_qmix", "structured_no_public"}

    def action_mask(self, environment: IoTEdgeEnvironment) -> np.ndarray:
        mask = environment.action_masks(constrained=self.uses_masks)
        if self.work_conserving:
            mask[mask[:, 1:].any(axis=1), 0] = False
        return mask

    @property
    def reward_component(self) -> str:
        if self.method == "vanilla_qmix":
            return "baseline"
        return self.method if self.method in {"no_fairness", "no_energy"} else "full"

    def prepare_observation(self, obs: np.ndarray) -> np.ndarray:
        # Standard QMIX baseline: local resource and task summaries only;
        # the adapted model also uses announced neighboring load/capability.
        if self.method in {"vanilla_qmix", "no_public", "structured_no_public"}:
            obs = obs.copy()
            # Ablation: remove announced state of other nodes, retaining the
            # originating node's own outgoing network estimates.
            obs[:, 10:10 + 4 * self.config.nodes] = 0.0
        return obs

    def reset_hidden(self):
        self.hidden = None

    def choose(self, obs: np.ndarray, mask: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        obs = self.prepare_observation(obs)
        with torch.inference_mode():
            q, self.hidden = self.agents(
                torch.from_numpy(obs).unsqueeze(0).unsqueeze(2), self.hidden
            )
            q = q[0, :, 0, :].numpy()
        q[~mask] = -1e9
        actions = q.argmax(axis=-1).astype(np.int64)
        explore = self.rng.random(len(actions)) < epsilon
        for i in np.flatnonzero(explore):
            actions[i] = int(self.rng.choice(np.flatnonzero(mask[i])))
        return actions

    def train_step(self, batch_size: int = 3):
        if len(self.replay) < batch_size:
            return None
        episodes = [self.replay[int(i)] for i in self.rng.choice(len(self.replay), batch_size, replace=False)]
        obs = torch.as_tensor(np.stack([e.observations for e in episodes]), dtype=torch.float32)
        state = torch.as_tensor(np.stack([e.states for e in episodes]), dtype=torch.float32)
        masks = torch.as_tensor(np.stack([e.masks for e in episodes]), dtype=torch.bool)
        actions = torch.as_tensor(np.stack([e.actions for e in episodes]), dtype=torch.int64)
        rewards = torch.as_tensor(np.stack([e.rewards for e in episodes]), dtype=torch.float32)
        dones = torch.as_tensor(np.stack([e.dones for e in episodes]), dtype=torch.float32)
        # B,T+1,K,F -> B,K,T+1,F; recurrent hidden states are carried along episodes.
        obs_seq = obs.permute(0, 2, 1, 3)
        online_q, _ = self.agents(obs_seq)
        b, k, _, a = online_q.shape
        now = online_q[:, :, :-1, :].gather(3, actions.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)
        now = now.permute(0, 2, 1)
        with torch.no_grad():
            next_online = online_q[:, :, 1:, :].detach()
            next_masks = masks[:, 1:, :, :].permute(0, 2, 1, 3)
            next_a = next_online.masked_fill(~next_masks, -1e9).argmax(dim=-1, keepdim=True)
            target_q, _ = self.target_agents(obs_seq)
            next_values = target_q[:, :, 1:, :].gather(3, next_a).squeeze(-1).permute(0, 2, 1)
        if self.mixer is not None:
            assert self.target_mixer is not None
            prediction = self.mixer(now, state[:, :-1])
            with torch.no_grad():
                next_joint = self.target_mixer(next_values, state[:, 1:])
                target = rewards + self.gamma * next_joint * (1.0 - dones)
        else:
            prediction = now
            target = rewards.unsqueeze(-1) + self.gamma * next_values * (1.0 - dones.unsqueeze(-1))
        loss = F.smooth_l1_loss(prediction, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = list(self.agents.parameters()) + (list(self.mixer.parameters()) if self.mixer is not None else [])
        nn.utils.clip_grad_norm_(params, 10.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % 20 == 0:
            self.target_agents.load_state_dict(self.agents.state_dict())
            if self.mixer is not None:
                self.target_mixer.load_state_dict(self.mixer.state_dict())
        return float(loss.item())

    def save(self, path: str):
        torch.save({"agent": self.agents.state_dict(), "mixer": None if self.mixer is None else self.mixer.state_dict(),
                    "method": self.method, "config": vars(self.config)}, path)

    def load(self, path: str):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["method"] != self.method:
            raise ValueError("Mismatched method")
        self.agents.load_state_dict(data["agent"])
        if self.mixer is not None and data["mixer"] is not None:
            self.mixer.load_state_dict(data["mixer"])


def generate_episode(env: IoTEdgeEnvironment, learner: Learner, epsilon: float) -> tuple[Episode, dict[str, float]]:
    learner.reset_hidden()
    obs = env.observations()
    observations, states, masks, actions, rewards, dones = [learner.prepare_observation(obs)], [env.state()], [], [], [], []
    while True:
        mask = learner.action_mask(env)
        masks.append(mask)
        action = learner.choose(obs, mask, epsilon)
        actions.append(action)
        obs, reward, done, _ = env.step(action, reward_component=learner.reward_component)
        observations.append(learner.prepare_observation(obs))
        states.append(env.state())
        rewards.append(reward)
        dones.append(done)
        if done:
            masks.append(learner.action_mask(env))
            break
    episode = Episode(
        observations=np.stack(observations),
        states=np.stack(states),
        masks=np.stack(masks),
        actions=np.stack(actions),
        rewards=np.array(rewards, dtype=np.float32),
        dones=np.array(dones, dtype=np.float32),
    )
    return episode, env.metrics()


def train(config: EnvironmentConfig, method: str, seed: int, episodes: int = 70):
    learner = Learner(config, seed=seed, method=method)
    history = []
    for epoch in range(episodes):
        env = IoTEdgeEnvironment(config, seed=seed * 100000 + epoch)
        eps = max(.05, .85 * (1.0 - epoch / max(1, .8 * episodes)))
        trial, metrics = generate_episode(env, learner, eps)
        learner.replay.append(trial)
        losses = [learner.train_step() for _ in range(3)]
        history.append({"episode": epoch, "epsilon": float(eps), "reward": float(trial.rewards.sum()),
                        "loss": next((x for x in reversed(losses) if x is not None), float("nan")), **metrics})
    return learner, history
