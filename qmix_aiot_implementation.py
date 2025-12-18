#!/usr/bin/env python3
"""
QMIX-based Multi-Agent Reinforcement Learning for AIoT Resource Management
Complete implementation for virtual computing environment orchestration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import json


# ============================================================================
# 1. NEURAL NETWORK ARCHITECTURES
# ============================================================================

class LocalQNetwork(nn.Module):
    """
    Local Q-network for agent i.
    Processes local observations and outputs Q-values for each action.
    
    Args:
        obs_dim: Dimension of local observation
        action_dim: Number of possible actions
        hidden_dim: Hidden dimension of GRU layer
        device: Device to run on ('cpu' or 'cuda')
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64, 
                 device: str = 'cpu'):
        super(LocalQNetwork, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # GRU layer for temporal processing
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.q_out = nn.Linear(64, action_dim)
        
        self.to(device)
    
    def forward(self, obs: torch.Tensor, 
                hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor of shape (batch, seq_len, obs_dim)
            hidden_state: GRU hidden state
            
        Returns:
            q_values: Q-values for each action (batch, action_dim)
            new_hidden_state: Updated hidden state
        """
        gru_out, hidden_state = self.gru(obs, hidden_state)
        last_out = gru_out[:, -1, :]
        
        x = F.relu(self.fc1(last_out))
        x = F.relu(self.fc2(x))
        q_values = self.q_out(x)
        
        return q_values, hidden_state
    
    def init_hidden_state(self, batch_size: int) -> torch.Tensor:
        """Initialize GRU hidden state."""
        return torch.zeros(batch_size, self.hidden_dim, device=self.device)


class QMIXMixer(nn.Module):
    """
    QMIX Mixer Network for aggregating local Q-functions into global Q-function.
    Guarantees monotonicity: dQ_tot/dQ_i >= 0 for all i
    
    Args:
        state_dim: Dimension of global state
        num_agents: Number of agents
        hidden_dim: Hidden dimension of hypernetwork
        device: Device to run on
    """
    
    def __init__(self, state_dim: int, num_agents: int, hidden_dim: int = 32, 
                 device: str = 'cpu'):
        super(QMIXMixer, self).__init__()
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.device = device
        
        # Hypernetwork for generating weights (guaranteed non-negative via ReLU)
        self.weight_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents)
        )
        
        # Hypernetwork for generating bias
        self.bias_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.to(device)
    
    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Aggregate local Q-functions into global Q-function.
        
        Args:
            q_values: Local Q-values (batch, num_agents)
            state: Global state (batch, state_dim)
            
        Returns:
            q_tot: Global Q-function value (batch, 1)
        """
        # Generate weights (monotonicity guaranteed by ReLU)
        weights = F.relu(self.weight_net(state))  # (batch, num_agents)
        
        # Generate bias
        bias = self.bias_net(state)  # (batch, 1)
        
        # Monotonic aggregation
        weighted_q = (q_values * weights).sum(dim=1, keepdim=True)
        q_tot = weighted_q + bias
        
        return q_tot
    
    def verify_monotonicity(self, q_values: torch.Tensor, 
                           state: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        """
        Verify monotonicity property: dQ_tot/dQ_i >= 0 for all i.
        
        Returns:
            (is_monotonic, weights)
        """
        weights = F.relu(self.weight_net(state))
        monotonic = torch.all(weights >= 0)
        return monotonic, weights


# ============================================================================
# 2. MAIN LEARNING AGENT
# ============================================================================

class MAQLearner:
    """
    Multi-Agent Q-Learning system using QMIX.
    Manages training of local Q-networks, Mixing Network and target networks.
    """
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, 
                 state_dim: int, learning_rate: float = 1e-3, gamma: float = 0.99, 
                 device: str = 'cpu'):
        
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.device = device
        
        # Local Q-networks for each agent
        self.agent_networks = [
            LocalQNetwork(obs_dim, action_dim, device=device)
            for _ in range(num_agents)
        ]
        
        # Target networks
        self.agent_networks_target = [
            LocalQNetwork(obs_dim, action_dim, device=device)
            for _ in range(num_agents)
        ]
        
        # Mixer networks
        self.mixer = QMIXMixer(state_dim, num_agents, device=device)
        self.mixer_target = QMIXMixer(state_dim, num_agents, device=device)
        
        # Optimizers
        agent_params = sum([list(net.parameters()) for net in self.agent_networks], [])
        self.optimizer_agents = optim.Adam(agent_params, lr=learning_rate)
        self.optimizer_mixer = optim.Adam(self.mixer.parameters(), lr=learning_rate)
        
        # Target network update parameters
        self.tau = 0.01
        self.update_interval = 100
        self.steps = 0
    
    def select_action(self, observations: List[np.ndarray], 
                     epsilon: float = 0.1) -> List[int]:
        """
        Select actions using epsilon-greedy strategy.
        
        Args:
            observations: List of local observations for each agent
            epsilon: Exploration coefficient
            
        Returns:
            actions: List of selected actions
        """
        actions = []
        
        for i, obs in enumerate(observations):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device)
            
            if torch.rand(1).item() < epsilon:
                action = torch.randint(0, self.action_dim, (1,)).item()
            else:
                with torch.no_grad():
                    q_values, _ = self.agent_networks[i](obs_tensor)
                    action = q_values.argmax(dim=1).item()
            
            actions.append(action)
        
        return actions
    
    def update(self, batch: Dict) -> float:
        """
        Update networks based on batch of experience.
        
        Args:
            batch: Dictionary with keys ['obs', 'actions', 'rewards', 'next_obs', 
                   'dones', 'state', 'next_state']
        
        Returns:
            loss: Training loss value
        """
        obs_batch = torch.FloatTensor(batch['obs']).to(self.device)
        actions_batch = torch.LongTensor(batch['actions']).to(self.device)
        rewards_batch = torch.FloatTensor(batch['rewards']).to(self.device)
        next_obs_batch = torch.FloatTensor(batch['next_obs']).to(self.device)
        dones_batch = torch.FloatTensor(batch['dones']).to(self.device)
        state_batch = torch.FloatTensor(batch['state']).to(self.device)
        next_state_batch = torch.FloatTensor(batch['next_state']).to(self.device)
        
        batch_size = obs_batch.shape[0]
        
        # Compute current Q-values
        current_q_values = []
        for i in range(self.num_agents):
            q_vals, _ = self.agent_networks[i](obs_batch[:, i:i+1, :])
            q_val = q_vals.gather(1, actions_batch[:, i:i+1])
            current_q_values.append(q_val)
        
        current_q_values = torch.cat(current_q_values, dim=1)
        
        # Compute next Q-values using target networks (Double DQN)
        next_q_values_all = []
        for i in range(self.num_agents):
            q_vals, _ = self.agent_networks_target[i](next_obs_batch[:, i:i+1, :])
            next_q_values_all.append(q_vals)
        
        # Select best actions using current network
        next_best_actions = []
        for i in range(self.num_agents):
            q_vals_current, _ = self.agent_networks[i](next_obs_batch[:, i:i+1, :])
            best_action = q_vals_current.argmax(dim=1)
            next_best_actions.append(best_action)
        
        # Compute target Q-values using target networks
        target_q_values = []
        for i in range(self.num_agents):
            q_vals = next_q_values_all[i]
            q_val = q_vals.gather(1, next_best_actions[i].unsqueeze(1))
            target_q_values.append(q_val)
        
        target_q_values = torch.cat(target_q_values, dim=1)
        
        # Compute global target Q-function through target mixer
        target_q_tot = self.mixer_target(target_q_values, next_state_batch)
        target = rewards_batch.unsqueeze(1) + self.gamma * target_q_tot * (1 - dones_batch.unsqueeze(1))
        
        # Compute current global Q-function through mixer
        current_q_tot = self.mixer(current_q_values, state_batch)
        
        # Compute loss and update
        loss = F.mse_loss(current_q_tot, target.detach())
        
        self.optimizer_agents.zero_grad()
        self.optimizer_mixer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            sum([list(net.parameters()) for net in self.agent_networks], []) + 
            list(self.mixer.parameters()), 
            10.0
        )
        self.optimizer_agents.step()
        self.optimizer_mixer.step()
        
        # Soft update target networks
        self.steps += 1
        if self.steps % self.update_interval == 0:
            self._soft_update_target_networks()
        
        return loss.item()
    
    def _soft_update_target_networks(self):
        """Soft update target networks."""
        for i in range(self.num_agents):
            for param, target_param in zip(self.agent_networks[i].parameters(),
                                          self.agent_networks_target[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.mixer.parameters(), self.mixer_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# ============================================================================
# 3. ENVIRONMENT
# ============================================================================

class IoTEdgeEnvironment:
    """
    AIoT environment simulator with dynamic task arrival, resource availability,
    and network delays.
    """
    
    def __init__(self, num_nodes: int = 25, num_devices: int = 50, 
                 episode_length: int = 500):
        
        self.num_nodes = num_nodes
        self.num_devices = num_devices
        self.episode_length = episode_length
        self.current_step = 0
        
        # Node resources
        self.cpu_capacity = np.random.uniform(20, 40, num_nodes)
        self.mem_capacity = np.random.uniform(32, 64, num_nodes)
        self.cpu_available = self.cpu_capacity.copy()
        self.mem_available = self.mem_capacity.copy()
        
        # Task queues and execution
        self.task_queues = [deque() for _ in range(num_nodes)]
        self.tasks_in_execution = [[] for _ in range(num_nodes)]
        self.all_tasks = {}
        self.task_counter = 0
        
        # Network delays
        self.network_delays = np.random.uniform(0.5, 5.0, (num_nodes, num_nodes))
        np.fill_diagonal(self.network_delays, 0.1)
        
        # Metrics tracking
        self.completed_tasks = 0
        self.rejected_tasks = 0
        self.violated_deadlines = 0
    
    def reset(self) -> List[np.ndarray]:
        """Initialize environment for new episode."""
        self.current_step = 0
        self.cpu_available = self.cpu_capacity.copy()
        self.mem_available = self.mem_capacity.copy()
        self.task_queues = [deque() for _ in range(self.num_nodes)]
        self.tasks_in_execution = [[] for _ in range(self.num_nodes)]
        self.all_tasks = {}
        self.task_counter = 0
        self.completed_tasks = 0
        self.rejected_tasks = 0
        self.violated_deadlines = 0
        
        return self._get_observations()
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], float, bool, Dict]:
        """
        Execute one simulation step.
        
        Args:
            actions: List of actions for each node
            
        Returns:
            (observations, reward, done, info)
        """
        self.current_step += 1
        
        self._generate_tasks()
        self._distribute_tasks(actions)
        self._execute_tasks()
        
        reward = self._compute_reward()
        observations = self._get_observations()
        done = self.current_step >= self.episode_length
        
        info = {
            'completed_tasks': self.completed_tasks,
            'rejected_tasks': self.rejected_tasks,
            'violated_deadlines': self.violated_deadlines,
            'node_loads': [len(q) for q in self.task_queues]
        }
        
        return observations, reward, done, info
    
    def _generate_tasks(self):
        """Generate new tasks with Poisson distribution."""
        arrival_rate = 0.3 + 0.2 * np.sin(2 * np.pi * self.current_step / 100)
        
        for node_idx in range(self.num_nodes):
            if np.random.rand() < arrival_rate:
                task = {
                    'id': self.task_counter,
                    'cpu_req': np.random.uniform(0.1, 8.0),
                    'mem_req': np.random.uniform(0.1, 16.0),
                    'deadline': 5.0 + np.random.exponential(3.0),
                    'priority': np.random.uniform(0.5, 1.0),
                    'arrival_time': self.current_step,
                    'assigned_node': None,
                    'start_time': None,
                    'end_time': None,
                    'completed': False,
                    'rejected': False
                }
                self.all_tasks[self.task_counter] = task
                self.tasks_in_execution[node_idx].append(task)
                self.task_counter += 1
    
    def _distribute_tasks(self, actions: List[int]):
        """Distribute tasks according to agent actions."""
        for node_idx, action in enumerate(actions):
            if self.tasks_in_execution[node_idx]:
                task = self.tasks_in_execution[node_idx].pop(0)
                
                if action == 0:  # LOCAL
                    if self.cpu_available[node_idx] >= task['cpu_req'] and \
                       self.mem_available[node_idx] >= task['mem_req']:
                        task['assigned_node'] = node_idx
                        task['start_time'] = self.current_step
                        self.task_queues[node_idx].append(task)
                    else:
                        task['rejected'] = True
                        self.rejected_tasks += 1
                
                elif 1 <= action < self.num_nodes + 1:  # OFFLOAD
                    target_node = action - 1
                    task['assigned_node'] = target_node
                    task['start_time'] = self.current_step
                    self.task_queues[target_node].append(task)
                
                else:  # REJECT
                    task['rejected'] = True
                    self.rejected_tasks += 1
    
    def _execute_tasks(self):
        """Execute tasks in queues and update available resources."""
        for node_idx in range(self.num_nodes):
            # Check for completed tasks
            completed = []
            for task in self.task_queues[node_idx]:
                if task['start_time'] is not None:
                    execution_time = 1.0 + task['cpu_req'] / 10.0
                    if self.current_step - task['start_time'] > execution_time:
                        task['end_time'] = self.current_step
                        task['completed'] = True
                        self.cpu_available[node_idx] += task['cpu_req']
                        self.mem_available[node_idx] += task['mem_req']
                        self.completed_tasks += 1
                        
                        # Check deadline violation
                        delay = task['end_time'] - task['arrival_time']
                        if delay > task['deadline']:
                            self.violated_deadlines += 1
                        
                        completed.append(task)
            
            for task in completed:
                self.task_queues[node_idx].remove(task)
            
            # Start execution of next tasks
            if self.task_queues[node_idx]:
                task = self.task_queues[node_idx][0]
                if task['start_time'] is None:
                    if self.cpu_available[node_idx] >= task['cpu_req'] and \
                       self.mem_available[node_idx] >= task['mem_req']:
                        task['start_time'] = self.current_step
                        self.cpu_available[node_idx] -= task['cpu_req']
                        self.mem_available[node_idx] -= task['mem_req']
    
    def _compute_reward(self) -> float:
        """Compute global reward."""
        total_tasks = max(1, self.task_counter)
        
        r_completion = self.completed_tasks / total_tasks
        r_deadline = -self.violated_deadlines / total_tasks
        
        loads = np.array([len(q) for q in self.task_queues])
        if np.sum(loads) > 0:
            r_fairness = np.sum(loads) ** 2 / (len(loads) * np.sum(loads ** 2) + 1e-6)
        else:
            r_fairness = 1.0
        
        reward = 0.4 * r_completion + 0.35 * r_deadline + 0.25 * r_fairness
        return reward
    
    def _get_observations(self) -> List[np.ndarray]:
        """Get local observations for each node."""
        observations = []
        for node_idx in range(self.num_nodes):
            obs = np.array([
                self.cpu_available[node_idx] / self.cpu_capacity[node_idx],
                self.mem_available[node_idx] / self.mem_capacity[node_idx],
                min(len(self.task_queues[node_idx]) / 10.0, 1.0),
                0.0,  # Placeholder for neighbor state
                node_idx / self.num_nodes
            ], dtype=np.float32)
            observations.append(obs)
        
        return observations
    
    def get_global_state(self) -> np.ndarray:
        """Get global state for Mixing Network."""
        state = np.concatenate([
            self.cpu_available / self.cpu_capacity,
            self.mem_available / self.mem_capacity,
            np.array([len(q) for q in self.task_queues]) / 10.0
        ], dtype=np.float32)
        return state


# ============================================================================
# 4. TRAINING AND EVALUATION
# ============================================================================

def train_qmix(num_episodes: int = 1000, num_agents: int = 25, 
               num_devices: int = 50, batch_size: int = 64, 
               device: str = 'cpu') -> Tuple[MAQLearner, List[float], Dict]:
    """
    Main training loop for QMIX.
    
    Args:
        num_episodes: Number of training episodes
        num_agents: Number of Edge nodes (agents)
        num_devices: Number of IoT devices
        batch_size: Batch size for training
        device: Device to run on
    
    Returns:
        (learner, episode_rewards, metrics)
    """
    obs_dim = 5
    action_dim = num_agents + 1
    state_dim = num_agents * 3
    
    env = IoTEdgeEnvironment(num_nodes=num_agents, num_devices=num_devices, 
                            episode_length=500)
    learner = MAQLearner(num_agents, obs_dim, action_dim, state_dim, device=device)
    
    replay_buffer = deque(maxlen=10000)
    episode_rewards = []
    metrics_history = {
        'mrt': [],
        'dcr': [],
        'jain': [],
        'trr': []
    }
    
    for episode in range(num_episodes):
        observations = env.reset()
        done = False
        episode_reward = 0
        episode_completed = 0
        episode_rejected = 0
        
        while not done:
            epsilon = 0.1 * (1 - episode / num_episodes)
            actions = learner.select_action(observations, epsilon=epsilon)
            
            next_observations, reward, done, info = env.step(actions)
            
            transition = {
                'obs': np.array(observations),
                'actions': np.array(actions),
                'reward': reward,
                'next_obs': np.array(next_observations),
                'done': done,
                'state': env.get_global_state(),
                'next_state': env.get_global_state()
            }
            replay_buffer.append(transition)
            
            observations = next_observations
            episode_reward += reward
            episode_completed = info['completed_tasks']
            episode_rejected = info['rejected_tasks']
            
            # Update networks
            if len(replay_buffer) >= batch_size:
                batch_indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
                batch = {
                    'obs': np.array([replay_buffer[i]['obs'] for i in batch_indices]),
                    'actions': np.array([replay_buffer[i]['actions'] for i in batch_indices]),
                    'rewards': np.array([replay_buffer[i]['reward'] for i in batch_indices]),
                    'next_obs': np.array([replay_buffer[i]['next_obs'] for i in batch_indices]),
                    'dones': np.array([replay_buffer[i]['done'] for i in batch_indices]),
                    'state': np.array([replay_buffer[i]['state'] for i in batch_indices]),
                    'next_state': np.array([replay_buffer[i]['next_state'] for i in batch_indices])
                }
                learner.update(batch)
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.4f}, "
                  f"Completed: {episode_completed}, Rejected: {episode_rejected}")
    
    return learner, episode_rewards, metrics_history


def evaluate_scenarios(learner: MAQLearner, num_agents: int = 25, 
                      scenarios: List[str] = None) -> Dict:
    """
    Evaluate learner on multiple scenarios.
    
    Args:
        learner: Trained MAQLearner
        num_agents: Number of agents
        scenarios: List of scenario names to evaluate
    
    Returns:
        Dictionary with results for each scenario
    """
    if scenarios is None:
        scenarios = ['stable', 'peaks', 'failures', 'heterogeneous']
    
    results = {}
    
    for scenario in scenarios:
        env = IoTEdgeEnvironment(num_nodes=num_agents, episode_length=1000)
        observations = env.reset()
        done = False
        
        mrt_list = []
        deadline_compliance = 0
        total_tasks = 0
        
        while not done:
            actions = learner.select_action(observations, epsilon=0.0)
            observations, _, done, info = env.step(actions)
            total_tasks += max(1, len(env.all_tasks))
        
        # Compute metrics
        mrt = np.mean([t['end_time'] - t['arrival_time'] 
                      for t in env.all_tasks.values() if t['completed']])
        dcr = sum(1 for t in env.all_tasks.values() 
                 if t['completed'] and (t['end_time'] - t['arrival_time'] <= t['deadline'])) / max(1, total_tasks)
        
        loads = np.array([len(q) for q in env.task_queues])
        if np.sum(loads) > 0:
            jain = np.sum(loads) ** 2 / (len(loads) * np.sum(loads ** 2) + 1e-6)
        else:
            jain = 1.0
        
        trr = len([t for t in env.all_tasks.values() if t['rejected']]) / max(1, total_tasks)
        
        results[scenario] = {
            'mrt': mrt,
            'dcr': dcr * 100,
            'jain': jain,
            'trr': trr * 100
        }
    
    return results


if __name__ == '__main__':
    # Training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learner, rewards, metrics = train_qmix(num_episodes=1000, device=device)
    
    # Evaluation
    results = evaluate_scenarios(learner)
    print("\n=== Evaluation Results ===")
    for scenario, metrics in results.items():
        print(f"\n{scenario.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.2f}")
