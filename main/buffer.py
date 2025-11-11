"""
Rollout buffer for storing and processing experience for MAPPO.
"""

import torch
import numpy as np


class RolloutBuffer:
    """Buffer for storing rollout data for on-policy MAPPO training."""
    
    def __init__(self, n_agents, obs_dim, state_dim, buffer_size, device='cpu'):
        """
        Initialize rollout buffer.
        
        Args:
            n_agents: Number of agents (20)
            obs_dim: Observation dimension per agent (15)
            state_dim: Full state dimension (140)
            buffer_size: Maximum buffer size (e.g., 2048 steps)
            device: PyTorch device
        """
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.buffer_size = buffer_size
        self.device = device
        
        # Storage arrays
        self.observations = torch.zeros((buffer_size, n_agents, obs_dim), device=device)
        self.states = torch.zeros((buffer_size, state_dim), device=device)
        self.actions = torch.zeros((buffer_size, n_agents, 1), device=device)
        self.log_probs = torch.zeros((buffer_size, n_agents, 1), device=device)
        self.rewards = torch.zeros((buffer_size, 1), device=device)
        self.values = torch.zeros((buffer_size, 1), device=device)
        self.dones = torch.zeros((buffer_size, 1), device=device)
        
        # Computed during buffer finalization
        self.advantages = torch.zeros((buffer_size, 1), device=device)
        self.returns = torch.zeros((buffer_size, 1), device=device)
        
        self.ptr = 0
        self.full = False
    
    def add(self, obs, state, actions, log_probs, reward, value, done):
        """
        Add a transition to the buffer.
        
        Args:
            obs: Observations [n_agents, obs_dim]
            state: Full state [state_dim]
            actions: Actions [n_agents, 1] or [n_agents]
            log_probs: Log probabilities [n_agents, 1] or [n_agents]
            reward: Shared reward (scalar)
            value: Value estimate (scalar)
            done: Done flag (bool)
        """
        # Ensure correct shapes
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
        if isinstance(log_probs, np.ndarray):
            log_probs = torch.tensor(log_probs, device=self.device, dtype=torch.float32)
        
        # Reshape if needed
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        if log_probs.dim() == 1:
            log_probs = log_probs.unsqueeze(-1)
        
        self.observations[self.ptr] = obs
        self.states[self.ptr] = state
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = float(done)
        
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
            self.ptr = 0
    
    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            last_value: Value estimate for the last state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        size = self.buffer_size if self.full else self.ptr
        
        # Compute advantages using GAE
        advantages = torch.zeros(size, 1, device=self.device)
        last_gae = 0
        
        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t + 1]
            
            # TD error
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            
            # GAE
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        # Compute returns
        returns = advantages + self.values[:size]
        
        # Store computed values
        self.advantages[:size] = advantages
        self.returns[:size] = returns
    
    def get(self, batch_size=None):
        """
        Get all stored data in shuffled minibatches.
        
        Args:
            batch_size: Size of minibatches (if None, return all data)
            
        Yields:
            Dictionary containing batch data
        """
        size = self.buffer_size if self.full else self.ptr
        
        indices = torch.randperm(size, device=self.device)
        
        if batch_size is None:
            batch_size = size
        
        # Normalize advantages (important for training stability)
        advantages = self.advantages[:size]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for start_idx in range(0, size, batch_size):
            end_idx = min(start_idx + batch_size, size)
            batch_indices = indices[start_idx:end_idx]
            
            yield {
                'observations': self.observations[batch_indices],
                'states': self.states[batch_indices],
                'actions': self.actions[batch_indices],
                'old_log_probs': self.log_probs[batch_indices],
                'returns': self.returns[batch_indices],
                'advantages': advantages[batch_indices],
            }
    
    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.full = False

