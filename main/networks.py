"""
Neural network architectures for MAPPO agent.

Actor: Decentralized policy network (15-dim local obs -> action distribution)
Critic: Centralized value network (140-dim full state -> value estimate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class Actor(nn.Module):
    """Decentralized actor network for each agent."""
    
    def __init__(self, obs_dim=15, hidden_dims=[128, 128], action_dim=1, log_std_min=-20, log_std_max=2):
        """
        Initialize actor network.
        
        Args:
            obs_dim: Observation dimension per agent (15)
            hidden_dims: List of hidden layer dimensions
            action_dim: Action dimension per agent (1)
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build network layers
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # Smaller initialization for output layer
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
    
    def forward(self, obs):
        """
        Forward pass through actor network.
        
        Args:
            obs: Observations [batch_size, n_agents, obs_dim] or [batch_size, obs_dim]
            
        Returns:
            action_dist: Normal distribution over actions
        """
        # Handle both batched and single agent observations
        original_shape = obs.shape
        if len(obs.shape) == 3:
            batch_size, n_agents, obs_dim = obs.shape
            obs = obs.reshape(batch_size * n_agents, obs_dim)
            batched = True
        else:
            batched = False
        
        # Forward pass
        features = self.shared_net(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # Create action distribution
        action_dist = Normal(mean, std)
        
        # Reshape if needed
        if batched:
            mean = mean.reshape(batch_size, n_agents, -1)
            std = std.reshape(batch_size, n_agents, -1)
        
        return action_dist
    
    def get_action(self, obs, deterministic=False):
        """
        Sample action from policy.
        
        Args:
            obs: Observations
            deterministic: If True, return mean action
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        action_dist = self.forward(obs)
        
        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.sample()
        
        log_prob = action_dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions under current policy.
        
        Args:
            obs: Observations [batch_size, n_agents, obs_dim]
            actions: Actions to evaluate [batch_size, n_agents, action_dim]
            
        Returns:
            log_prob: Log probabilities of actions
            entropy: Entropy of action distribution
        """
        action_dist = self.forward(obs)
        
        # Reshape actions if needed
        if len(actions.shape) == 3:
            batch_size, n_agents, action_dim = actions.shape
            actions = actions.reshape(batch_size * n_agents, action_dim)
        
        log_prob = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = action_dist.entropy().sum(dim=-1, keepdim=True)
        
        # Reshape back
        if len(obs.shape) == 3:
            batch_size, n_agents, obs_dim = obs.shape
            log_prob = log_prob.reshape(batch_size, n_agents, 1)
            entropy = entropy.reshape(batch_size, n_agents, 1)
        
        return log_prob, entropy


class Critic(nn.Module):
    """Centralized critic network for value estimation."""
    
    def __init__(self, state_dim=140, hidden_dims=[256, 256]):
        """
        Initialize critic network.
        
        Args:
            state_dim: Full state dimension (140)
            hidden_dims: List of hidden layer dimensions
        """
        super(Critic, self).__init__()
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        """
        Forward pass through critic network.
        
        Args:
            state: Full state [batch_size, state_dim]
            
        Returns:
            value: Value estimate [batch_size, 1]
        """
        return self.net(state)

