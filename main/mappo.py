"""
Multi-Agent Proximal Policy Optimization (MAPPO) implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from networks import Actor, Critic
from buffer import RolloutBuffer
import numpy as np


class MAPPO:
    """MAPPO agent with centralized critic and decentralized actors."""
    
    def __init__(self,
                 n_agents,
                 obs_dim,
                 state_dim,
                 action_dim=1,
                 lr_actor=3e-4,
                 lr_critic=1e-3,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_epsilon=0.2,
                 entropy_coef=0.01,
                 value_coef=0.5,
                 max_grad_norm=0.5,
                 device='cpu'):
        """
        Initialize MAPPO agent.
        
        Args:
            n_agents: Number of agents (10)
            obs_dim: Observation dimension per agent (15)
            state_dim: Full state dimension (55)
            action_dim: Action dimension per agent (1)
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: PyTorch device
        """
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Create actor and critic networks
        self.actor = Actor(obs_dim=obs_dim, action_dim=action_dim).to(device)
        self.critic = Critic(state_dim=state_dim).to(device)
        
        # Create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        print(f"MAPPO Agent initialized:")
        print(f"  Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        print(f"  Critic parameters: {sum(p.numel() for p in self.critic.parameters())}")
    
    def select_action(self, obs, state, deterministic=False):
        """
        Select actions for all agents.
        
        Args:
            obs: Observations [n_agents, obs_dim]
            state: Full state [state_dim]
            deterministic: If True, return mean actions
            
        Returns:
            actions: Actions for all agents [n_agents]
            log_probs: Log probabilities [n_agents]
            value: Value estimate (scalar)
        """
        with torch.no_grad():
            # Convert to tensors if needed
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
            if isinstance(state, np.ndarray):
                state = torch.tensor(state, device=self.device, dtype=torch.float32)
            
            # Get actions from actor
            actions, log_probs = self.actor.get_action(obs, deterministic=deterministic)
            
            # Get value from critic
            value = self.critic(state)
            
            # Squeeze to remove extra dimensions
            actions = actions.squeeze(-1)  # [n_agents]
            log_probs = log_probs.squeeze(-1)  # [n_agents]
            value = value.item()  # scalar
        
        # Clip actions to [-1, 1] range
        actions = torch.clamp(actions, -1.0, 1.0)
        
        return actions.cpu().numpy(), log_probs.cpu().numpy(), value
    
    def update(self, buffer, n_epochs=10, batch_size=256):
        """
        Update actor and critic networks using PPO.
        
        Args:
            buffer: RolloutBuffer with experience
            n_epochs: Number of update epochs
            batch_size: Minibatch size
            
        Returns:
            Dictionary with training statistics
        """
        # Compute last value for GAE
        size = buffer.buffer_size if buffer.full else buffer.ptr
        if size == 0:
            return {}
        
        last_obs = buffer.observations[size - 1]
        last_state = buffer.states[size - 1]
        
        with torch.no_grad():
            last_value = self.critic(last_state).item()
        
        # Compute returns and advantages
        buffer.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)
        
        # Training statistics
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        # Multiple epochs over the data
        for epoch in range(n_epochs):
            for batch in buffer.get(batch_size=batch_size):
                # Extract batch data
                obs = batch['observations']  # [batch_size, n_agents, obs_dim]
                states = batch['states']  # [batch_size, state_dim]
                actions = batch['actions']  # [batch_size, n_agents, 1]
                old_log_probs = batch['old_log_probs']  # [batch_size, n_agents, 1]
                returns = batch['returns']  # [batch_size, 1]
                advantages = batch['advantages']  # [batch_size, 1]
                
                # ===== Update Critic =====
                values = self.critic(states)
                value_loss = self.value_coef * nn.MSELoss()(values, returns)
                
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # ===== Update Actor =====
                # Evaluate actions under current policy
                log_probs, entropy = self.actor.evaluate_actions(obs, actions)
                
                # Sum log probs across agents (joint action)
                log_probs_sum = log_probs.sum(dim=1, keepdim=True)  # [batch_size, 1]
                old_log_probs_sum = old_log_probs.sum(dim=1, keepdim=True)  # [batch_size, 1]
                
                # Compute ratio and clipped objective
                ratio = torch.exp(log_probs_sum - old_log_probs_sum)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus (encourage exploration)
                entropy_loss = -entropy.mean()
                
                # Total actor loss
                total_loss = actor_loss + self.entropy_coef * entropy_loss
                
                self.actor_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Track statistics
                total_actor_loss += actor_loss.item()
                total_critic_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # Compute average statistics
        stats = {
            'actor_loss': total_actor_loss / max(n_updates, 1),
            'critic_loss': total_critic_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
            'n_updates': n_updates,
        }
        
        return stats
    
    def save(self, filepath):
        """Save model weights."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {filepath}")

