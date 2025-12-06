"""
Training script for MAPPO on power grid environment.
"""

import torch
import numpy as np
from power_grid_env import PowerGridEnv
from mappo import MAPPO
from buffer import RolloutBuffer
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """Trainer for MAPPO agent."""
    
    def __init__(self,
                 env,
                 agent,
                 buffer_size=2048,
                 n_epochs=10,
                 batch_size=256,
                 log_interval=10,
                 save_interval=100,
                 eval_interval=50,
                 save_dir='checkpoints',
                 device='cpu'):
        """
        Initialize trainer.
        
        Args:
            env: PowerGridEnv instance
            agent: MAPPO agent
            buffer_size: Rollout buffer size
            n_epochs: Number of PPO epochs per update
            batch_size: Minibatch size
            log_interval: Episodes between logging
            save_interval: Episodes between checkpoints
            eval_interval: Episodes between evaluations
            save_dir: Directory to save checkpoints and logs
            device: PyTorch device
        """
        self.env = env
        self.agent = agent
        self.buffer_size = buffer_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        self.device = device
        
        # Create buffer
        self.buffer = RolloutBuffer(
            n_agents=env.n_agents,
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            buffer_size=buffer_size,
            device=device
        )
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Create TensorBoard writer
        log_dir = os.path.join('runs', f'mappo_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.eval_rewards = []
        
        print(f"Trainer initialized:")
        print(f"  Buffer size: {buffer_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Update epochs: {n_epochs}")
        print(f"  Save directory: {save_dir}")
        print(f"  TensorBoard logs: {log_dir}")
    
    def collect_rollout(self, max_steps=500):
        """
        Collect a single episode rollout.
        
        Args:
            max_steps: Maximum steps per episode
            
        Returns:
            episode_reward: Total episode reward
            episode_length: Episode length
            final_info: Final info dict with episode statistics
        """
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        final_info = info
        
        for step in range(max_steps):
            # Get full state for critic
            state = self.env.get_full_state().cpu().numpy()
            
            # Select actions
            actions, log_probs, value = self.agent.select_action(obs, state)
            
            # Step environment
            next_obs, reward, terminated, truncated, next_info = self.env.step(actions)
            
            # Store transition
            self.buffer.add(obs, state, actions, log_probs, reward, value, terminated or truncated)
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            final_info = next_info  # Keep updating to get final stats
            
            if terminated or truncated:
                break
        
        return episode_reward, episode_length, final_info
    
    def train(self, n_episodes=1000, max_steps_per_episode=500):
        """
        Main training loop.
        
        Args:
            n_episodes: Total number of training episodes
            max_steps_per_episode: Maximum steps per episode
        """
        print(f"\nStarting training for {n_episodes} episodes...")
        print("=" * 60)
        
        total_steps = 0
        best_eval_reward = -float('inf')
        
        for episode in range(1, n_episodes + 1):
            # Set episode for curriculum learning
            self.env.set_episode(episode)
            
            # Collect rollout
            episode_reward, episode_length, final_info = self.collect_rollout(max_steps_per_episode)
            total_steps += episode_length
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Log basic metrics to TensorBoard
            self.writer.add_scalar('Train/EpisodeReward', episode_reward, episode)
            self.writer.add_scalar('Train/EpisodeLength', episode_length, episode)
            
            # Log comprehensive environment diagnostics
            if final_info:
                # Frequency metrics
                self.writer.add_scalar('Env/MeanFrequency', final_info.get('mean_frequency', 0), episode)
                self.writer.add_scalar('Env/FrequencyStd', final_info.get('frequency_std', 0), episode)
                self.writer.add_scalar('Env/FrequencyMin', final_info.get('frequency_min', 0), episode)
                self.writer.add_scalar('Env/FrequencyMax', final_info.get('frequency_max', 0), episode)
                self.writer.add_scalar('Env/FrequencyRange', final_info.get('frequency_range', 0), episode)
                
                # Power balance
                self.writer.add_scalar('Env/TotalGeneration', final_info.get('total_generation', 0), episode)
                self.writer.add_scalar('Env/TotalLoad', final_info.get('total_load', 0), episode)
                self.writer.add_scalar('Env/PowerImbalance', final_info.get('power_imbalance', 0), episode)
                self.writer.add_scalar('Env/PowerImbalancePct', final_info.get('power_imbalance_pct', 0), episode)
                
                # Violations
                self.writer.add_scalar('Env/SafetyViolations', final_info.get('safety_violations', 0), episode)
                self.writer.add_scalar('Env/CriticalViolations', final_info.get('critical_violations', 0), episode)
                self.writer.add_scalar('Env/CatastrophicViolations', final_info.get('catastrophic_violations', 0), episode)
                
                # Agent actions
                self.writer.add_scalar('Env/ActionMean', final_info.get('action_mean', 0), episode)
                self.writer.add_scalar('Env/ActionStd', final_info.get('action_std', 0), episode)
                self.writer.add_scalar('Env/ActionMax', final_info.get('action_max', 0), episode)
                self.writer.add_scalar('Env/MeanCapacityUtilization', final_info.get('mean_capacity_utilization', 0), episode)
                
                # Reward components (CRITICAL for debugging!)
                self.writer.add_scalar('Reward/FrequencyPenalty', final_info.get('reward_frequency_penalty', 0), episode)
                self.writer.add_scalar('Reward/ExponentialPenalty', final_info.get('reward_exponential_penalty', 0), episode)
                self.writer.add_scalar('Reward/AgentCosts', final_info.get('reward_agent_costs', 0), episode)
                self.writer.add_scalar('Reward/WearCosts', final_info.get('reward_wear_costs', 0), episode)
                self.writer.add_scalar('Reward/SafetyViolations', final_info.get('reward_safety_violations', 0), episode)
                self.writer.add_scalar('Reward/ViolationCount', final_info.get('reward_freq_violation_count', 0), episode)
                self.writer.add_scalar('Reward/SurvivalBonus', final_info.get('reward_survival_bonus', 0), episode)
            
            # Log curriculum bounds (for tracking training stages)
            if hasattr(self.env, 'current_crit_bound'):
                self.writer.add_scalar('Curriculum/CriticalBound', self.env.current_crit_bound, episode)
                self.writer.add_scalar('Curriculum/CatastrophicBound', self.env.current_cat_bound, episode)
            
            # Update policy when buffer is full (standard MAPPO: collect full rollout before updating)
            if self.buffer.full:
                stats = self.agent.update(self.buffer, n_epochs=self.n_epochs, batch_size=self.batch_size)
                self.buffer.clear()
                
                if stats:
                    self.actor_losses.append(stats['actor_loss'])
                    self.critic_losses.append(stats['critic_loss'])
                    
                    # Log losses to TensorBoard
                    self.writer.add_scalar('Train/ActorLoss', stats['actor_loss'], episode)
                    self.writer.add_scalar('Train/CriticLoss', stats['critic_loss'], episode)
                    self.writer.add_scalar('Train/Entropy', stats['entropy'], episode)
            
            # Logging
            if episode % self.log_interval == 0:
                recent_rewards = self.episode_rewards[-self.log_interval:]
                recent_lengths = self.episode_lengths[-self.log_interval:]
                avg_reward = np.mean(recent_rewards)
                avg_length = np.mean(recent_lengths)
                
                print(f"Episode {episode}/{n_episodes}")
                print(f"  Total steps: {total_steps}")
                print(f"  Avg reward (last {self.log_interval}): {avg_reward:.2f}")
                print(f"  Avg length (last {self.log_interval}): {avg_length:.1f}")
                
                # Show curriculum stage info
                if hasattr(self.env, 'current_crit_bound'):
                    print(f"  Curriculum bounds: ±{self.env.current_crit_bound:.1f} / ±{self.env.current_cat_bound:.1f} Hz")
                
                if self.actor_losses:
                    print(f"  Actor loss: {self.actor_losses[-1]:.4f}")
                    print(f"  Critic loss: {self.critic_losses[-1]:.4f}")
                print("-" * 60)
            
            # Evaluation
            if episode % self.eval_interval == 0:
                eval_reward = self.evaluate(n_eval_episodes=5)
                self.eval_rewards.append((episode, eval_reward))
                print(f"Evaluation at episode {episode}: {eval_reward:.2f}")
                
                # Log evaluation to TensorBoard
                self.writer.add_scalar('Eval/AverageReward', eval_reward, episode)
                
                # Save best model
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.agent.save(os.path.join(self.save_dir, 'best_model.pt'))
                    print(f"  New best model saved! Reward: {eval_reward:.2f}")
            
            # Save checkpoint
            if episode % self.save_interval == 0:
                self.save_checkpoint(episode)
        
        print("\nTraining completed!")
        print(f"Best evaluation reward: {best_eval_reward:.2f}")
        
        # Final update with any remaining data in buffer (if buffer has enough samples)
        if self.buffer.ptr >= self.batch_size:
            stats = self.agent.update(self.buffer, n_epochs=self.n_epochs, batch_size=self.batch_size)
            if stats:
                self.actor_losses.append(stats['actor_loss'])
                self.critic_losses.append(stats['critic_loss'])
        
        # Save final model and statistics
        self.agent.save(os.path.join(self.save_dir, 'final_model.pt'))
        self.save_training_stats()
        self.plot_training_curves()
        
        # Close TensorBoard writer
        self.writer.close()
        print(f"TensorBoard logs saved. View with: tensorboard --logdir=runs")
    
    def evaluate(self, n_eval_episodes=10, max_steps=500):
        """
        Evaluate the agent.
        
        Args:
            n_eval_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            
        Returns:
            mean_reward: Mean reward across evaluation episodes
        """
        eval_rewards = []
        
        for _ in range(n_eval_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                state = self.env.get_full_state().cpu().numpy()
                actions, _, _ = self.agent.select_action(obs, state, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(actions)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards)
    
    def save_checkpoint(self, episode):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_ep{episode}.pt')
        self.agent.save(checkpoint_path)
    
    def save_training_stats(self):
        """Save training statistics to JSON."""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'eval_rewards': self.eval_rewards,
        }
        
        stats_path = os.path.join(self.save_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Training statistics saved to {stats_path}")
    
    def plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Episode reward')
        if len(self.episode_rewards) >= 10:
            # Moving average
            window = 50
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(self.episode_rewards)), moving_avg, 
                           label=f'{window}-episode moving avg', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths, alpha=0.5)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].grid(True)
        
        # Actor loss
        if self.actor_losses:
            axes[1, 0].plot(self.actor_losses)
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Actor Loss')
            axes[1, 0].grid(True)
        
        # Critic loss
        if self.critic_losses:
            axes[1, 1].plot(self.critic_losses)
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Critic Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Training curves saved to {plot_path}")
        plt.close()
        
        # Plot evaluation rewards separately
        if self.eval_rewards:
            plt.figure(figsize=(10, 6))
            episodes, rewards = zip(*self.eval_rewards)
            plt.plot(episodes, rewards, 'o-', linewidth=2, markersize=8)
            plt.xlabel('Episode')
            plt.ylabel('Evaluation Reward')
            plt.title('Evaluation Performance')
            plt.grid(True)
            eval_plot_path = os.path.join(self.save_dir, 'eval_rewards.png')
            plt.savefig(eval_plot_path, dpi=150)
            print(f"Evaluation plot saved to {eval_plot_path}")
            plt.close()


def main():
    """Main training function."""
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment
    env = PowerGridEnv(device=device)
    
    # Create MAPPO agent
    agent = MAPPO(
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=1,
        lr_actor=4e-4,  # Balanced LR for stable yet adaptive learning
        lr_critic=3e-4,  # Reduced from 1e-3 for more stable critic learning
        gamma=0.99,
        gae_lambda=0.99,  # Increased from 0.95 - trust actual rewards more than bootstrapped values
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=2.0,  # Increased from 1.0 - FORCE critic to learn before actor updates
        max_grad_norm=0.5,
        device=device
    )
    
    # Create trainer
    trainer = Trainer(
        env=env,
        agent=agent,
        buffer_size=2048,
        n_epochs=10,
        batch_size=256,
        log_interval=10,
        save_interval=100,
        eval_interval=50,
        save_dir='checkpoints',
        device=device
    )
    
    # Start training with EXTENDED curriculum (4000 episodes)
    # Stage 1 (Ep 1-1500): Very lenient (±2.5/3.5 Hz) - EXTENDED for solid foundation
    # Stage 2 (Ep 1501-2500): Moderate-high (±2.2/3.2 Hz) - Gentle transition
    # Stage 3 (Ep 2501-3500): Moderate (±2.0/3.0 Hz) - Practicing coordination
    # Stage 4 (Ep 3501+): Final (±1.8/2.5 Hz) - Operational bounds (±1.2 proved too hard)
    trainer.train(n_episodes=4000, max_steps_per_episode=500)


if __name__ == "__main__":
    main()

