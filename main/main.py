"""
Multi-Agent Reinforcement Learning for Power Grid Energy Flow Balancing

Main training script with TensorBoard integration and visualization.
"""

import torch
import numpy as np
import argparse
import os
from datetime import datetime
from power_grid_env import PowerGridEnv
from mappo import MAPPO
from buffer import RolloutBuffer
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def test_random_agents(env, n_episodes=5, max_steps=100):
    """Test the environment with random agents."""
    print("Testing PowerGridEnv with random agents...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"\nEpisode {episode + 1}:")
        print(f"Initial state - Mean frequency: {info['mean_frequency']:.3f} Hz")
        
        for step in range(max_steps):
            # Random actions for all agents
            actions = np.random.uniform(-1, 1, size=(env.n_agents,))
            
            obs, reward, terminated, truncated, info = env.step(actions)
            episode_reward += reward
            step_count += 1
            
            # Print every 20 steps
            if step % 20 == 0:
                print(f"  Step {step}: Reward={reward:.2f}, Mean freq={info['mean_frequency']:.3f} Hz, "
                      f"Safety violations={info['safety_violations']}")
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        print(f"Episode {episode + 1} finished: Total reward={episode_reward:.2f}, Steps={step_count}")
        if terminated:
            print("  Terminated due to safety violations")
        elif truncated:
            print("  Truncated due to max steps")
    
    print(f"\nTest Results:")
    print(f"Average episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    return episode_rewards, episode_lengths


def demonstrate_environment_features(env):
    """Demonstrate key features of the updated environment."""
    print("\n" + "="*60)
    print("POWER GRID ENVIRONMENT DEMONSTRATION")
    print("="*60)
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    print(f"\nEnvironment Configuration:")
    print(f"  Number of buses: {env.n_buses}")
    print(f"  Number of agents: {env.n_agents}")
    print(f"  Agent types: {env.agent_types}")
    print(f"  Ramp rates (MW/min): {env.ramp_rates.cpu().numpy()}")
    print(f"  Power limits (MW): min={env.power_min.cpu().numpy()}, max={env.power_max.cpu().numpy()}")
    print(f"  Cost coefficients ($/MW): {env.cost_coefficients.cpu().numpy()}")
    print(f"  Frequency bounds: {env.frequency_bounds} Hz")
    print(f"  Load range: {env.load_range} MW")
    print(f"  Communication delay: {env.delay_steps} steps")
    print(f"  Observation dimension: {env.obs_dim} per agent (15 as specified in proposal)")
    
    print(f"\nInitial State:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action space: {env.action_space}")
    print(f"  Mean frequency: {info['mean_frequency']:.3f} Hz")
    print(f"  System freq deviation: {info['system_freq_deviation']:.4f} Hz")
    print(f"  Total load: {info['total_load']:.1f} MW")
    print(f"  Total generation: {info['total_generation']:.1f} MW")
    print(f"  Current time: {info['current_hour']:.1f}h, Day {info['current_day']:.0f}")
    
    # Test full state vector
    full_state = env.get_full_state()
    print(f"  Full state shape: {full_state.shape} (for centralized critic)")
    
    # Test different action scenarios
    print(f"\nTesting Action Scenarios:")
    
    # Scenario 1: All agents increase output (respecting capacity)
    print("\n1. All agents increase output:")
    actions = np.ones(env.n_agents) * 0.5
    obs, reward, terminated, truncated, info = env.step(actions)
    print(f"   Reward: {reward:.2f}")
    print(f"   Mean frequency: {info['mean_frequency']:.3f} Hz")
    print(f"   System freq deviation: {info['system_freq_deviation']:.4f} Hz")
    print(f"   Total generation: {info['total_generation']:.1f} MW")
    print(f"   Capacity utilization: {[f'{u:.2f}' for u in info['agent_capacity_utilization'][:5]]}")
    
    # Scenario 2: Test capacity constraints
    print("\n2. Test capacity constraints (large actions):")
    actions = np.ones(env.n_agents) * 1.0  # Maximum actions
    obs, reward, terminated, truncated, info = env.step(actions)
    print(f"   Reward: {reward:.2f}")
    print(f"   Mean frequency: {info['mean_frequency']:.3f} Hz")
    print(f"   Safety violations: {info['safety_violations']}")
    print(f"   Capacity utilization: {[f'{u:.2f}' for u in info['agent_capacity_utilization'][:5]]}")
    
    # Scenario 3: Mixed actions by agent type
    print("\n3. Mixed actions by agent type:")
    actions = np.zeros(env.n_agents)
    actions[:5] = 0.3    # Batteries moderate increase
    actions[5:13] = -0.2 # Gas plants decrease
    actions[13:] = 0.1   # DR slight increase (load reduction)
    obs, reward, terminated, truncated, info = env.step(actions)
    print(f"   Reward: {reward:.2f}")
    print(f"   Mean frequency: {info['mean_frequency']:.3f} Hz")
    print(f"   System freq deviation: {info['system_freq_deviation']:.4f} Hz")
    print(f"   Total generation: {info['total_generation']:.1f} MW")
    
    print(f"\nAdvanced Features:")
    print(f"  Communication delay active: observations are {env.delay_steps} steps delayed")
    print(f"  Renewable forecasts: {env.renewable_forecasts.shape}")
    print(f"  Safety violations: {info['safety_violations']}")
    print(f"  Contingency active: {info['contingency_active']}")


def train_agent(env, agent, n_episodes, max_steps_per_episode, buffer_size, 
                n_epochs, batch_size, eval_interval, save_dir, device, writer):
    """
    Train the MAPPO agent with TensorBoard logging.
    
    Args:
        env: PowerGridEnv instance
        agent: MAPPO agent
        n_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        buffer_size: Rollout buffer size
        n_epochs: PPO update epochs
        batch_size: Minibatch size
        eval_interval: Episodes between evaluations
        save_dir: Directory to save models
        device: PyTorch device
        writer: TensorBoard writer
        
    Returns:
        Training statistics dictionary
    """
    # Create buffer
    buffer = RolloutBuffer(
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        buffer_size=buffer_size,
        device=device
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    actor_losses = []
    critic_losses = []
    entropies = []
    eval_rewards = []
    best_eval_reward = -float('inf')
    
    print(f"\nStarting training for {n_episodes} episodes...")
    print("=" * 60)
    print(f"TensorBoard logging to: {writer.log_dir}")
    print("=" * 60)
    
    for episode in range(1, n_episodes + 1):
        # Collect rollout
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps_per_episode):
            # Get full state for critic
            state = env.get_full_state().cpu().numpy()
            
            # Select actions
            actions, log_probs, value = agent.select_action(obs, state)
            
            # Step environment
            next_obs, reward, terminated, truncated, next_info = env.step(actions)
            
            # Store transition
            buffer.add(obs, state, actions, log_probs, reward, value, terminated or truncated)
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Log to TensorBoard
        writer.add_scalar('Train/EpisodeReward', episode_reward, episode)
        writer.add_scalar('Train/EpisodeLength', episode_length, episode)
        
        # Update policy when buffer has enough data
        if buffer.ptr >= batch_size or buffer.full:
            stats = agent.update(buffer, n_epochs=n_epochs, batch_size=batch_size)
            buffer.clear()
            
            if stats:
                actor_losses.append(stats['actor_loss'])
                critic_losses.append(stats['critic_loss'])
                entropies.append(stats['entropy'])
                
                # Log to TensorBoard
                writer.add_scalar('Train/ActorLoss', stats['actor_loss'], episode)
                writer.add_scalar('Train/CriticLoss', stats['critic_loss'], episode)
                writer.add_scalar('Train/Entropy', stats['entropy'], episode)
        
        # Periodic logging
        if episode % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode}/{n_episodes} | Avg Reward: {avg_reward:.2f} | Length: {episode_length}")
        
        # Evaluation
        if episode % eval_interval == 0:
            eval_reward = evaluate_agent(env, agent, n_episodes=5, max_steps=max_steps_per_episode)
            eval_rewards.append((episode, eval_reward))
            writer.add_scalar('Eval/AverageReward', eval_reward, episode)
            
            print(f"  Evaluation at episode {episode}: {eval_reward:.2f}")
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(os.path.join(save_dir, 'best_model.pt'))
                print(f"  ✓ New best model saved! Reward: {eval_reward:.2f}")
    
    # Save final model
    agent.save(os.path.join(save_dir, 'final_model.pt'))
    print("\n✓ Training completed!")
    print(f"Best evaluation reward: {best_eval_reward:.2f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'entropies': entropies,
        'eval_rewards': eval_rewards,
    }


def evaluate_agent(env, agent, n_episodes=5, max_steps=500):
    """Evaluate agent performance."""
    eval_rewards = []
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            state = env.get_full_state().cpu().numpy()
            actions, _, _ = agent.select_action(obs, state, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(actions)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        eval_rewards.append(episode_reward)
    
    return np.mean(eval_rewards)


def plot_training_results(stats, save_path):
    """
    Create comprehensive training visualizations.
    
    Args:
        stats: Dictionary of training statistics
        save_path: Path to save the plot
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Episode Rewards (with moving average)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(stats['episode_rewards'], alpha=0.3, label='Episode Reward', color='blue')
    if len(stats['episode_rewards']) >= 50:
        window = 50
        moving_avg = np.convolve(stats['episode_rewards'], np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(stats['episode_rewards'])), moving_avg, 
                label=f'{window}-Episode Moving Average', linewidth=2, color='darkblue')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress: Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Evaluation Rewards
    ax2 = fig.add_subplot(gs[0, 2])
    if stats['eval_rewards']:
        episodes, rewards = zip(*stats['eval_rewards'])
        ax2.plot(episodes, rewards, 'o-', linewidth=2, markersize=6, color='green')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Avg Reward')
        ax2.set_title('Evaluation Performance')
        ax2.grid(True, alpha=0.3)
    
    # 3. Actor Loss
    ax3 = fig.add_subplot(gs[1, 0])
    if stats['actor_losses']:
        ax3.plot(stats['actor_losses'], linewidth=1.5, color='red')
        ax3.set_xlabel('Update')
        ax3.set_ylabel('Loss')
        ax3.set_title('Actor Loss')
        ax3.grid(True, alpha=0.3)
    
    # 4. Critic Loss
    ax4 = fig.add_subplot(gs[1, 1])
    if stats['critic_losses']:
        ax4.plot(stats['critic_losses'], linewidth=1.5, color='orange')
        ax4.set_xlabel('Update')
        ax4.set_ylabel('Loss')
        ax4.set_title('Critic Loss')
        ax4.grid(True, alpha=0.3)
    
    # 5. Entropy
    ax5 = fig.add_subplot(gs[1, 2])
    if stats['entropies']:
        ax5.plot(stats['entropies'], linewidth=1.5, color='purple')
        ax5.set_xlabel('Update')
        ax5.set_ylabel('Entropy')
        ax5.set_title('Policy Entropy')
        ax5.grid(True, alpha=0.3)
    
    # 6. Episode Lengths
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(stats['episode_lengths'], alpha=0.5, color='teal')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Steps')
    ax6.set_title('Episode Lengths')
    ax6.grid(True, alpha=0.3)
    
    # 7. Cumulative Reward
    ax7 = fig.add_subplot(gs[2, 1])
    cumulative_rewards = np.cumsum(stats['episode_rewards'])
    ax7.plot(cumulative_rewards, linewidth=2, color='darkgreen')
    ax7.set_xlabel('Episode')
    ax7.set_ylabel('Cumulative Reward')
    ax7.set_title('Cumulative Training Reward')
    ax7.grid(True, alpha=0.3)
    
    # 8. Training Statistics Summary
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    separator = '=' * 25
    final_actor = f"{stats['actor_losses'][-1]:.4f}" if stats['actor_losses'] else 'N/A'
    final_critic = f"{stats['critic_losses'][-1]:.4f}" if stats['critic_losses'] else 'N/A'
    
    summary_text = f"""
    TRAINING SUMMARY
    {separator}
    
    Total Episodes: {len(stats['episode_rewards'])}
    
    Final Reward: {stats['episode_rewards'][-1]:.2f}
    Mean Reward: {np.mean(stats['episode_rewards']):.2f}
    Best Reward: {np.max(stats['episode_rewards']):.2f}
    
    Mean Length: {np.mean(stats['episode_lengths']):.1f}
    
    Final Actor Loss: {final_actor}
    Final Critic Loss: {final_critic}
    """
    
    ax8.text(0.1, 0.5, summary_text, transform=ax8.transAxes,
            fontsize=10, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training results saved to: {save_path}")
    plt.close()


def main():
    """Main training function with configurable parameters."""
    parser = argparse.ArgumentParser(description='Train MAPPO agent on power grid environment')
    
    # Training parameters
    parser.add_argument('--n_episodes', type=int, default=500, 
                       help='Number of training episodes (default: 500)')
    parser.add_argument('--max_steps', type=int, default=500,
                       help='Maximum steps per episode (default: 500)')
    parser.add_argument('--buffer_size', type=int, default=2048,
                       help='Rollout buffer size (default: 2048)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Minibatch size (default: 256)')
    parser.add_argument('--n_epochs', type=int, default=10,
                       help='PPO update epochs (default: 10)')
    parser.add_argument('--eval_interval', type=int, default=50,
                       help='Episodes between evaluations (default: 50)')
    
    # Model parameters
    parser.add_argument('--lr_actor', type=float, default=3e-4,
                       help='Actor learning rate (default: 3e-4)')
    parser.add_argument('--lr_critic', type=float, default=1e-3,
                       help='Critic learning rate (default: 1e-3)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                       help='GAE lambda (default: 0.95)')
    
    # Other parameters
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save models (default: checkpoints)')
    parser.add_argument('--test_env', action='store_true',
                       help='Run environment test before training')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: cpu, cuda, or auto (default: auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("=" * 60)
    print("MAPPO TRAINING FOR POWER GRID CONTROL")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Training episodes: {args.n_episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Buffer size: {args.buffer_size}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    # Create environment
    env = PowerGridEnv(device=device)
    
    # Optional: Test environment first
    if args.test_env:
        print("\nTesting environment with random agents...")
        demonstrate_environment_features(env)
        test_random_agents(env, n_episodes=3, max_steps=50)
        print("\n" + "=" * 60)
    
    # Create MAPPO agent
    agent = MAPPO(
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=1,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        device=device
    )
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create TensorBoard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('runs', f'mappo_{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)
    
    print(f"\n📊 TensorBoard Dashboard:")
    print(f"   Run: tensorboard --logdir=runs")
    print(f"   Then open: http://localhost:6006")
    print("=" * 60)
    
    # Train agent
    stats = train_agent(
        env=env,
        agent=agent,
        n_episodes=args.n_episodes,
        max_steps_per_episode=args.max_steps,
        buffer_size=args.buffer_size,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        save_dir=args.save_dir,
        device=device,
        writer=writer
    )
    
    # Close TensorBoard writer
    writer.close()
    
    # Generate final plots
    plot_path = os.path.join(args.save_dir, 'training_results.png')
    plot_training_results(stats, plot_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"✓ Models saved to: {args.save_dir}/")
    print(f"✓ TensorBoard logs: {log_dir}/")
    print(f"✓ Training plots: {plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
