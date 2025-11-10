"""
Evaluation and visualization script for trained MAPPO agent.
"""

import torch
import numpy as np
from power_grid_env import PowerGridEnv
from mappo import MAPPO
import matplotlib.pyplot as plt
import argparse


def evaluate_agent(env, agent, n_episodes=10, max_steps=500, render=False):
    """
    Evaluate a trained agent.
    
    Args:
        env: PowerGridEnv instance
        agent: MAPPO agent
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render during evaluation
        
    Returns:
        Dictionary with evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []
    frequency_violations = []
    mean_frequencies = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_violations = 0
        episode_frequencies = []
        
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        for step in range(max_steps):
            state = env.get_full_state().cpu().numpy()
            actions, _, _ = agent.select_action(obs, state, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(actions)
            
            episode_reward += reward
            episode_length += 1
            episode_violations += info['safety_violations']
            episode_frequencies.append(info['mean_frequency'])
            
            if render and step % 20 == 0:
                print(f"  Step {step}: Reward={reward:.2f}, Mean freq={info['mean_frequency']:.3f} Hz, "
                      f"Violations={info['safety_violations']}")
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        frequency_violations.append(episode_violations)
        mean_frequencies.append(np.mean(episode_frequencies))
        
        print(f"  Episode reward: {episode_reward:.2f}")
        print(f"  Episode length: {episode_length}")
        print(f"  Total violations: {episode_violations}")
        print(f"  Mean frequency: {np.mean(episode_frequencies):.3f} Hz")
    
    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_violations': np.mean(frequency_violations),
        'mean_frequency': np.mean(mean_frequencies),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
    }
    
    return stats


def visualize_episode(env, agent, max_steps=500, save_path='episode_visualization.png'):
    """
    Visualize a single episode in detail.
    
    Args:
        env: PowerGridEnv instance
        agent: MAPPO agent
        max_steps: Maximum steps
        save_path: Path to save visualization
    """
    obs, info = env.reset(seed=42)
    
    # Storage for visualization
    frequencies = []
    actions_history = []
    rewards_history = []
    loads = []
    generations = []
    
    for step in range(max_steps):
        state = env.get_full_state().cpu().numpy()
        actions, _, _ = agent.select_action(obs, state, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(actions)
        
        frequencies.append(info['mean_frequency'])
        actions_history.append(actions)
        rewards_history.append(reward)
        loads.append(info['total_load'])
        generations.append(info['total_generation'])
        
        if terminated or truncated:
            break
    
    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    time_steps = range(len(rewards_history))
    
    # Frequency over time
    axes[0, 0].plot(time_steps, frequencies, linewidth=2)
    axes[0, 0].axhline(y=60.0, color='k', linestyle='--', label='Nominal (60 Hz)')
    axes[0, 0].axhline(y=59.5, color='r', linestyle='--', alpha=0.5, label='Lower bound')
    axes[0, 0].axhline(y=60.5, color='r', linestyle='--', alpha=0.5, label='Upper bound')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Mean Frequency (Hz)')
    axes[0, 0].set_title('System Frequency Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Reward over time
    axes[0, 1].plot(time_steps, rewards_history, linewidth=2)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].set_title('Reward Over Time')
    axes[0, 1].grid(True)
    
    # Load vs Generation
    axes[1, 0].plot(time_steps, loads, label='Total Load', linewidth=2)
    axes[1, 0].plot(time_steps, generations, label='Total Generation', linewidth=2)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Power (MW)')
    axes[1, 0].set_title('Load vs Generation Balance')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Power imbalance
    imbalance = np.array(generations) - np.array(loads)
    axes[1, 1].plot(time_steps, imbalance, linewidth=2, color='purple')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Imbalance (MW)')
    axes[1, 1].set_title('Power Imbalance (Generation - Load)')
    axes[1, 1].grid(True)
    
    # Actions by agent type
    actions_array = np.array(actions_history)
    axes[2, 0].plot(time_steps, actions_array[:, :5].mean(axis=1), label='Batteries (avg)', linewidth=2)
    axes[2, 0].plot(time_steps, actions_array[:, 5:13].mean(axis=1), label='Gas Plants (avg)', linewidth=2)
    axes[2, 0].plot(time_steps, actions_array[:, 13:].mean(axis=1), label='DR (avg)', linewidth=2)
    axes[2, 0].set_xlabel('Time Step')
    axes[2, 0].set_ylabel('Action Value')
    axes[2, 0].set_title('Average Actions by Agent Type')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    # Cumulative reward
    cumulative_reward = np.cumsum(rewards_history)
    axes[2, 1].plot(time_steps, cumulative_reward, linewidth=2, color='green')
    axes[2, 1].set_xlabel('Time Step')
    axes[2, 1].set_ylabel('Cumulative Reward')
    axes[2, 1].set_title('Cumulative Reward Over Time')
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to {save_path}")
    plt.close()


def compare_with_baseline(env, agent, n_episodes=10):
    """
    Compare trained agent with random baseline.
    
    Args:
        env: PowerGridEnv instance
        agent: Trained MAPPO agent
        n_episodes: Number of episodes
    """
    print("\n" + "="*60)
    print("COMPARISON: Trained Agent vs Random Baseline")
    print("="*60)
    
    # Evaluate trained agent
    print("\nEvaluating trained agent...")
    trained_stats = evaluate_agent(env, agent, n_episodes=n_episodes, max_steps=500, render=False)
    
    # Evaluate random baseline
    print("\nEvaluating random baseline...")
    random_rewards = []
    random_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(500):
            actions = np.random.uniform(-1, 1, size=(env.n_agents,))
            obs, reward, terminated, truncated, info = env.step(actions)
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        random_rewards.append(episode_reward)
        random_lengths.append(episode_length)
    
    # Print comparison
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"\nTrained Agent:")
    print(f"  Mean reward: {trained_stats['mean_reward']:.2f} ± {trained_stats['std_reward']:.2f}")
    print(f"  Mean length: {trained_stats['mean_length']:.1f}")
    print(f"  Mean violations: {trained_stats['mean_violations']:.1f}")
    print(f"  Mean frequency: {trained_stats['mean_frequency']:.3f} Hz")
    
    print(f"\nRandom Baseline:")
    print(f"  Mean reward: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
    print(f"  Mean length: {np.mean(random_lengths):.1f}")
    
    improvement = ((trained_stats['mean_reward'] - np.mean(random_rewards)) / 
                   abs(np.mean(random_rewards)) * 100)
    print(f"\nImprovement: {improvement:+.1f}%")
    print("="*60)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained MAPPO agent')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pt',
                      help='Path to trained model')
    parser.add_argument('--n_episodes', type=int, default=10,
                      help='Number of evaluation episodes')
    parser.add_argument('--visualize', action='store_true',
                      help='Create detailed episode visualization')
    parser.add_argument('--compare', action='store_true',
                      help='Compare with random baseline')
    parser.add_argument('--render', action='store_true',
                      help='Render during evaluation')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment
    env = PowerGridEnv(device=device)
    
    # Create agent
    agent = MAPPO(
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=1,
        device=device
    )
    
    # Load trained model
    try:
        agent.load(args.model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        print("Please train a model first using train.py")
        return
    
    # Evaluate
    stats = evaluate_agent(env, agent, n_episodes=args.n_episodes, 
                          max_steps=500, render=args.render)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Mean episode length: {stats['mean_length']:.1f}")
    print(f"Mean safety violations: {stats['mean_violations']:.1f}")
    print(f"Mean system frequency: {stats['mean_frequency']:.3f} Hz")
    
    # Visualize
    if args.visualize:
        print("\nCreating episode visualization...")
        visualize_episode(env, agent, max_steps=500)
    
    # Compare with baseline
    if args.compare:
        compare_with_baseline(env, agent, n_episodes=args.n_episodes)


if __name__ == "__main__":
    main()

