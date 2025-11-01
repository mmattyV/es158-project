"""
Multi-Agent Reinforcement Learning for Power Grid Energy Flow Balancing

This script demonstrates the usage of the PowerGridEnv environment
and provides a simple test with random agents.
"""

import torch
import numpy as np
from power_grid_env import PowerGridEnv
import matplotlib.pyplot as plt


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


def main():
    """Main function to run the power grid environment demonstration."""
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment
    env = PowerGridEnv(device=device)
    
    # Demonstrate environment features
    demonstrate_environment_features(env)
    
    # Test with random agents
    episode_rewards, episode_lengths = test_random_agents(env, n_episodes=3, max_steps=50)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, 'o-')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths, 'o-')
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/Users/dereksmith/Documents/code/AP158/es158-project/test_results.png', dpi=150)
    plt.show()
    
    print(f"\nEnvironment test completed successfully!")
    print(f"Results saved to: test_results.png")


if __name__ == "__main__":
    main()
