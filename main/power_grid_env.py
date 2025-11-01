"""
Multi-Agent Power Grid Environment for Reinforcement Learning

This environment simulates a 68-bus power grid with 20 agents:
- 5 batteries (ramp rate: 50 MW/min)
- 8 gas plants (ramp rate: 10 MW/min) 
- 7 demand response units (ramp rate: 5 MW/min)

State space: 140-dimensional (bus frequencies, generator outputs, loads)
Each agent observes: 15-dimensional local observation
Actions: 20 continuous power changes ΔP^i within ramp rate limits
"""

import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import math
import random


class PowerGridEnv(gym.Env):
    """Multi-agent power grid environment using gymnasium interface."""
    
    def __init__(self, 
                 n_buses: int = 68,
                 n_agents: int = 20,
                 dt: float = 1.0,  # time step in minutes
                 frequency_bounds: Tuple[float, float] = (59.5, 60.5),
                 load_range: Tuple[float, float] = (2000.0, 5000.0),
                 contingency_prob: float = 0.001,
                 device: str = 'cpu'):
        """
        Initialize the power grid environment.
        
        Args:
            n_buses: Number of buses in the power grid (68)
            n_agents: Number of agents (20: 5 batteries + 8 gas + 7 DR)
            dt: Time step in minutes
            frequency_bounds: Frequency bounds in Hz [59.5, 60.5]
            load_range: Load range in MW [2000, 5000]
            contingency_prob: Probability of N-1 contingency per step
            device: PyTorch device ('cpu' or 'cuda')
        """
        super().__init__()
        
        self.device = torch.device(device)
        self.n_buses = n_buses
        self.n_agents = n_agents
        self.dt = dt
        self.frequency_bounds = frequency_bounds
        self.load_range = load_range
        self.contingency_prob = contingency_prob
        
        # Agent configuration: [batteries, gas plants, demand response]
        self.agent_types = ['battery'] * 5 + ['gas'] * 8 + ['dr'] * 7
        self.ramp_rates = torch.tensor([50.0] * 5 + [10.0] * 8 + [5.0] * 7, device=self.device)  # MW/min
        
        # Agent capacity constraints [P_min, P_max] in MW
        self.power_min = torch.tensor([0.0] * 5 + [50.0] * 8 + [-200.0] * 7, device=self.device)
        self.power_max = torch.tensor([100.0] * 5 + [500.0] * 8 + [0.0] * 7, device=self.device)
        
        # Agent-specific cost coefficients ($/MW)
        self.cost_coefficients = torch.tensor([5.0] * 5 + [50.0] * 8 + [20.0] * 7, device=self.device)
        
        # Wear-and-tear coefficients for different agent types
        self.wear_coefficients = torch.tensor([0.1] * 5 + [0.05] * 8 + [0.2] * 7, device=self.device)
        
        # Power system parameters
        self.nominal_frequency = 60.0  # Hz
        self.base_power = 100.0  # MVA base
        self.inertia_constants = torch.rand(n_buses, device=self.device) * 5.0 + 2.0  # H in seconds
        
        # Grid topology - simplified admittance matrix (68x68)
        self._initialize_grid_topology()
        
        # State space: [frequencies (68), generator outputs (20), renewables (14), loads (30), time features (8)] = 140
        self.state_dim = 140
        self.obs_dim = 15  # Local observation dimension per agent as specified in proposal
        
        # Action space: continuous power changes for each agent
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_agents,), dtype=np.float32
        )
        
        # Observation space: each agent gets local 15-dim observation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_agents, self.obs_dim), dtype=np.float32
        )
        
        # Communication delay buffer (2-second SCADA delay)
        self.delay_steps = 2  # 2 time steps delay
        self.observation_buffer = []
        
        # Renewable forecast parameters
        self.forecast_horizon = 5  # 5 time steps ahead
        self.renewable_forecasts = torch.zeros(14, self.forecast_horizon, device=self.device)
        
        # Initialize state variables
        self.reset()
    
    def _initialize_grid_topology(self):
        """Initialize the power grid topology and admittance matrix."""
        # Simplified 68-bus system admittance matrix
        # In practice, this would be based on actual grid topology
        self.admittance_matrix = torch.zeros(self.n_buses, self.n_buses, device=self.device)
        
        # Create a connected graph structure
        for i in range(self.n_buses):
            # Self admittance (diagonal elements)
            self.admittance_matrix[i, i] = torch.rand(1, device=self.device) * 10.0 + 5.0
            
            # Connect to nearby buses (simplified ring + radial structure)
            if i < self.n_buses - 1:
                admittance = torch.rand(1, device=self.device) * 2.0 + 1.0
                self.admittance_matrix[i, i+1] = -admittance
                self.admittance_matrix[i+1, i] = -admittance
                self.admittance_matrix[i, i] += admittance
                self.admittance_matrix[i+1, i+1] += admittance
        
        # Add some additional connections for robustness
        for _ in range(self.n_buses // 4):
            i, j = torch.randint(0, self.n_buses, (2,))
            if i != j:
                admittance = torch.rand(1, device=self.device) * 1.0 + 0.5
                self.admittance_matrix[i, j] = -admittance
                self.admittance_matrix[j, i] = -admittance
                self.admittance_matrix[i, i] += admittance
                self.admittance_matrix[j, j] += admittance
        
        # Agent-to-bus mapping (which buses have controllable agents)
        self.agent_bus_mapping = torch.randint(0, self.n_buses, (self.n_agents,), device=self.device)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            observations: Array of shape (n_agents, obs_dim)
            info: Dictionary with additional information
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize bus frequencies around nominal (60 Hz)
        self.frequencies = torch.ones(self.n_buses, device=self.device) * self.nominal_frequency
        self.frequencies += torch.randn(self.n_buses, device=self.device) * 0.01  # Small initial deviation
        
        # Initialize generator outputs (MW) within capacity bounds
        self.generator_outputs = torch.zeros(self.n_agents, device=self.device)
        # Set initial outputs to minimum capacity for gas plants and mid-range for batteries
        self.generator_outputs[:5] = 50.0  # Batteries at 50% capacity
        self.generator_outputs[5:13] = 100.0  # Gas plants at minimum
        self.generator_outputs[13:] = -50.0  # DR at 25% load reduction
        
        # Initialize loads (MW) - random within specified range
        load_min, load_max = self.load_range
        self.loads = torch.rand(self.n_buses, device=self.device) * (load_max - load_min) + load_min
        
        # Initialize renewable generation (14 buses with renewables)
        self.renewable_buses = torch.randint(0, self.n_buses, (14,), device=self.device)
        self.renewable_generation = torch.rand(14, device=self.device) * 500.0  # 0-500 MW
        
        # Initialize voltage angles (radians)
        self.voltage_angles = torch.zeros(self.n_buses, device=self.device)
        
        # Reset contingency state
        self.contingency_active = False
        self.contingency_bus = None
        
        # Time step counter and time features
        self.time_step = 0
        self.current_hour = 12.0  # Start at noon
        self.current_day = 1.0    # Monday
        
        # Initialize observation buffer for communication delay
        initial_obs = self._get_observations_immediate()
        self.observation_buffer = [initial_obs.clone() for _ in range(self.delay_steps + 1)]
        
        # Initialize renewable forecasts
        self._update_renewable_forecasts()
        
        # Get initial observations (with delay)
        observations = self._get_observations()
        info = self._get_info()
        
        return observations.cpu().numpy(), info
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step of the environment.
        
        Args:
            actions: Array of shape (n_agents,) with continuous actions in [-1, 1]
            
        Returns:
            observations: Next state observations
            reward: Shared reward for all agents
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
        
        # Convert normalized actions to actual power changes (MW)
        power_changes = actions * self.ramp_rates * self.dt  # Scale by ramp rate and time step
        
        # Apply action constraints - check capacity limits before updating
        feasible_changes = torch.zeros_like(power_changes)
        for i in range(self.n_agents):
            current_power = self.generator_outputs[i]
            desired_change = power_changes[i]
            
            # Calculate feasible change respecting capacity bounds
            max_increase = self.power_max[i] - current_power
            max_decrease = self.power_min[i] - current_power
            
            feasible_changes[i] = torch.clamp(desired_change, max_decrease, max_increase)
        
        # Apply feasible power changes to generator outputs
        self.generator_outputs += feasible_changes
        
        # Ensure outputs stay within bounds (safety check)
        self.generator_outputs = torch.clamp(self.generator_outputs, self.power_min, self.power_max)
        
        # Store action for wear calculation
        self.last_actions = feasible_changes
        
        # Update time features
        self._update_time_features()
        
        # Update stochastic loads and renewable generation
        self._update_stochastic_components()
        
        # Update renewable forecasts
        self._update_renewable_forecasts()
        
        # Check for N-1 contingencies
        self._check_contingencies()
        
        # Solve power flow and update frequencies using swing equation
        self._update_system_dynamics()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated, truncated = self._check_termination()
        
        # Update observation buffer and get delayed observations
        current_obs = self._get_observations_immediate()
        self.observation_buffer.append(current_obs)
        if len(self.observation_buffer) > self.delay_steps + 1:
            self.observation_buffer.pop(0)
        
        # Get delayed observations
        observations = self._get_observations()
        
        # Update time step
        self.time_step += 1
        
        info = self._get_info()
        
        return observations.cpu().numpy(), reward, terminated, truncated, info
    
    def _update_stochastic_components(self):
        """Update stochastic load and renewable generation."""
        # Add random variations to loads (±5% per step)
        load_variation = torch.randn(self.n_buses, device=self.device) * 0.05
        self.loads *= (1.0 + load_variation)
        self.loads = torch.clamp(self.loads, self.load_range[0], self.load_range[1])
        
        # Update renewable generation with stochastic variations
        renewable_variation = torch.randn(14, device=self.device) * 0.1
        self.renewable_generation *= (1.0 + renewable_variation)
        self.renewable_generation = torch.clamp(self.renewable_generation, 0.0, 1000.0)
    
    def _check_contingencies(self):
        """Check for N-1 contingency events."""
        if torch.rand(1).item() < self.contingency_prob:
            if not self.contingency_active:
                # Activate contingency - disconnect a random bus
                self.contingency_active = True
                self.contingency_bus = torch.randint(0, self.n_buses, (1,)).item()
                # Reduce load at contingency bus
                self.loads[self.contingency_bus] *= 0.1
        else:
            # Recover from contingency
            if self.contingency_active:
                self.contingency_active = False
                if self.contingency_bus is not None:
                    # Restore load
                    load_min, load_max = self.load_range
                    self.loads[self.contingency_bus] = torch.rand(1, device=self.device) * (load_max - load_min) + load_min
                self.contingency_bus = None
    
    def _update_system_dynamics(self):
        """Update system dynamics using swing equation."""
        # Calculate power imbalance at each bus
        power_injection = torch.zeros(self.n_buses, device=self.device)
        
        # Add generator outputs to their respective buses
        for i, bus_idx in enumerate(self.agent_bus_mapping):
            power_injection[bus_idx] += self.generator_outputs[i]
        
        # Add renewable generation
        for i, bus_idx in enumerate(self.renewable_buses):
            power_injection[bus_idx] += self.renewable_generation[i]
        
        # Subtract loads
        power_injection -= self.loads
        
        # Simplified swing equation: df/dt = (P_mech - P_elec) / (2 * H * f_nom)
        # where P_elec is calculated from power flow
        
        # Calculate electrical power flow (simplified)
        frequency_deviations = self.frequencies - self.nominal_frequency
        electrical_power = torch.matmul(self.admittance_matrix, frequency_deviations)
        
        # Power imbalance
        power_imbalance = power_injection - electrical_power
        
        # Update frequencies using swing equation
        frequency_derivative = power_imbalance / (2.0 * self.inertia_constants * self.nominal_frequency)
        self.frequencies += frequency_derivative * self.dt * 60.0  # Convert minutes to seconds
        
        # Enforce frequency bounds
        self.frequencies = torch.clamp(self.frequencies, self.frequency_bounds[0], self.frequency_bounds[1])
    
    def _calculate_reward(self):
        """Calculate the shared reward based on equation 2 with exact coefficients."""
        # 1. Frequency deviation penalty: -1000 * sum of squared deviations
        frequency_deviations = self.frequencies - self.nominal_frequency
        frequency_penalty = 1000.0 * torch.sum(frequency_deviations ** 2)
        
        # 2. Agent-specific costs: C_i per MW adjusted (based on last actions)
        if hasattr(self, 'last_actions'):
            agent_costs = torch.sum(self.cost_coefficients * torch.abs(self.last_actions))
        else:
            agent_costs = 0.0
        
        # 3. Wear-and-tear functions: 0.1 * W_i (based on action magnitude)
        if hasattr(self, 'last_actions'):
            wear_costs = 0.1 * torch.sum(self.wear_coefficients * (self.last_actions ** 2))
        else:
            wear_costs = 0.0
        
        # 4. Safety constraint violations: exactly 10,000 per violation
        safety_violations = 0.0
        freq_violations = torch.sum((self.frequencies < self.frequency_bounds[0]) | 
                                  (self.frequencies > self.frequency_bounds[1]))
        safety_violations = freq_violations * 10000.0
        
        # Total reward (negative because we minimize costs)
        reward = -(frequency_penalty + agent_costs + wear_costs + safety_violations)
        
        return reward.item()
    
    def _get_observations(self):
        """Get delayed observations for each agent (2-second SCADA delay)."""
        if len(self.observation_buffer) >= self.delay_steps + 1:
            return self.observation_buffer[-(self.delay_steps + 1)]  # Return delayed observation
        else:
            return self.observation_buffer[0]  # Return most recent if buffer not full
    
    def _get_observations_immediate(self):
        """Get immediate (non-delayed) observations for each agent."""
        observations = torch.zeros(self.n_agents, self.obs_dim, device=self.device)
        
        # Calculate system frequency deviation (key coordination signal)
        system_freq_deviation = torch.mean(self.frequencies - self.nominal_frequency)
        
        for i in range(self.n_agents):
            bus_idx = self.agent_bus_mapping[i]
            obs_idx = 0
            
            # Local bus frequency (1)
            observations[i, obs_idx] = self.frequencies[bus_idx]
            obs_idx += 1
            
            # Local bus load (1)
            observations[i, obs_idx] = self.loads[bus_idx]
            obs_idx += 1
            
            # Own generator output (1)
            observations[i, obs_idx] = self.generator_outputs[i]
            obs_idx += 1
            
            # System frequency deviation Δf_sys = (1/68)Σ(f_k - 60) (1)
            observations[i, obs_idx] = system_freq_deviation
            obs_idx += 1
            
            # Nearby bus frequencies (5 nearest buses)
            distances = torch.abs(torch.arange(self.n_buses, device=self.device) - bus_idx)
            _, nearest_indices = torch.topk(distances, k=6, largest=False)  # 6 to exclude self
            nearest_indices = nearest_indices[1:]  # Remove self
            observations[i, obs_idx:obs_idx+5] = self.frequencies[nearest_indices]
            obs_idx += 5
            
            # Renewable generation forecasts - next 3 time steps (3)
            # Use actual forecast values for next 3 time steps, averaged across all renewable sources
            if self.renewable_forecasts.shape[1] >= 3:
                # Average forecast across all renewable sources for next 3 time steps
                observations[i, obs_idx:obs_idx+3] = torch.mean(self.renewable_forecasts[:, :3], dim=0)
            else:
                # Fallback if not enough forecast steps
                observations[i, obs_idx:obs_idx+3] = torch.mean(self.renewable_generation)
            obs_idx += 3
            
            # Time features: hour of day, day of week (2)
            observations[i, obs_idx] = self.current_hour / 24.0  # Normalized hour
            observations[i, obs_idx+1] = self.current_day / 7.0  # Normalized day
            
            # Total: 1+1+1+1+5+3+2 = 14, need 1 more for 15
            # Add own capacity utilization (1)
            capacity_range = self.power_max[i] - self.power_min[i]
            if capacity_range > 0:
                utilization = (self.generator_outputs[i] - self.power_min[i]) / capacity_range
            else:
                utilization = 0.0
            observations[i, 14] = utilization
        
        return observations
    
    def _check_termination(self):
        """Check if episode should be terminated or truncated."""
        # Terminate if frequency violations are too severe
        severe_violations = torch.sum((self.frequencies < 59.0) | (self.frequencies > 61.0))
        terminated = severe_violations > 0
        
        # Truncate after maximum time steps (e.g., 1000 steps = ~16.7 hours)
        truncated = self.time_step >= 1000
        
        return terminated, truncated
    
    def _update_time_features(self):
        """Update time-based features (hour of day, day of week)."""
        # Advance time by dt minutes
        self.current_hour += self.dt / 60.0
        if self.current_hour >= 24.0:
            self.current_hour -= 24.0
            self.current_day += 1.0
            if self.current_day > 7.0:
                self.current_day = 1.0
    
    def _update_renewable_forecasts(self):
        """Update renewable generation forecasts for next 5-15 minutes."""
        # Simple forecast model: current + trend + noise
        for i in range(14):
            current_gen = self.renewable_generation[i]
            
            # Add trend (seasonal pattern)
            trend = 10.0 * torch.sin(2 * math.pi * self.current_hour / 24.0)  # Daily pattern
            
            # Add noise
            noise = torch.randn(self.forecast_horizon, device=self.device) * 20.0
            
            # Generate forecast
            for t in range(self.forecast_horizon):
                forecast = current_gen + trend * (t + 1) + noise[t]
                self.renewable_forecasts[i, t] = torch.clamp(forecast, 0.0, 1000.0)
    
    def get_full_state(self):
        """Get the complete 140-dimensional state vector for centralized critic."""
        # State components: frequencies (68) + generator outputs (20) + renewables (14) + loads (30) + time features (8) = 140
        state = torch.zeros(self.state_dim, device=self.device)
        
        idx = 0
        # Bus frequencies (68)
        state[idx:idx+68] = self.frequencies
        idx += 68
        
        # Generator outputs (20)
        state[idx:idx+20] = self.generator_outputs
        idx += 20
        
        # Renewable generation (14)
        state[idx:idx+14] = self.renewable_generation
        idx += 14
        
        # Loads (first 30 buses - most critical/largest loads)
        state[idx:idx+30] = self.loads[:30]
        idx += 30
        
        # Time features (8): hour, day, hour_sin, hour_cos, day_sin, day_cos, load_pattern, renewable_pattern
        state[idx] = self.current_hour / 24.0  # Normalized hour
        state[idx+1] = self.current_day / 7.0  # Normalized day
        state[idx+2] = torch.sin(2 * math.pi * self.current_hour / 24.0)  # Hour sine
        state[idx+3] = torch.cos(2 * math.pi * self.current_hour / 24.0)  # Hour cosine
        state[idx+4] = torch.sin(2 * math.pi * self.current_day / 7.0)   # Day sine
        state[idx+5] = torch.cos(2 * math.pi * self.current_day / 7.0)   # Day cosine
        state[idx+6] = torch.mean(self.loads)  # Average load pattern
        state[idx+7] = torch.mean(self.renewable_generation)  # Average renewable pattern
        
        return state
    
    def _get_info(self):
        """Get additional information about the environment state."""
        system_freq_deviation = torch.mean(self.frequencies - self.nominal_frequency).item()
        
        return {
            'time_step': self.time_step,
            'mean_frequency': torch.mean(self.frequencies).item(),
            'frequency_std': torch.std(self.frequencies).item(),
            'system_freq_deviation': system_freq_deviation,
            'total_generation': torch.sum(self.generator_outputs).item(),
            'total_load': torch.sum(self.loads).item(),
            'contingency_active': self.contingency_active,
            'safety_violations': torch.sum((self.frequencies < self.frequency_bounds[0]) | 
                                         (self.frequencies > self.frequency_bounds[1])).item(),
            'current_hour': self.current_hour,
            'current_day': self.current_day,
            'agent_capacity_utilization': [(self.generator_outputs[i] - self.power_min[i]) / 
                                         (self.power_max[i] - self.power_min[i]) 
                                         for i in range(self.n_agents)]
        }
    
    def render(self, mode='human'):
        """Render the environment (optional)."""
        if mode == 'human':
            print(f"Step: {self.time_step}")
            print(f"Mean frequency: {torch.mean(self.frequencies):.3f} Hz")
            print(f"Frequency range: [{torch.min(self.frequencies):.3f}, {torch.max(self.frequencies):.3f}] Hz")
            print(f"Total generation: {torch.sum(self.generator_outputs):.1f} MW")
            print(f"Total load: {torch.sum(self.loads):.1f} MW")
            print(f"Contingency active: {self.contingency_active}")
            print("-" * 50)
    
    def close(self):
        """Clean up resources."""
        pass
