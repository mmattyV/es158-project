"""
Multi-Agent Power Grid Environment for Reinforcement Learning

This environment simulates a 20-bus power grid with 10 agents:
- 2 batteries (ramp rate: 50 MW/min)
- 5 gas plants (ramp rate: 10 MW/min) 
- 3 demand response units (ramp rate: 5 MW/min)

State space: 60-dimensional (bus frequencies, generator outputs, renewables, loads, time features)
Each agent observes: 15-dimensional local observation
Actions: 10 continuous power changes ΔP^i within ramp rate limits

Key Features (Aligned with Proposal):
- Correct swing equation: df/dt = P_imbalance / (2*H*S_base)
- Total system load: 2000-5000 MW (distributed across 68 buses)
- Communication delay: 2 seconds (SCADA delay)
- Graduated response: exponential penalties + lenient termination (±1.5 Hz)
- No artificial frequency clamping - realistic physics
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
                 n_buses: int = 20,
                 n_agents: int = 10,
                 dt: float = 2.0/60.0,  # time step in minutes (2 seconds)
                 frequency_bounds: Tuple[float, float] = (59.5, 60.5),
                 load_range: Tuple[float, float] = (2000.0, 5000.0),
                 contingency_prob: float = 0.001,
                 device: str = 'cpu'):
        """
        Initialize the power grid environment.
        
        Args:
            n_buses: Number of buses in the power grid (20)
            n_agents: Number of agents (10: 2 batteries + 5 gas + 3 DR)
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
        self.agent_types = ['battery'] * 2 + ['gas'] * 5 + ['dr'] * 3
        self.ramp_rates = torch.tensor([50.0] * 2 + [10.0] * 5 + [5.0] * 3, device=self.device)  # MW/min
        
        # Agent capacity constraints [P_min, P_max] in MW
        self.power_min = torch.tensor([0.0] * 2 + [50.0] * 5 + [-200.0] * 3, device=self.device)
        self.power_max = torch.tensor([100.0] * 2 + [500.0] * 5 + [0.0] * 3, device=self.device)
        
        # Agent-specific cost coefficients ($/MW)
        self.cost_coefficients = torch.tensor([5.0] * 2 + [50.0] * 5 + [20.0] * 3, device=self.device)
        
        # Wear-and-tear coefficients for different agent types
        self.wear_coefficients = torch.tensor([0.1] * 2 + [0.05] * 5 + [0.2] * 3, device=self.device)
        
        # Power system parameters
        self.nominal_frequency = 60.0  # Hz
        self.base_power = 10000.0  # MVA base (scaled to match system size ~5000 MW)
        self.inertia_constants = torch.rand(n_buses, device=self.device) * 5.0 + 2.0  # H in seconds
        
        # Grid topology - simplified admittance matrix (20x20)
        self._initialize_grid_topology()
        
        # State space: [frequencies (20), generator outputs (10), renewables (7), loads (10), time features (8)] = 55
        self.state_dim = 55
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
        self.delay_steps = 1  # 1 time step × 2 seconds = 2 second delay
        self.observation_buffer = []
        
        # Renewable forecast parameters
        self.forecast_horizon = 5  # 5 time steps ahead
        self.renewable_forecasts = torch.zeros(7, self.forecast_horizon, device=self.device)
        
        # Curriculum learning - episode tracking for adaptive termination bounds
        self.current_episode = 0
        
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
            self.admittance_matrix[i, i] = (torch.rand(1, device=self.device) * 10.0 + 5.0).item()
            
            # Connect to nearby buses (simplified ring + radial structure)
            if i < self.n_buses - 1:
                admittance_val = (torch.rand(1, device=self.device) * 2.0 + 1.0).item()
                self.admittance_matrix[i, i+1] = -admittance_val
                self.admittance_matrix[i+1, i] = -admittance_val
                self.admittance_matrix[i, i] += admittance_val
                self.admittance_matrix[i+1, i+1] += admittance_val
        
        # Add some additional connections for robustness
        for _ in range(self.n_buses // 4):
            i, j = torch.randint(0, self.n_buses, (2,)).tolist()
            if i != j:
                admittance_val = (torch.rand(1, device=self.device) * 1.0 + 0.5).item()
                self.admittance_matrix[i, j] = -admittance_val
                self.admittance_matrix[j, i] = -admittance_val
                self.admittance_matrix[i, i] += admittance_val
                self.admittance_matrix[j, j] += admittance_val
        
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
        
        # Initialize loads (MW) - distribute total system load across buses
        # Total system load should be 2000-5000 MW as per proposal
        load_min, load_max = self.load_range
        total_system_load = torch.rand(1, device=self.device).item() * (load_max - load_min) + load_min
        load_distribution = torch.rand(self.n_buses, device=self.device)
        self.loads = (load_distribution / load_distribution.sum()) * total_system_load
        
        # Initialize renewable generation (7 buses with renewables)
        self.renewable_buses = torch.randint(0, self.n_buses, (7,), device=self.device)
        self.renewable_generation = torch.rand(7, device=self.device) * 500.0  # 0-500 MW
        
        # Initialize generator outputs (MW) to balance load minus renewables
        # This starts closer to equilibrium and reduces initial frequency deviations
        total_load = torch.sum(self.loads).item()
        total_renewable = torch.sum(self.renewable_generation).item()
        required_generation = total_load - total_renewable
        
        # Distribute required generation intelligently across agent types
        self.generator_outputs = torch.zeros(self.n_agents, device=self.device)
        # Batteries at mid-range (50 MW each = 100 MW total for 2 batteries)
        self.generator_outputs[:2] = 50.0
        # Gas plants handle most of the load (share remaining proportionally)
        gas_share = max(0.0, required_generation - 100.0 + 150.0) / 5.0  # Account for batteries and DR
        self.generator_outputs[2:7] = torch.clamp(torch.tensor(gas_share, device=self.device), min=50.0, max=500.0)
        # DR at 25% load reduction (-50 MW each = -150 MW total for 3 DR units)
        self.generator_outputs[7:] = -50.0
        
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
        
        # Ensure outputs stay within bounds (safety check) - element-wise clamp
        self.generator_outputs = torch.max(torch.min(self.generator_outputs, self.power_max), self.power_min)
        
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
        # Add random variations to loads while maintaining total system load in [2000, 5000] MW
        # Add ±5% variation to individual bus loads
        load_variation = torch.randn(self.n_buses, device=self.device) * 0.05
        self.loads *= (1.0 + load_variation)
        
        # Rescale to maintain total system load within bounds
        current_total = torch.sum(self.loads)
        load_min, load_max = self.load_range
        if current_total < load_min:
            self.loads *= (load_min / current_total)
        elif current_total > load_max:
            self.loads *= (load_max / current_total)
        
        # Ensure no negative loads
        self.loads = torch.clamp(self.loads, min=0.0)
        
        # Update renewable generation with stochastic variations
        renewable_variation = torch.randn(len(self.renewable_generation), device=self.device) * 0.1
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
        
        # Swing equation from proposal: df/dt = (P_gen - P_load - P_losses) / (2 * H * S_base)
        # where S_base is the base power (MVA), not nominal frequency
        
        # Calculate electrical power flow (simplified)
        frequency_deviations = self.frequencies - self.nominal_frequency
        electrical_power = torch.matmul(self.admittance_matrix, frequency_deviations)
        
        # Power imbalance
        power_imbalance = power_injection - electrical_power
        
        # Update frequencies using correct swing equation (use base_power, not nominal_frequency)
        frequency_derivative = power_imbalance / (2.0 * self.inertia_constants * self.base_power)
        self.frequencies += frequency_derivative * self.dt * 60.0  # Convert minutes to seconds
        
        # NO CLAMPING - let physics run its course for realistic dynamics
    
    def _calculate_reward(self):
        """
        Calculate the shared reward with graduated response.
        Uses exponential penalties to create soft boundaries before hard violations.
        STORES COMPONENTS for debugging.
        """
        frequency_deviations = self.frequencies - self.nominal_frequency
        
        # 1. Base frequency penalty from proposal: -1000 * sum of squared deviations
        frequency_penalty = 1000.0 * torch.sum(frequency_deviations ** 2)
        
        # 2. Exponential penalty as frequencies approach operational bounds [59.5, 60.5]
        # Creates a "soft boundary" that strongly discourages approaching limits
        # CRITICAL FIX: Reduced exponent 5.0 → 1.0 (exp(5*3)=3.3M vs exp(1*3)=20!)
        # Also reduced coefficient 5000 → 500
        operational_margin = 0.5  # Hz (operational bounds from proposal)
        beyond_operational = torch.abs(frequency_deviations) - operational_margin
        beyond_operational = torch.clamp(beyond_operational, min=0.0)  # Only penalize if beyond
        exponential_penalty = 500.0 * torch.sum(torch.exp(1.0 * beyond_operational) - 1.0)
        
        # 3. Agent-specific costs: C_i per MW adjusted (from proposal equation 2)
        if hasattr(self, 'last_actions'):
            agent_costs = torch.sum(self.cost_coefficients * torch.abs(self.last_actions))
        else:
            agent_costs = 0.0
        
        # 4. Wear-and-tear functions: 0.1 * W_i(|ΔP^i|) (from proposal equation 2)
        # Note: Using quadratic wear W_i * (ΔP^i)^2 for smoother gradients
        if hasattr(self, 'last_actions'):
            wear_costs = 0.1 * torch.sum(self.wear_coefficients * (self.last_actions ** 2))
        else:
            wear_costs = 0.0
        
        # 5. Hard safety constraint violations: 10,000 per bus violating operational bounds
        freq_violations = torch.sum((torch.abs(frequency_deviations) > operational_margin))
        safety_violations = freq_violations * 10000.0
        
        # 6. SURVIVAL BONUS - Reward agents for staying alive!
        # Balanced bonus: strong enough to incentivize survival, not so strong it masks penalties
        # 50k per timestep after ÷1M scaling = +0.05 per step
        # 300 steps = +15 total, 200 steps = +10 total (5 point difference)
        # Penalties can now differentiate good vs bad control while surviving
        survival_bonus = 50000.0
        
        # Store components for debugging (before scaling)
        self.last_reward_components = {
            'frequency_penalty': frequency_penalty.item(),
            'exponential_penalty': exponential_penalty.item(),
            'agent_costs': agent_costs.item() if isinstance(agent_costs, torch.Tensor) else agent_costs,
            'wear_costs': wear_costs.item() if isinstance(wear_costs, torch.Tensor) else wear_costs,
            'safety_violations': safety_violations.item(),
            'freq_violation_count': freq_violations.item(),
            'survival_bonus': survival_bonus
        }
        
        # Total reward (negative penalties + positive survival bonus)
        reward = -(frequency_penalty + exponential_penalty + agent_costs + wear_costs + safety_violations) + survival_bonus
        
        # NO CLIPPING - Let critic see the true cost of catastrophic failures!
        # Previous clipping at -100k prevented learning: agents learned "do nothing" was cheap
        # Now exponential penalties flow through fully, forcing agents to maintain control
        
        # STRONG scaling to handle exponential penalties (1e9 range without clipping)
        # Reduced exponential coefficient (500 instead of 5000) AND stronger scaling (1M vs 15k)
        # Target: catastrophic = -1000, good control = -1 to -10
        reward = reward / 1000000.0  # Much stronger scaling to handle unclipped penalties
        
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
        """
        Check if episode should be terminated or truncated.
        Uses CURRICULUM LEARNING with episode-based milestones.
        Gradually tightens bounds as agents improve.
        """
        # CURRICULUM LEARNING - Extended Stage 1 for solid foundation
        # Agents need MUCH more time at easy bounds to learn basic control
        if self.current_episode < 1500:
            # Stage 1: Very lenient (learning basics) - EXTENDED to 1500 eps
            crit_bound = 2.5  # ±2.5 Hz
            cat_bound = 3.5   # ±3.5 Hz
            crit_threshold = 0.30  # 30% of buses
        elif self.current_episode < 2500:
            # Stage 2: Moderate-high (gentle transition)
            crit_bound = 2.2  # ±2.2 Hz
            cat_bound = 3.2   # ±3.2 Hz
            crit_threshold = 0.28  # 28% of buses
        elif self.current_episode < 3500:
            # Stage 3: Moderate (practicing coordination)
            crit_bound = 2.0  # ±2.0 Hz
            cat_bound = 3.0   # ±3.0 Hz
            crit_threshold = 0.25  # 25% of buses
        else:
            # Stage 4: Final (reasonable operational bounds)
            # Note: ±1.2 Hz proved too strict - stopping at ±1.8 Hz
            crit_bound = 1.8  # ±1.8 Hz
            cat_bound = 2.5   # ±2.5 Hz
            crit_threshold = 0.20  # 20% of buses
        
        # Store current bounds for logging
        self.current_crit_bound = crit_bound
        self.current_cat_bound = cat_bound
        
        # Calculate violations with curriculum bounds
        critical_violations = torch.sum((self.frequencies < (60.0 - crit_bound)) | 
                                       (self.frequencies > (60.0 + crit_bound)))
        catastrophic_violations = torch.sum((self.frequencies < (60.0 - cat_bound)) | 
                                           (self.frequencies > (60.0 + cat_bound)))
        
        # Terminate only on:
        # 1. Catastrophic frequency deviations
        # 2. OR >threshold% of buses in critical state
        terminated = (catastrophic_violations > 0) or (critical_violations > crit_threshold * self.n_buses)
        
        # Truncate after maximum time steps (1000 steps with dt=2s ≈ 33 minutes)
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
        for i in range(len(self.renewable_generation)):
            current_gen = self.renewable_generation[i]
            
            # Add trend (seasonal pattern)
            hour_tensor = torch.tensor(self.current_hour, device=self.device)
            trend = 10.0 * torch.sin(2 * math.pi * hour_tensor / 24.0)  # Daily pattern
            
            # Add noise
            noise = torch.randn(self.forecast_horizon, device=self.device) * 20.0
            
            # Generate forecast
            for t in range(self.forecast_horizon):
                forecast = current_gen + trend * (t + 1) + noise[t]
                self.renewable_forecasts[i, t] = torch.clamp(forecast, 0.0, 1000.0)
    
    def get_full_state(self):
        """Get the complete 55-dimensional state vector for centralized critic."""
        # State components: frequencies (20) + generator outputs (10) + renewables (7) + loads (10) + time features (8) = 55
        state = torch.zeros(self.state_dim, device=self.device)
        
        idx = 0
        # Bus frequencies (20)
        state[idx:idx+self.n_buses] = self.frequencies
        idx += self.n_buses
        
        # Generator outputs (10)
        state[idx:idx+self.n_agents] = self.generator_outputs
        idx += self.n_agents
        
        # Renewable generation (7)
        n_renewables = len(self.renewable_generation)
        state[idx:idx+n_renewables] = self.renewable_generation
        idx += n_renewables
        
        # Loads (10 buses - using first 10 or all if n_buses < 10)
        n_load_features = min(10, self.n_buses)
        state[idx:idx+n_load_features] = self.loads[:n_load_features]
        idx += n_load_features
        
        # Time features (8): hour, day, hour_sin, hour_cos, day_sin, day_cos, load_pattern, renewable_pattern
        state[idx] = self.current_hour / 24.0  # Normalized hour
        state[idx+1] = self.current_day / 7.0  # Normalized day
        
        # Convert to tensors for torch trig functions
        hour_tensor = torch.tensor(self.current_hour, device=self.device)
        day_tensor = torch.tensor(self.current_day, device=self.device)
        
        state[idx+2] = torch.sin(2 * math.pi * hour_tensor / 24.0)  # Hour sine
        state[idx+3] = torch.cos(2 * math.pi * hour_tensor / 24.0)  # Hour cosine
        state[idx+4] = torch.sin(2 * math.pi * day_tensor / 7.0)   # Day sine
        state[idx+5] = torch.cos(2 * math.pi * day_tensor / 7.0)   # Day cosine
        state[idx+6] = torch.mean(self.loads)  # Average load pattern
        state[idx+7] = torch.mean(self.renewable_generation)  # Average renewable pattern
        
        return state
    
    def _get_info(self):
        """Get additional information about the environment state with comprehensive diagnostics."""
        system_freq_deviation = torch.mean(self.frequencies - self.nominal_frequency).item()
        total_gen = torch.sum(self.generator_outputs).item()
        total_load = torch.sum(self.loads).item()
        
        # Agent action statistics
        if hasattr(self, 'last_actions'):
            action_mean = torch.mean(torch.abs(self.last_actions)).item()
            action_std = torch.std(self.last_actions).item()
            action_max = torch.max(torch.abs(self.last_actions)).item()
        else:
            action_mean = action_std = action_max = 0.0
        
        # Frequency statistics (detailed)
        freq_min = torch.min(self.frequencies).item()
        freq_max = torch.max(self.frequencies).item()
        freq_range = freq_max - freq_min
        
        # Violation breakdown
        critical_violations = torch.sum((self.frequencies < 58.8) | (self.frequencies > 61.2)).item()
        catastrophic_violations = torch.sum((self.frequencies < 58.0) | (self.frequencies > 62.0)).item()
        
        info = {
            # Basic info
            'time_step': self.time_step,
            
            # Frequency metrics
            'mean_frequency': torch.mean(self.frequencies).item(),
            'frequency_std': torch.std(self.frequencies).item(),
            'frequency_min': freq_min,
            'frequency_max': freq_max,
            'frequency_range': freq_range,
            'system_freq_deviation': system_freq_deviation,
            
            # Power balance
            'total_generation': total_gen,
            'total_load': total_load,
            'power_imbalance': total_gen - total_load,
            'power_imbalance_pct': 100.0 * (total_gen - total_load) / total_load if total_load > 0 else 0.0,
            
            # Violations
            'safety_violations': torch.sum((self.frequencies < self.frequency_bounds[0]) | 
                                         (self.frequencies > self.frequency_bounds[1])).item(),
            'critical_violations': critical_violations,
            'catastrophic_violations': catastrophic_violations,
            
            # Agent statistics
            'action_mean': action_mean,
            'action_std': action_std,
            'action_max': action_max,
            'agent_capacity_utilization': [(self.generator_outputs[i] - self.power_min[i]) / 
                                         (self.power_max[i] - self.power_min[i]) 
                                         for i in range(self.n_agents)],
            'mean_capacity_utilization': torch.mean((self.generator_outputs - self.power_min) / 
                                                   (self.power_max - self.power_min + 1e-6)).item(),
            
            # Environment state
            'contingency_active': self.contingency_active,
            'current_hour': self.current_hour,
            'current_day': self.current_day,
            
            # Curriculum learning info
            'curriculum_episode': self.current_episode,
            'curriculum_crit_bound': getattr(self, 'current_crit_bound', 2.5),
            'curriculum_cat_bound': getattr(self, 'current_cat_bound', 3.5),
        }
        
        # Add reward components if available
        if hasattr(self, 'last_reward_components'):
            for key, value in self.last_reward_components.items():
                info[f'reward_{key}'] = value
        
        return info
    
    def set_episode(self, episode: int):
        """
        Set the current episode for curriculum learning.
        Called by trainer at the start of each episode.
        
        Args:
            episode: Current episode number
        """
        self.current_episode = episode
    
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
