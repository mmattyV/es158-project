"""
Power Grid Environment Dashboard

Interactive dashboard for visualizing and controlling the multi-agent power grid environment.
Features real-time plots, grid topology, agent controls, and key metrics.
Supports loading trained MAPPO agents for evaluation.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import numpy as np
import torch
from power_grid_env import PowerGridEnv
from collections import deque
import time
import os


class PowerGridDashboard:
    """Interactive dashboard for the power grid environment."""
    
    def __init__(self, env=None, max_history=200, agent=None, model_path=None):
        """
        Initialize the dashboard.
        
        Args:
            env: PowerGridEnv instance (creates new one if None)
            max_history: Maximum number of time steps to keep in history
            agent: Trained MAPPO agent (optional)
            model_path: Path to trained model checkpoint (optional, loads if provided)
        """
        self.env = env if env is not None else PowerGridEnv()
        self.max_history = max_history
        
        # AI Agent support
        self.agent = agent
        self.model_path = model_path
        self.control_mode = 'manual'  # Options: 'manual', 'random', 'ai'
        
        # Load model if path provided
        if model_path is not None and agent is None:
            self._load_agent(model_path)
        
        # Data storage for real-time plotting
        self.time_history = deque(maxlen=max_history)
        self.frequency_history = deque(maxlen=max_history)
        self.generation_history = deque(maxlen=max_history)
        self.load_history = deque(maxlen=max_history)
        self.reward_history = deque(maxlen=max_history)
        self.safety_violations_history = deque(maxlen=max_history)
        self.agent_outputs_history = {i: deque(maxlen=max_history) for i in range(self.env.n_agents)}
        
        # Current state
        self.current_step = 0
        self.is_running = False
        self.current_actions = np.zeros(self.env.n_agents)
        
        # Initialize environment
        self.obs, self.info = self.env.reset(seed=42)
        self._update_history()
        
        # Create the dashboard
        self._create_dashboard()
    
    def _load_agent(self, model_path):
        """Load a trained MAPPO agent from checkpoint."""
        try:
            from mappo import MAPPO
            
            if not os.path.exists(model_path):
                print(f"⚠️  Model file not found: {model_path}")
                print("   Dashboard will run without AI agent.")
                return
            
            device = self.env.device
            self.agent = MAPPO(
                n_agents=self.env.n_agents,
                obs_dim=self.env.obs_dim,
                state_dim=self.env.state_dim,
                action_dim=1,
                device=device
            )
            
            self.agent.load(model_path)
            self.control_mode = 'ai'  # Default to AI mode if model loaded
            print(f"✓ Loaded trained agent from: {model_path}")
            print(f"  Control mode: AI Agent")
            
        except Exception as e:
            print(f"⚠️  Error loading agent: {e}")
            print("   Dashboard will run without AI agent.")
            self.agent = None
    
    def _create_dashboard(self):
        """Create the matplotlib dashboard interface."""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 12))
        
        # Dynamic title based on control mode
        title = 'Power Grid Multi-Agent RL Dashboard'
        if self.agent is not None:
            title += ' - 🤖 AI Agent Loaded'
        self.fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = self.fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. System Frequency Plot (top left)
        self.ax_freq = self.fig.add_subplot(gs[0, :2])
        self.ax_freq.set_title('System Frequencies (Hz)')
        self.ax_freq.set_ylabel('Frequency (Hz)')
        self.ax_freq.grid(True, alpha=0.3)
        self.ax_freq.axhline(y=60.0, color='g', linestyle='--', alpha=0.7, label='Nominal')
        self.ax_freq.axhline(y=59.5, color='r', linestyle='--', alpha=0.7, label='Lower Bound')
        self.ax_freq.axhline(y=60.5, color='r', linestyle='--', alpha=0.7, label='Upper Bound')
        self.ax_freq.legend()
        
        # 2. Generation vs Load (top right)
        self.ax_power = self.fig.add_subplot(gs[0, 2:])
        self.ax_power.set_title('Generation vs Load (MW)')
        self.ax_power.set_ylabel('Power (MW)')
        self.ax_power.grid(True, alpha=0.3)
        
        # 3. Agent Outputs (middle left)
        self.ax_agents = self.fig.add_subplot(gs[1, :2])
        self.ax_agents.set_title('Agent Power Outputs by Type')
        self.ax_agents.set_ylabel('Power Output (MW)')
        self.ax_agents.grid(True, alpha=0.3)
        
        # 4. Rewards and Violations (middle right)
        self.ax_reward = self.fig.add_subplot(gs[1, 2:])
        self.ax_reward.set_title('Rewards and Safety Violations')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.grid(True, alpha=0.3)
        
        # 5. Grid Topology Visualization (bottom left)
        self.ax_grid = self.fig.add_subplot(gs[2:, :2])
        self.ax_grid.set_title('Grid Topology (68-Bus System)')
        self.ax_grid.set_aspect('equal')
        
        # 6. Control Panel (bottom right)
        self.ax_controls = self.fig.add_subplot(gs[2:, 2:])
        self.ax_controls.set_title('Agent Controls & Metrics')
        self.ax_controls.axis('off')
        
        # Initialize plots
        self._init_plots()
        
        # Add control widgets
        self._add_controls()
        
        # Set up animation
        self.animation = animation.FuncAnimation(
            self.fig, self._update_plots, interval=100, blit=False
        )
    
    def _init_plots(self):
        """Initialize plot elements."""
        # Frequency plot lines
        self.freq_mean_line, = self.ax_freq.plot([], [], 'b-', linewidth=2, label='Mean Frequency')
        self.freq_std_fill = None
        
        # Power plot lines
        self.generation_line, = self.ax_power.plot([], [], 'g-', linewidth=2, label='Total Generation')
        self.load_line, = self.ax_power.plot([], [], 'r-', linewidth=2, label='Total Load')
        self.ax_power.legend()
        
        # Agent output lines
        colors = ['blue', 'green', 'red']
        labels = ['Batteries (5)', 'Gas Plants (8)', 'Demand Response (7)']
        self.agent_lines = []
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            line, = self.ax_agents.plot([], [], color=color, linewidth=2, label=label)
            self.agent_lines.append(line)
        self.ax_agents.legend()
        
        # Reward plot lines
        self.reward_line, = self.ax_reward.plot([], [], 'purple', linewidth=2, label='Reward')
        self.ax_reward_twin = self.ax_reward.twinx()
        self.violations_line, = self.ax_reward_twin.plot([], [], 'orange', linewidth=2, label='Safety Violations')
        self.ax_reward_twin.set_ylabel('Safety Violations', color='orange')
        
        # Grid topology
        self._draw_grid_topology()
    
    def _draw_grid_topology(self):
        """Draw the grid topology visualization."""
        # Create a circular layout for the 68 buses
        n_buses = self.env.n_buses
        angles = np.linspace(0, 2*np.pi, n_buses, endpoint=False)
        
        # Bus positions
        radius = 1.0
        bus_x = radius * np.cos(angles)
        bus_y = radius * np.sin(angles)
        
        # Draw buses
        self.ax_grid.scatter(bus_x, bus_y, c='lightblue', s=30, alpha=0.7, label='Buses')
        
        # Highlight agent buses
        agent_buses = self.env.agent_bus_mapping.cpu().numpy()
        agent_colors = ['red'] * 5 + ['green'] * 8 + ['blue'] * 7  # Battery, Gas, DR
        
        for i, bus_idx in enumerate(agent_buses):
            self.ax_grid.scatter(bus_x[bus_idx], bus_y[bus_idx], 
                               c=agent_colors[i], s=100, marker='s', 
                               alpha=0.8, edgecolors='black', linewidth=1)
        
        # Draw some connections (simplified)
        for i in range(0, n_buses, 4):  # Every 4th bus
            next_bus = (i + 1) % n_buses
            self.ax_grid.plot([bus_x[i], bus_x[next_bus]], 
                            [bus_y[i], bus_y[next_bus]], 
                            'gray', alpha=0.3, linewidth=0.5)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=8, label='Regular Bus'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                      markersize=8, label='Battery'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                      markersize=8, label='Gas Plant'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', 
                      markersize=8, label='Demand Response')
        ]
        self.ax_grid.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        self.ax_grid.set_xlim(-1.2, 1.2)
        self.ax_grid.set_ylim(-1.2, 1.2)
    
    def _add_controls(self):
        """Add control widgets to the dashboard."""
        # Control buttons
        ax_start = plt.axes([0.55, 0.02, 0.08, 0.04])
        ax_stop = plt.axes([0.64, 0.02, 0.08, 0.04])
        ax_reset = plt.axes([0.73, 0.02, 0.08, 0.04])
        ax_step = plt.axes([0.82, 0.02, 0.08, 0.04])
        
        self.btn_start = Button(ax_start, 'Start')
        self.btn_stop = Button(ax_stop, 'Stop')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_step = Button(ax_step, 'Step')
        
        self.btn_start.on_clicked(self._start_simulation)
        self.btn_stop.on_clicked(self._stop_simulation)
        self.btn_reset.on_clicked(self._reset_environment)
        self.btn_step.on_clicked(self._step_once)
        
        # Control mode toggle button
        ax_mode = plt.axes([0.55, 0.08, 0.35, 0.04])
        self.btn_mode = Button(ax_mode, self._get_mode_button_text())
        self.btn_mode.on_clicked(self._toggle_control_mode)
        
        # Agent action sliders (simplified - show first 5 agents)
        # Only active in manual mode
        self.agent_sliders = []
        for i in range(min(5, self.env.n_agents)):
            ax_slider = plt.axes([0.55, 0.35 - i*0.04, 0.35, 0.02])
            slider = Slider(ax_slider, f'Agent {i+1}', -1.0, 1.0, valinit=0.0)
            slider.on_changed(lambda val, idx=i: self._update_agent_action(idx, val))
            self.agent_sliders.append(slider)
    
    def _get_mode_button_text(self):
        """Get the text for the control mode button."""
        mode_icons = {
            'manual': '🎮',
            'random': '🎲',
            'ai': '🤖'
        }
        mode_names = {
            'manual': 'Manual',
            'random': 'Random',
            'ai': 'AI Agent'
        }
        
        icon = mode_icons.get(self.control_mode, '')
        name = mode_names.get(self.control_mode, 'Unknown')
        return f'{icon} Mode: {name} (click to change)'
    
    def _toggle_control_mode(self, event):
        """Toggle between control modes."""
        modes = ['manual', 'random']
        if self.agent is not None:
            modes.append('ai')
        
        current_idx = modes.index(self.control_mode)
        next_idx = (current_idx + 1) % len(modes)
        self.control_mode = modes[next_idx]
        
        # Update button text
        self.btn_mode.label.set_text(self._get_mode_button_text())
        
        print(f"Control mode changed to: {self.control_mode}")
    
    def _update_agent_action(self, agent_idx, value):
        """Update action for a specific agent."""
        self.current_actions[agent_idx] = value
    
    def _start_simulation(self, event):
        """Start automatic simulation."""
        self.is_running = True
    
    def _stop_simulation(self, event):
        """Stop automatic simulation."""
        self.is_running = False
    
    def _reset_environment(self, event):
        """Reset the environment."""
        self.obs, self.info = self.env.reset()
        self.current_step = 0
        self._clear_history()
        self._update_history()
    
    def _step_once(self, event):
        """Execute one environment step."""
        self._step_environment()
    
    def _step_environment(self):
        """Execute one step of the environment."""
        # Get actions based on control mode
        if self.control_mode == 'ai' and self.agent is not None:
            # Use trained AI agent
            state = self.env.get_full_state().cpu().numpy()
            actions, _, _ = self.agent.select_action(self.obs, state, deterministic=True)
        elif self.control_mode == 'random':
            # Random actions
            actions = np.random.uniform(-0.5, 0.5, size=self.env.n_agents)
        else:
            # Manual control from sliders
            actions = self.current_actions.copy()
        
        # Step environment
        self.obs, reward, terminated, truncated, self.info = self.env.step(actions)
        self.current_step += 1
        
        # Update history
        self._update_history()
        
        # Reset if terminated
        if terminated or truncated:
            self.obs, self.info = self.env.reset()
            self.current_step = 0
    
    def _update_history(self):
        """Update data history for plotting."""
        self.time_history.append(self.current_step)
        
        # Frequency data
        frequencies = self.env.frequencies.cpu().numpy()
        self.frequency_history.append({
            'mean': np.mean(frequencies),
            'std': np.std(frequencies),
            'min': np.min(frequencies),
            'max': np.max(frequencies)
        })
        
        # Power data
        total_generation = self.info['total_generation']
        total_load = self.info['total_load']
        self.generation_history.append(total_generation)
        self.load_history.append(total_load)
        
        # Reward and violations
        reward = self.env._calculate_reward()
        self.reward_history.append(reward)
        self.safety_violations_history.append(self.info['safety_violations'])
        
        # Agent outputs by type
        agent_outputs = self.env.generator_outputs.cpu().numpy()
        battery_total = np.sum(agent_outputs[:5])
        gas_total = np.sum(agent_outputs[5:13])
        dr_total = np.sum(agent_outputs[13:])
        
        self.agent_outputs_history[0].append(battery_total)
        self.agent_outputs_history[1].append(gas_total)
        self.agent_outputs_history[2].append(dr_total)
    
    def _clear_history(self):
        """Clear all history data."""
        self.time_history.clear()
        self.frequency_history.clear()
        self.generation_history.clear()
        self.load_history.clear()
        self.reward_history.clear()
        self.safety_violations_history.clear()
        for history in self.agent_outputs_history.values():
            history.clear()
    
    def _update_plots(self, frame):
        """Update all plots (called by animation)."""
        if self.is_running:
            self._step_environment()
        
        if len(self.time_history) < 2:
            return []
        
        time_data = list(self.time_history)
        
        # Update frequency plot
        freq_means = [f['mean'] for f in self.frequency_history]
        freq_stds = [f['std'] for f in self.frequency_history]
        
        self.freq_mean_line.set_data(time_data, freq_means)
        
        # Add frequency standard deviation as fill
        if self.freq_std_fill is not None:
            self.freq_std_fill.remove()
        
        freq_upper = [m + s for m, s in zip(freq_means, freq_stds)]
        freq_lower = [m - s for m, s in zip(freq_means, freq_stds)]
        self.freq_std_fill = self.ax_freq.fill_between(
            time_data, freq_lower, freq_upper, alpha=0.3, color='blue'
        )
        
        # Update power plot
        self.generation_line.set_data(time_data, list(self.generation_history))
        self.load_line.set_data(time_data, list(self.load_history))
        
        # Update agent outputs plot
        for i, line in enumerate(self.agent_lines):
            if i < len(self.agent_outputs_history):
                line.set_data(time_data, list(self.agent_outputs_history[i]))
        
        # Update reward plot
        self.reward_line.set_data(time_data, list(self.reward_history))
        self.violations_line.set_data(time_data, list(self.safety_violations_history))
        
        # Auto-scale axes
        for ax in [self.ax_freq, self.ax_power, self.ax_agents, self.ax_reward]:
            ax.relim()
            ax.autoscale_view()
        
        self.ax_reward_twin.relim()
        self.ax_reward_twin.autoscale_view()
        
        # Update control panel text
        self._update_metrics_display()
        
        return []
    
    def _update_metrics_display(self):
        """Update the metrics display in the control panel."""
        self.ax_controls.clear()
        self.ax_controls.axis('off')
        
        # Control mode indicator
        mode_icons = {'manual': '🎮', 'random': '🎲', 'ai': '🤖'}
        mode_names = {'manual': 'Manual Control', 'random': 'Random Actions', 'ai': 'AI Agent'}
        mode_icon = mode_icons.get(self.control_mode, '')
        mode_name = mode_names.get(self.control_mode, 'Unknown')
        
        # Current metrics
        metrics_text = f"""
CONTROL MODE: {mode_icon} {mode_name}
{'=' * 40}

CURRENT METRICS
Step: {self.current_step}
Time: {self.info.get('current_hour', 0):.1f}h, Day {self.info.get('current_day', 0):.0f}

SYSTEM STATUS
Mean Frequency: {self.info['mean_frequency']:.3f} Hz
Freq Deviation: {self.info['system_freq_deviation']:.4f} Hz
Safety Violations: {self.info['safety_violations']:.0f}

POWER BALANCE
Total Generation: {self.info['total_generation']:.1f} MW
Total Load: {self.info['total_load']:.1f} MW
Imbalance: {self.info['total_generation'] - self.info['total_load']:.1f} MW

AGENT STATUS
Batteries (2): {np.sum(self.env.generator_outputs[:2].cpu().numpy()):.1f} MW
Gas Plants (5): {np.sum(self.env.generator_outputs[2:7].cpu().numpy()):.1f} MW
Demand Response (3): {np.sum(self.env.generator_outputs[7:].cpu().numpy()):.1f} MW

CONTINGENCY
Active: {self.info['contingency_active']}
        """
        
        self.ax_controls.text(0.05, 0.95, metrics_text, transform=self.ax_controls.transAxes,
                            fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    def show(self):
        """Display the dashboard."""
        plt.show()
    
    def save_screenshot(self, filename='dashboard_screenshot.png'):
        """Save a screenshot of the current dashboard."""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Dashboard screenshot saved as {filename}")


def main():
    """Run the dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Power Grid Dashboard')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained MAPPO model (e.g., checkpoints/best_model.pt)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cpu, cuda, or auto')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("POWER GRID DASHBOARD")
    print("=" * 60)
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Device: {device}")
    
    # Create environment
    env = PowerGridEnv(device=device)
    
    # Create and show dashboard
    dashboard = PowerGridDashboard(env, model_path=args.model)
    
    print("\n" + "=" * 60)
    print("DASHBOARD CONTROLS")
    print("=" * 60)
    print("- Start/Stop: Run/pause automatic simulation")
    print("- Reset: Reset environment to initial state")
    print("- Step: Execute single time step")
    print("- Mode Toggle: Switch between Manual/Random/AI control")
    print("- Sliders: Manual control for first 5 agents")
    print("=" * 60)
    
    if dashboard.agent is not None:
        print("\n🤖 AI Agent loaded! Switch to AI mode to watch it control the grid.")
    else:
        print("\n💡 Tip: Load a trained model with --model checkpoints/best_model.pt")
    
    print("\nStarting dashboard...\n")
    dashboard.show()


if __name__ == "__main__":
    main()
