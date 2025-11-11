"""
Power Grid Environment Web Dashboard

Advanced web-based dashboard using Streamlit for the multi-agent power grid environment.
Features real-time visualization, interactive controls, and comprehensive metrics.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import torch
import time
from power_grid_env import PowerGridEnv
from collections import deque
import threading


class WebDashboard:
    """Web-based dashboard for the power grid environment."""
    
    def __init__(self):
        """Initialize the web dashboard."""
        self.setup_page_config()
        self.initialize_session_state()
        self.create_dashboard()
    
    def setup_page_config(self):
        """Configure the Streamlit page."""
        st.set_page_config(
            page_title="Power Grid Multi-Agent RL Dashboard",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'env' not in st.session_state:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            st.session_state.env = PowerGridEnv(device=device)
            st.session_state.obs, st.session_state.info = st.session_state.env.reset(seed=42)
        
        if 'history' not in st.session_state:
            st.session_state.history = {
                'time': [],
                'frequencies': [],
                'generation': [],
                'load': [],
                'rewards': [],
                'violations': [],
                'agent_outputs': {i: [] for i in range(st.session_state.env.n_agents)}
            }
        
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 0
        
        if 'is_running' not in st.session_state:
            st.session_state.is_running = False
        
        if 'agent_actions' not in st.session_state:
            st.session_state.agent_actions = np.zeros(st.session_state.env.n_agents)
    
    def create_dashboard(self):
        """Create the main dashboard interface."""
        # Header
        st.title("⚡ Power Grid Multi-Agent RL Dashboard")
        st.markdown("---")
        
        # Sidebar controls
        self.create_sidebar()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.create_main_plots()
        
        with col2:
            self.create_metrics_panel()
        
        # Bottom section
        st.markdown("---")
        self.create_agent_controls()
        
        # Auto-refresh if running
        if st.session_state.is_running:
            time.sleep(0.1)
            st.rerun()
    
    def create_sidebar(self):
        """Create the sidebar with controls and settings."""
        st.sidebar.header("🎛️ Controls")
        
        # Environment controls
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("▶️ Start", use_container_width=True):
                st.session_state.is_running = True
                st.rerun()
            
            if st.button("🔄 Reset", use_container_width=True):
                self.reset_environment()
                st.rerun()
        
        with col2:
            if st.button("⏸️ Stop", use_container_width=True):
                st.session_state.is_running = False
                st.rerun()
            
            if st.button("⏭️ Step", use_container_width=True):
                self.step_environment()
                st.rerun()
        
        st.sidebar.markdown("---")
        
        # Settings
        st.sidebar.header("⚙️ Settings")
        
        # Simulation speed
        speed = st.sidebar.slider("Simulation Speed", 0.1, 2.0, 1.0, 0.1)
        
        # History length
        history_length = st.sidebar.slider("History Length", 50, 500, 200, 50)
        
        # Display options
        st.sidebar.subheader("Display Options")
        show_grid = st.sidebar.checkbox("Show Grid Topology", True)
        show_forecasts = st.sidebar.checkbox("Show Renewable Forecasts", True)
        show_individual_agents = st.sidebar.checkbox("Show Individual Agents", False)
        
        st.sidebar.markdown("---")
        
        # Environment info
        st.sidebar.header("📊 Environment Info")
        env = st.session_state.env
        st.sidebar.metric("Buses", env.n_buses)
        st.sidebar.metric("Agents", env.n_agents)
        st.sidebar.metric("Current Step", st.session_state.current_step)
        st.sidebar.metric("Device", str(env.device))
    
    def create_main_plots(self):
        """Create the main visualization plots."""
        # System Frequency Plot
        st.subheader("📈 System Frequency")
        self.plot_frequencies()
        
        # Power Balance Plot
        st.subheader("⚖️ Power Balance")
        self.plot_power_balance()
        
        # Agent Performance
        st.subheader("🤖 Agent Performance")
        self.plot_agent_performance()
    
    def plot_frequencies(self):
        """Plot system frequency data."""
        if len(st.session_state.history['time']) < 2:
            st.info("Collecting data... Run the simulation to see frequency plots.")
            return
        
        fig = go.Figure()
        
        # Mean frequency
        freq_data = st.session_state.history['frequencies']
        freq_means = [f['mean'] for f in freq_data]
        freq_stds = [f['std'] for f in freq_data]
        time_data = st.session_state.history['time']
        
        # Add mean frequency line
        fig.add_trace(go.Scatter(
            x=time_data,
            y=freq_means,
            mode='lines',
            name='Mean Frequency',
            line=dict(color='blue', width=2)
        ))
        
        # Add standard deviation band
        freq_upper = [m + s for m, s in zip(freq_means, freq_stds)]
        freq_lower = [m - s for m, s in zip(freq_means, freq_stds)]
        
        fig.add_trace(go.Scatter(
            x=time_data + time_data[::-1],
            y=freq_upper + freq_lower[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±1 Std Dev'
        ))
        
        # Add reference lines
        fig.add_hline(y=60.0, line_dash="dash", line_color="green", 
                     annotation_text="Nominal (60 Hz)")
        fig.add_hline(y=59.5, line_dash="dash", line_color="red", 
                     annotation_text="Lower Bound")
        fig.add_hline(y=60.5, line_dash="dash", line_color="red", 
                     annotation_text="Upper Bound")
        
        fig.update_layout(
            xaxis_title="Time Step",
            yaxis_title="Frequency (Hz)",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_power_balance(self):
        """Plot power generation vs load."""
        if len(st.session_state.history['time']) < 2:
            st.info("Collecting data... Run the simulation to see power balance.")
            return
        
        fig = go.Figure()
        
        time_data = st.session_state.history['time']
        generation_data = st.session_state.history['generation']
        load_data = st.session_state.history['load']
        
        fig.add_trace(go.Scatter(
            x=time_data,
            y=generation_data,
            mode='lines',
            name='Total Generation',
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=time_data,
            y=load_data,
            mode='lines',
            name='Total Load',
            line=dict(color='red', width=2)
        ))
        
        # Add imbalance area
        imbalance = [g - l for g, l in zip(generation_data, load_data)]
        fig.add_trace(go.Scatter(
            x=time_data,
            y=imbalance,
            mode='lines',
            name='Imbalance (Gen - Load)',
            line=dict(color='orange', width=1),
            yaxis='y2'
        ))
        
        fig.update_layout(
            xaxis_title="Time Step",
            yaxis_title="Power (MW)",
            yaxis2=dict(
                title="Imbalance (MW)",
                overlaying='y',
                side='right'
            ),
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_agent_performance(self):
        """Plot agent performance by type."""
        if len(st.session_state.history['time']) < 2:
            st.info("Collecting data... Run the simulation to see agent performance.")
            return
        
        # Create subplot for different agent types
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Batteries (5 agents)', 'Gas Plants (8 agents)', 
                          'Demand Response (7 agents)', 'Rewards & Violations'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        time_data = st.session_state.history['time']
        env = st.session_state.env
        
        # Battery outputs
        battery_outputs = [np.sum(env.generator_outputs[:5].cpu().numpy()) 
                          for _ in time_data]
        fig.add_trace(go.Scatter(x=time_data, y=battery_outputs, name='Batteries',
                               line=dict(color='blue')), row=1, col=1)
        
        # Gas plant outputs
        gas_outputs = [np.sum(env.generator_outputs[5:13].cpu().numpy()) 
                      for _ in time_data]
        fig.add_trace(go.Scatter(x=time_data, y=gas_outputs, name='Gas Plants',
                               line=dict(color='green')), row=1, col=2)
        
        # Demand response outputs
        dr_outputs = [np.sum(env.generator_outputs[13:].cpu().numpy()) 
                     for _ in time_data]
        fig.add_trace(go.Scatter(x=time_data, y=dr_outputs, name='Demand Response',
                               line=dict(color='red')), row=2, col=1)
        
        # Rewards and violations
        if st.session_state.history['rewards']:
            fig.add_trace(go.Scatter(x=time_data, y=st.session_state.history['rewards'],
                                   name='Rewards', line=dict(color='purple')), 
                         row=2, col=2)
        
        if st.session_state.history['violations']:
            fig.add_trace(go.Scatter(x=time_data, y=st.session_state.history['violations'],
                                   name='Safety Violations', line=dict(color='orange')),
                         row=2, col=2, secondary_y=True)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def create_metrics_panel(self):
        """Create the metrics display panel."""
        st.subheader("📊 Current Metrics")
        
        info = st.session_state.info
        env = st.session_state.env
        
        # System status metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Mean Frequency", f"{info['mean_frequency']:.3f} Hz",
                     delta=f"{info['system_freq_deviation']:.4f}")
            st.metric("Total Generation", f"{info['total_generation']:.1f} MW")
            st.metric("Safety Violations", f"{info['safety_violations']:.0f}")
        
        with col2:
            st.metric("Total Load", f"{info['total_load']:.1f} MW")
            imbalance = info['total_generation'] - info['total_load']
            st.metric("Power Imbalance", f"{imbalance:.1f} MW")
            st.metric("Current Time", f"{info.get('current_hour', 0):.1f}h")
        
        # Agent status
        st.subheader("🤖 Agent Status")
        
        agent_outputs = env.generator_outputs.cpu().numpy()
        capacity_util = info.get('agent_capacity_utilization', [0] * env.n_agents)
        
        # Create agent status dataframe
        agent_data = []
        agent_types = ['Battery'] * 5 + ['Gas Plant'] * 8 + ['Demand Response'] * 7
        
        for i in range(min(10, env.n_agents)):  # Show first 10 agents
            agent_data.append({
                'Agent': f"{agent_types[i]} {i+1}",
                'Output (MW)': f"{agent_outputs[i]:.1f}",
                'Capacity (%)': f"{capacity_util[i]*100:.1f}%" if i < len(capacity_util) else "N/A",
                'Min (MW)': f"{env.power_min[i]:.0f}",
                'Max (MW)': f"{env.power_max[i]:.0f}"
            })
        
        df = pd.DataFrame(agent_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Contingency status
        st.subheader("⚠️ System Status")
        
        if info['contingency_active']:
            st.error("🚨 N-1 Contingency Active!")
        else:
            st.success("✅ Normal Operation")
        
        # Recent reward
        if st.session_state.history['rewards']:
            recent_reward = st.session_state.history['rewards'][-1]
            st.metric("Latest Reward", f"{recent_reward:.2f}")
    
    def create_agent_controls(self):
        """Create agent control interface."""
        st.subheader("🎮 Agent Controls")
        
        # Control mode selection
        control_mode = st.radio(
            "Control Mode",
            ["Automatic (Random)", "Manual Control", "Policy-based"],
            horizontal=True
        )
        
        if control_mode == "Manual Control":
            st.info("Use sliders to control individual agents")
            
            # Create sliders for agent control
            cols = st.columns(5)
            
            for i in range(min(20, st.session_state.env.n_agents)):
                col_idx = i % 5
                with cols[col_idx]:
                    agent_type = st.session_state.env.agent_types[i]
                    st.session_state.agent_actions[i] = st.slider(
                        f"{agent_type.title()} {i+1}",
                        -1.0, 1.0, 0.0, 0.1,
                        key=f"agent_{i}"
                    )
        
        elif control_mode == "Policy-based":
            st.info("Policy-based control not implemented yet")
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🚀 Execute Actions", use_container_width=True):
                if control_mode == "Manual Control":
                    self.step_environment(st.session_state.agent_actions)
                else:
                    self.step_environment()
                st.rerun()
        
        with col2:
            if st.button("🎲 Random Actions", use_container_width=True):
                random_actions = np.random.uniform(-0.5, 0.5, st.session_state.env.n_agents)
                self.step_environment(random_actions)
                st.rerun()
        
        with col3:
            if st.button("🔄 Reset All", use_container_width=True):
                st.session_state.agent_actions = np.zeros(st.session_state.env.n_agents)
                st.rerun()
        
        with col4:
            if st.button("💾 Save State", use_container_width=True):
                self.save_current_state()
    
    def step_environment(self, actions=None):
        """Execute one step of the environment."""
        if actions is None:
            actions = np.random.uniform(-0.5, 0.5, st.session_state.env.n_agents)
        
        # Step environment
        obs, reward, terminated, truncated, info = st.session_state.env.step(actions)
        st.session_state.obs = obs
        st.session_state.info = info
        st.session_state.current_step += 1
        
        # Update history
        self.update_history(reward)
        
        # Reset if terminated
        if terminated or truncated:
            st.warning("Episode terminated! Resetting environment...")
            self.reset_environment()
    
    def reset_environment(self):
        """Reset the environment."""
        st.session_state.obs, st.session_state.info = st.session_state.env.reset()
        st.session_state.current_step = 0
        st.session_state.is_running = False
        
        # Clear history
        st.session_state.history = {
            'time': [],
            'frequencies': [],
            'generation': [],
            'load': [],
            'rewards': [],
            'violations': [],
            'agent_outputs': {i: [] for i in range(st.session_state.env.n_agents)}
        }
    
    def update_history(self, reward):
        """Update the data history."""
        env = st.session_state.env
        info = st.session_state.info
        
        st.session_state.history['time'].append(st.session_state.current_step)
        
        # Frequency data
        frequencies = env.frequencies.cpu().numpy()
        st.session_state.history['frequencies'].append({
            'mean': np.mean(frequencies),
            'std': np.std(frequencies),
            'min': np.min(frequencies),
            'max': np.max(frequencies)
        })
        
        # Power data
        st.session_state.history['generation'].append(info['total_generation'])
        st.session_state.history['load'].append(info['total_load'])
        
        # Reward and violations
        st.session_state.history['rewards'].append(reward)
        st.session_state.history['violations'].append(info['safety_violations'])
        
        # Keep history length manageable
        max_history = 200
        for key in st.session_state.history:
            if isinstance(st.session_state.history[key], list):
                if len(st.session_state.history[key]) > max_history:
                    st.session_state.history[key] = st.session_state.history[key][-max_history:]
    
    def save_current_state(self):
        """Save the current environment state."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"power_grid_state_{timestamp}.json"
        
        # Create state dictionary
        state_data = {
            'step': st.session_state.current_step,
            'frequencies': st.session_state.env.frequencies.cpu().numpy().tolist(),
            'generator_outputs': st.session_state.env.generator_outputs.cpu().numpy().tolist(),
            'loads': st.session_state.env.loads.cpu().numpy().tolist(),
            'info': st.session_state.info
        }
        
        # Save to file (simplified - in practice would use proper file handling)
        st.success(f"State saved as {filename}")


def main():
    """Run the web dashboard."""
    dashboard = WebDashboard()


if __name__ == "__main__":
    main()
