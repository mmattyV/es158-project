# Power Grid Dashboard

Interactive dashboards for visualizing and controlling the multi-agent power grid environment.

## 🚀 Quick Start

### Option 1: Matplotlib Dashboard (Recommended for Development)
```bash
cd main/
python dashboard.py
```

**NEW: Evaluate Trained AI Agent on Dashboard** 🤖
```bash
cd main/
python dashboard.py --model checkpoints/best_model.pt
```

### Option 2: Web Dashboard (Best for Presentations)
```bash
cd main/
streamlit run web_dashboard.py
```

### Option 3: Use the Launcher
```bash
cd main/
python run_dashboard.py --type matplotlib  # or 'web' or 'both'
```

## 📊 Features

### Matplotlib Dashboard (`dashboard.py`)
- **Real-time Visualization**: Live plots updating automatically
- **Interactive Controls**: Start/Stop/Reset/Step buttons
- **🤖 AI Agent Evaluation**: Load and watch trained MAPPO agents control the grid
- **Control Mode Toggle**: Switch between Manual/Random/AI control
- **Agent Control Sliders**: Manual control of first 5 agents
- **Grid Topology**: 68-bus system visualization with agent locations
- **Comprehensive Metrics**: Frequencies, power balance, rewards, violations

### Web Dashboard (`web_dashboard.py`)
- **Modern Web Interface**: Clean, responsive Streamlit interface
- **Advanced Plots**: Interactive Plotly charts with zoom/pan
- **Full Agent Control**: Control all 20 agents individually
- **Real-time Metrics**: Live updating system status
- **Data Export**: Save current state and screenshots

## 🎛️ Dashboard Components

### 1. System Frequency Plot
- **Mean frequency** over time with standard deviation bands
- **Reference lines** for nominal (60 Hz) and safety bounds (59.5-60.5 Hz)
- **Real-time updates** showing frequency stability

### 2. Power Balance Visualization
- **Total generation vs total load** comparison
- **Power imbalance** tracking (Generation - Load)
- **Color-coded lines** for easy identification

### 3. Agent Performance by Type
- **Batteries (5 agents)**: Fast response, limited capacity
- **Gas Plants (8 agents)**: High capacity, slower response
- **Demand Response (7 agents)**: Load reduction capabilities

### 4. Grid Topology
- **68-bus circular layout** with connections
- **Agent locations** highlighted by type:
  - 🔴 Red squares: Batteries
  - 🟢 Green squares: Gas Plants  
  - 🔵 Blue squares: Demand Response
- **Connection lines** showing grid structure

### 5. Real-time Metrics
- **System Status**: Frequency deviation, safety violations
- **Power Balance**: Generation, load, imbalance
- **Agent Status**: Individual outputs and capacity utilization
- **Contingency Alerts**: N-1 contingency events

### 6. Interactive Controls
- **Simulation Control**: Start/Stop/Reset/Step
- **Agent Actions**: Manual control sliders for each agent
- **Control Modes**: Automatic, Manual, Policy-based
- **Settings**: Simulation speed, history length, display options

## 🎮 How to Use

### Basic Operation
1. **Start the dashboard** using one of the methods above
2. **Click "Start"** to begin automatic simulation with random actions
3. **Observe** the real-time plots and metrics
4. **Use "Stop"** to pause and **"Reset"** to restart

### Manual Control
1. **Stop** the automatic simulation
2. **Adjust agent sliders** to set desired actions (-1 to +1)
3. **Click "Step"** to execute one time step with your actions
4. **Observe** the impact on system frequency and power balance

### Advanced Features
- **Grid Topology**: See which buses have controllable agents
- **Safety Monitoring**: Watch for frequency bound violations
- **Contingency Events**: Observe N-1 contingency responses
- **Agent Economics**: Monitor cost-based reward function

## 📈 Key Metrics Explained

### Frequency Metrics
- **Mean Frequency**: Average across all 68 buses
- **System Freq Deviation**: Δf_sys = (1/68)Σ(f_k - 60)
- **Safety Violations**: Count of buses outside [59.5, 60.5] Hz

### Power Metrics
- **Total Generation**: Sum of all agent outputs
- **Total Load**: Sum of all bus loads (including stochastic variations)
- **Power Imbalance**: Generation - Load (should be near zero)

### Agent Metrics
- **Capacity Utilization**: Current output / Maximum capacity
- **Ramp Rate Limits**: 50 MW/min (batteries), 10 MW/min (gas), 5 MW/min (DR)
- **Cost Coefficients**: $5/MW (batteries), $50/MW (gas), $20/MW (DR)

### Reward Components
- **Frequency Penalty**: -1000 × Σ(f_k - 60)²
- **Agent Costs**: C_i × |action_i|
- **Wear Costs**: 0.1 × W_i × action_i²
- **Safety Penalties**: -10,000 per violation

## 🔧 Customization

### Modify Visualization
- Edit `dashboard.py` or `web_dashboard.py`
- Adjust plot colors, layouts, update intervals
- Add new metrics or visualizations

### Environment Parameters
- Modify `PowerGridEnv` parameters in the dashboard initialization
- Change number of agents, buses, or system parameters
- Adjust simulation speed and history length

### Control Logic
- Implement custom control policies in the dashboard
- Add new control modes (PID, MPC, RL policies)
- Integrate with trained RL models

## 🐛 Troubleshooting

### Common Issues
1. **Import Errors**: Install requirements with `pip install -r requirements.txt`
2. **Slow Performance**: Reduce history length or update interval
3. **Display Issues**: Check matplotlib backend or browser compatibility

### Performance Tips
- Use **matplotlib dashboard** for faster updates during development
- Use **web dashboard** for presentations and sharing
- Reduce history length for better performance
- Close other applications to free up resources

## 🎯 Use Cases

### Research & Development
- **Algorithm Testing**: Test RL algorithms in real-time
- **Parameter Tuning**: Adjust environment parameters interactively
- **Behavior Analysis**: Observe agent coordination patterns

### Education & Demonstrations
- **Power Systems Education**: Visualize grid dynamics
- **RL Demonstrations**: Show multi-agent learning in action
- **Interactive Learning**: Let students control agents manually

### Debugging & Analysis
- **Environment Debugging**: Verify environment behavior
- **Agent Analysis**: Understand individual agent contributions
- **System Monitoring**: Track safety violations and contingencies

---

**Happy Grid Controlling! ⚡🤖**
