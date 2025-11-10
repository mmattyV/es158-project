# Multi-Agent Reinforcement Learning for Power Grid Energy Flow Balancing

This project implements a Multi-Agent Proximal Policy Optimization (MAPPO) algorithm for coordinating energy resources in a 68-bus power grid with 20 agents.

## System Architecture

### Environment (`power_grid_env.py`)
- **68-bus power grid** with **20 cooperative agents**
  - 5 Battery Energy Storage Systems (BESS) - 50 MW/min ramp rate
  - 8 Gas-fired power plants - 10 MW/min ramp rate
  - 7 Demand Response (DR) units - 5 MW/min ramp rate

- **Observation Space**: 15-dimensional local observations per agent
  - Local bus frequency and load
  - Own generator output
  - System-wide frequency deviation
  - 5 nearby bus frequencies
  - 3-step renewable forecasts
  - Time features (hour, day)
  - Capacity utilization

- **State Space**: 140-dimensional full state (for centralized critic)
  - 68 bus frequencies
  - 20 generator outputs
  - 14 renewable generation values
  - 30 largest loads
  - 8 time/pattern features

- **Action Space**: Continuous actions in [-1, 1] per agent
  - Scaled by agent-specific ramp rates

- **Reward Function**: Shared global reward
  ```
  R = -(1000 * Σ(Δf²) + ΣC_i|ΔP_i| + 0.1ΣW_i(ΔP_i²) + 10000 * violations)
  ```
  - Frequency deviation penalty
  - Operational costs
  - Wear-and-tear costs
  - Safety constraint violations

- **Key Features**:
  - Simplified swing equation dynamics
  - 2-step SCADA communication delay
  - Stochastic loads and renewable variations
  - N-1 contingency events
  - 5-step renewable forecasts

### MAPPO Algorithm

**Actor Network** (`networks.py`):
- Input: 15-dimensional local observation
- Architecture: [15 → 128 → 128 → 1] with LayerNorm and ReLU
- Output: Gaussian policy (mean and std)
- Decentralized execution

**Critic Network** (`networks.py`):
- Input: 140-dimensional full state
- Architecture: [140 → 256 → 256 → 1] with LayerNorm and ReLU
- Output: Value estimate
- Centralized training

**Training** (`mappo.py`, `train.py`):
- On-policy rollout collection
- Generalized Advantage Estimation (GAE)
- PPO clipped objective
- Entropy bonus for exploration
- Gradient clipping for stability

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Test Environment (Random Agents)
```bash
python main.py
```
This will:
- Test the environment with random agents
- Display environment features
- Save results to `test_results.png`

### 2. Train MAPPO Agent
```bash
python train.py
```
Training configuration:
- **Episodes**: 1000
- **Max steps per episode**: 500
- **Buffer size**: 2048 steps
- **Batch size**: 256
- **PPO epochs**: 10
- **Learning rates**: Actor 3e-4, Critic 1e-3
- **Discount factor (γ)**: 0.99
- **GAE lambda (λ)**: 0.95

Outputs:
- Model checkpoints saved to `checkpoints/`
- Best model: `checkpoints/best_model.pt`
- Final model: `checkpoints/final_model.pt`
- Training statistics: `checkpoints/training_stats.json`
- Training curves: `checkpoints/training_curves.png`

### 3. Evaluate Trained Agent
```bash
# Basic evaluation
python evaluate.py --model_path checkpoints/best_model.pt --n_episodes 10

# With visualization
python evaluate.py --model_path checkpoints/best_model.pt --visualize

# Compare with random baseline
python evaluate.py --model_path checkpoints/best_model.pt --compare

# Render during evaluation
python evaluate.py --model_path checkpoints/best_model.pt --render
```

## File Structure

```
main/
├── power_grid_env.py      # Power grid environment
├── networks.py            # Actor and Critic networks
├── buffer.py              # Rollout buffer with GAE
├── mappo.py               # MAPPO agent implementation
├── train.py               # Training script
├── evaluate.py            # Evaluation and visualization
├── main.py                # Environment testing
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lr_actor` | 3e-4 | Actor learning rate |
| `lr_critic` | 1e-3 | Critic learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_epsilon` | 0.2 | PPO clipping parameter |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |
| `value_coef` | 0.5 | Value loss coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |
| `buffer_size` | 2048 | Rollout buffer size |
| `batch_size` | 256 | Minibatch size |
| `n_epochs` | 10 | PPO update epochs |

## Expected Results

The trained agent should:
- ✓ Maintain system frequency near 60 Hz (within [59.5, 60.5] Hz bounds)
- ✓ Minimize safety violations
- ✓ Balance generation and load efficiently
- ✓ Coordinate different agent types (batteries for fast response, gas for sustained power)
- ✓ Outperform random baseline by significant margin
- ✓ Handle stochastic loads and renewable variations
- ✓ Respond appropriately to contingency events

## Algorithm Details

**MAPPO (Multi-Agent PPO)** with:
1. **Centralized Training, Decentralized Execution (CTDE)**
   - Critic uses full state during training
   - Actor uses local observations during execution

2. **Proximal Policy Optimization**
   - Clipped surrogate objective for stable updates
   - Multiple epochs over collected data

3. **Generalized Advantage Estimation**
   - Bias-variance tradeoff via λ parameter
   - Reduces variance in advantage estimates

4. **Shared Reward Signal**
   - All agents receive same global reward
   - Encourages cooperative behavior

## Troubleshooting

**Training is unstable:**
- Reduce learning rates
- Increase buffer size
- Decrease batch size
- Adjust clip_epsilon

**Agent learns slowly:**
- Increase buffer size for more diverse experience
- Adjust entropy coefficient for more exploration
- Check reward scaling

**GPU out of memory:**
- Reduce batch size
- Use smaller network architectures
- Switch to CPU: `device='cpu'`

## Future Improvements

- [ ] Add prioritized experience replay
- [ ] Implement attention mechanism for agent communication
- [ ] Add curriculum learning for harder scenarios
- [ ] Include more realistic power flow constraints
- [ ] Multi-level hierarchical control

