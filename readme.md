# Multi-Agent Reinforcement Learning for Power Grid Frequency Regulation

Implementation of Multi-Agent Proximal Policy Optimization (MAPPO) for real-time frequency regulation in a 68-bus power grid with 20 heterogeneous agents. This project demonstrates centralized training with decentralized execution (CTDE) for cooperative multi-agent control of power systems.

**Project Report:** See [`doc/midterm/main.pdf`](doc/midterm/main.pdf) for complete technical details.

## 📋 Overview

### System Architecture

**Environment:** 68-bus power grid with stochastic loads, renewable generation, and N-1 contingencies

**Agents (20 total):**
- 5 Battery Energy Storage Systems (BESS) - 50 MW/min ramp rate, [0-100 MW]
- 8 Gas-fired power plants - 10 MW/min ramp rate, [50-500 MW]
- 7 Demand Response (DR) units - 5 MW/min ramp rate, [-200-0 MW]

**Key Features:**
- Swing equation dynamics with realistic power system physics
- Partial observability (15-dim local obs vs 140-dim global state)
- 2-second SCADA communication delays
- N-1 contingency events (probability 0.001/step)
- Stochastic renewable generation with 3-step forecasts

### Algorithm: MAPPO

**Architecture:**
- **Actor Network:** [15→128→128→1] with LayerNorm, Gaussian policy (shared across agents)
- **Critic Network:** [140→256→256→1] with LayerNorm (centralized value function)
- **Training:** PPO with GAE (λ=0.95), entropy bonus, gradient clipping

**Performance:**
- Outperforms random baseline by 60%
- Outperforms independent PPO by 35%
- 1000 episodes trained in ~3 hours (GPU)

---

## 🚀 Quick Start: Reproduce Results

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd es158-project

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- NumPy, Matplotlib, Gymnasium

### 2. Reproduce Training Results

```bash
cd main/

# Full training run (1000 episodes, ~3 hours on GPU)
python train.py

# Quick test (100 episodes, ~15 minutes)
python train.py --n_episodes 100 --eval_interval 25
```

**Outputs:**
- `checkpoints/best_model.pt` - Best performing model
- `checkpoints/final_model.pt` - Final model
- `checkpoints/training_results.png` - Training curves (Figure in report)
- `runs/mappo_*/` - TensorBoard logs

### 3. Monitor Training (Optional)

In a separate terminal while training:
```bash
tensorboard --logdir=runs
# Open http://localhost:6006
```

### 4. Evaluate Trained Model

```bash
# Basic evaluation
python evaluate.py --model_path checkpoints/best_model.pt --n_episodes 20

# With visualization and comparison to random baseline
python evaluate.py --model_path checkpoints/best_model.pt --visualize --compare
```

---

## 📊 Interactive Dashboard

Visualize the power grid and control agents in real-time.

### Option 1: Matplotlib Dashboard (Development)
```bash
cd main/
python dashboard.py

# Or with trained AI agent
python dashboard.py --model checkpoints/best_model.pt
```

### Option 2: Web Dashboard (Presentations)
```bash
cd main/
streamlit run web_dashboard.py
```

**Dashboard Features:**
- Real-time frequency and power balance plots
- 68-bus grid topology visualization
- Manual agent control (sliders for all 20 agents)
- AI agent evaluation mode
- Safety violation monitoring
- N-1 contingency response visualization

---

## 🔬 Detailed Usage

### Training Configuration

```bash
python train.py \
  --n_episodes 1000 \          # Number of training episodes
  --max_steps 500 \            # Max steps per episode
  --buffer_size 2048 \         # Rollout buffer size
  --batch_size 256 \           # Minibatch size
  --n_epochs 10 \              # PPO update epochs
  --lr_actor 3e-4 \            # Actor learning rate
  --lr_critic 1e-3 \           # Critic learning rate
  --gamma 0.99 \               # Discount factor
  --gae_lambda 0.95 \          # GAE lambda
  --eval_interval 50 \         # Episodes between evaluations
  --save_dir checkpoints \     # Model save directory
  --device auto                # Device: cpu, cuda, or auto
```

### Environment Testing

Test the environment before training:
```bash
cd main/
python main.py
```

This runs random agents and generates `test_results.png` with environment diagnostics.

### Evaluation Options

```bash
# Detailed evaluation with multiple metrics
python evaluate.py \
  --model_path checkpoints/best_model.pt \
  --n_episodes 50 \
  --visualize \               # Generate plots
  --compare \                 # Compare with random baseline
  --render                    # Print step-by-step output
```

---

## 📁 Repository Structure

```
es158-project/
├── main/                          # Source code
│   ├── power_grid_env.py         # 68-bus power grid environment (581 lines)
│   ├── networks.py                # Actor and Critic networks (203 lines)
│   ├── buffer.py                  # Rollout buffer with GAE (162 lines)
│   ├── mappo.py                   # MAPPO agent implementation (221 lines)
│   ├── train.py                   # Training script (356 lines)
│   ├── evaluate.py                # Evaluation and visualization (303 lines)
│   ├── main.py                    # Environment testing (527 lines)
│   ├── dashboard.py               # Interactive matplotlib dashboard (547 lines)
│   ├── web_dashboard.py           # Streamlit web dashboard (520 lines)
│   ├── run_dashboard.py           # Dashboard launcher (76 lines)
│   └── requirements.txt           # Python dependencies
├── checkpoints/                   # Saved models and results
│   ├── best_model.pt             # Best model checkpoint
│   ├── final_model.pt            # Final model checkpoint
│   └── training_results.png      # Training visualization
├── doc/                           # Documentation
│   ├── midterm/main.pdf          # Midterm report (5 pages)
│   └── proposal/Proposal.pdf     # Project proposal
├── runs/                          # TensorBoard logs
└── README.md                      # This file
```

---

## 🧪 Experimental Results

### Training Metrics (1000 Episodes)

| Metric | Value |
|--------|-------|
| Mean Episode Reward | -2.0×10⁷ |
| Best Episode Reward | -1.77×10⁷ |
| Mean Episode Length | 96.2 steps |
| Actor Loss (final) | 0.97 |
| Critic Loss (final) | 5.4×10¹² |
| Policy Entropy | 3.4 (stable) |
| Training Time | ~3 hours (GPU) |

### Performance Comparisons

| Method | Mean Reward | Episode Length |
|--------|-------------|----------------|
| **MAPPO (ours)** | -2.0×10⁷ | 96 steps |
| Independent PPO | -3.2×10⁷ | 68 steps |
| Random Baseline | -5.1×10⁷ | 35 steps |

**Key Findings:**
- MAPPO outperforms random by **60%**
- MAPPO outperforms independent learners by **35%**
- Centralized training provides significant coordination benefits
- 4 failure modes identified: insufficient response (40%), miscoordination (30%), oscillations (20%), capacity saturation (10%)

### Success Criteria Status

| Criterion | Target | Status |
|-----------|--------|--------|
| Frequency stability | 99% within 0.2 Hz | 🔶 Partial (performance gap remains) |
| Cost reduction | ≥25% vs PI-AGC | 🔜 PI-AGC baseline planned |
| Critical violations | Zero | 🔶 Occasional violations (~2%) |
| Coordination benefit | ≥15% vs independent | ✅ 35% improvement achieved |

---

## 🔧 Hyperparameters

### MAPPO Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lr_actor` | 3×10⁻⁴ | Actor learning rate |
| `lr_critic` | 1×10⁻³ | Critic learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda parameter |
| `clip_epsilon` | 0.2 | PPO clipping parameter |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |
| `value_coef` | 0.5 | Value loss coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |
| `buffer_size` | 2048 | Rollout buffer size |
| `batch_size` | 256 | Minibatch size |
| `n_epochs` | 10 | PPO update epochs |

### Environment Configuration

| Parameter | Value |
|-----------|-------|
| Buses | 68 |
| Agents | 20 (5 battery + 8 gas + 7 DR) |
| Time step | 2 seconds |
| Episode length | 500 steps max |
| Frequency bounds | [59.5, 60.5] Hz (safe), [58.5, 61.5] Hz (terminal) |
| Load range | [2000, 5000] MW |
| Contingency probability | 0.001 per step |
| SCADA delay | 1 time step (2 seconds) |

---

## 🛠️ Troubleshooting

### Installation Issues

**Import errors:**
```bash
pip install -r requirements.txt
```

**CUDA out of memory:**
```bash
# Use CPU
python train.py --device cpu

# Or reduce batch size
python train.py --batch_size 128
```

### Training Issues

**Training too slow:**
```bash
# Reduce episodes or steps
python train.py --n_episodes 200 --max_steps 300
```

**Training unstable:**
- Reduce learning rates: `--lr_actor 1e-4 --lr_critic 5e-4`
- Increase buffer size: `--buffer_size 4096`
- Adjust PPO clipping: `--clip_epsilon 0.1`

**Agent learns slowly:**
- Increase exploration: `--entropy_coef 0.05`
- Check reward scaling in environment
- Increase buffer size for more diverse experience

### Dashboard Issues

**Display problems:**
```bash
# For matplotlib dashboard
export MPLBACKEND=TkAgg  # or Qt5Agg

# For web dashboard, ensure streamlit is installed
pip install streamlit
```

**Slow performance:**
- Reduce history length in dashboard settings
- Close other applications
- Use matplotlib dashboard instead of web for faster updates

---

## 📈 Expected Training Time

**CPU:**
- 100 episodes: ~15 minutes
- 500 episodes: ~60 minutes
- 1000 episodes: ~2 hours

**GPU (CUDA):**
- 100 episodes: ~5 minutes
- 500 episodes: ~20 minutes
- 1000 episodes: ~3 hours

---

## 🎯 Future Improvements

Based on midterm report analysis:

### Immediate Priorities
1. **Reward shaping:** Add anticipatory term, piecewise costs, DR utilization bonus (30-40% improvement expected)
2. **Enhanced observations:** Add aggregate features, rate-of-change, 3-step history (50% miscoordination reduction expected)
3. **PI-AGC baseline:** Implement classical controller for quantitative comparison

### Algorithmic Enhancements
1. **Curriculum learning:** Progressive difficulty increase
2. **Safe exploration:** Action masking and safety layers
3. **Value improvements:** Target networks, reward normalization

### Evaluation Extensions
1. Compare vs. PI-AGC, MPC oracle
2. Test on extreme scenarios (renewable ramps, double contingencies)
3. Comprehensive ablation studies

---

## 📚 References

See `doc/midterm/main.pdf` for complete bibliography including:
- Kundur (1994) - Power System Stability and Control
- Lowe et al. (2017) - Multi-Agent Actor-Critic (MADDPG)
- Cao et al. (2020) - RL for Modern Power Systems
- And 7 more references

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@misc{smith2025mappo,
  title={Multi-Agent Reinforcement Learning for Real-Time Frequency Regulation in Power Grids},
  author={Smith, Derek and Vu, Matthew},
  year={2025},
  institution={Harvard University},
  note={ES 158: Sequential Decision Making in Dynamic Environments}
}
```

---

## 📧 Contact

For questions about this implementation:
- See complete technical details in `doc/midterm/main.pdf`
- Check code documentation in `main/` directory
- Review training results in `checkpoints/training_results.png`

---

**License:** MIT (or specify your license)

**Status:** Midterm report completed (November 2025). Final improvements ongoing.

