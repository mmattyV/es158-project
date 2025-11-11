# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Running Training

### Basic Training (500 episodes, default settings)
```bash
python main.py
```

### Quick Test Run (100 episodes)
```bash
python main.py --n_episodes 100
```

### Long Training Run (2000 episodes)
```bash
python main.py --n_episodes 2000 --eval_interval 100
```

### Custom Configuration
```bash
python main.py \
  --n_episodes 1000 \
  --max_steps 500 \
  --buffer_size 2048 \
  --batch_size 256 \
  --n_epochs 10 \
  --lr_actor 3e-4 \
  --lr_critic 1e-3
```

### With Environment Testing First
```bash
python main.py --test_env --n_episodes 500
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_episodes` | 500 | Number of training episodes |
| `--max_steps` | 500 | Maximum steps per episode |
| `--buffer_size` | 2048 | Rollout buffer size |
| `--batch_size` | 256 | Minibatch size |
| `--n_epochs` | 10 | PPO update epochs |
| `--eval_interval` | 50 | Episodes between evaluations |
| `--lr_actor` | 3e-4 | Actor learning rate |
| `--lr_critic` | 1e-3 | Critic learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--gae_lambda` | 0.95 | GAE lambda |
| `--save_dir` | checkpoints | Directory to save models |
| `--test_env` | False | Test environment before training |
| `--device` | auto | Device: cpu, cuda, or auto |

## Monitoring Training

### TensorBoard Dashboard (Real-time)
While training is running, open a new terminal and run:

```bash
tensorboard --logdir=runs
```

Then open your browser to: **http://localhost:6006**

You'll see real-time plots of:
- Episode rewards
- Actor and critic losses
- Policy entropy
- Episode lengths
- Evaluation performance

### Training Output
The script will print progress every 10 episodes:
```
Episode 10/500 | Avg Reward: -12543.21 | Length: 245
Episode 20/500 | Avg Reward: -10234.56 | Length: 312
...
```

## Outputs

After training completes, you'll find:

1. **Models**:
   - `checkpoints/best_model.pt` - Best performing model
   - `checkpoints/final_model.pt` - Final model after all episodes

2. **TensorBoard Logs**:
   - `runs/mappo_YYYYMMDD_HHMMSS/` - TensorBoard event files

3. **Training Graphs**:
   - `checkpoints/training_results.png` - Comprehensive 9-panel visualization showing:
     - Episode rewards with moving average
     - Evaluation performance
     - Actor loss
     - Critic loss  
     - Policy entropy
     - Episode lengths
     - Cumulative reward
     - Training summary statistics

## Evaluating Trained Models

After training, evaluate your model:

```bash
# Basic evaluation
python evaluate.py --model_path checkpoints/best_model.pt --n_episodes 10

# With visualization
python evaluate.py --model_path checkpoints/best_model.pt --visualize

# Compare with random baseline
python evaluate.py --model_path checkpoints/best_model.pt --compare
```

## Example Workflow

```bash
# 1. Quick test (5 minutes)
python main.py --n_episodes 50

# 2. View results
open checkpoints/training_results.png

# 3. If results look good, run longer training
python main.py --n_episodes 1000

# 4. Monitor in real-time with TensorBoard (in another terminal)
tensorboard --logdir=runs

# 5. Evaluate best model
python evaluate.py --model_path checkpoints/best_model.pt --visualize --compare
```

## Expected Training Time

On CPU:
- 100 episodes: ~10-15 minutes
- 500 episodes: ~45-60 minutes
- 1000 episodes: ~1.5-2 hours

On GPU (CUDA):
- 100 episodes: ~3-5 minutes
- 500 episodes: ~15-20 minutes
- 1000 episodes: ~30-40 minutes

## Troubleshooting

**Import errors:**
```bash
pip install -r requirements.txt
```

**CUDA out of memory:**
```bash
python main.py --device cpu
# or reduce batch size:
python main.py --batch_size 128
```

**Training too slow:**
```bash
# Reduce episodes or steps
python main.py --n_episodes 200 --max_steps 300
```

**Want to resume training:**
Currently not supported - training starts fresh each time. To continue training, you'd need to modify the code to load a checkpoint.

