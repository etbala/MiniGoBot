# MiniGoBot

MiniGoBot is a reinforcement learning project that combines an actor-critic neural network with Monte Carlo Tree Search (MCTS) to play the game of Go.

## Getting Started

### Prerequisites

- Python 3.9.x - 3.12.x
- (Optional) CUDA Compatible GPU

### Run Locally

1. Clone and install the GymGo environment:
    ```
    git clone https://github.com/huangeddie/GymGo.git
    cd GymGo
    pip install -e .
    ```
2. Clone the MiniGoBot repo and install dependencies:
   ```
   git clone https://github.com/etbala/MiniGoBot.git
   cd MiniGoBot
   pip install -r requirements.txt
   ```
   > **Note:** For GPU support, follow [PyTorch installation instructions](https://pytorch.org/get-started/locally/) to install the appropriate CUDA version.
3. Run the main training script:
   ```
   python main.py
   ```

## Command Line Arguments

#### Go Environment
- `--size`: Board size (default: 9)
- `--reward`: Reward system (`real` (default) or `heuristic`)

#### Training Params
- `--mcts`: Number of Monte Carlo searches (default: 0)
- `--lr`: Learning Rate (default: 1e-3)
- `--temp`: Initial temperature for exploration (default: 1)

#### Data Sizes
- `--batchsize`: Training batch size (default: 32)
- `--replaysize`: Maximum number of stored games in replay buffer (default: 64)
- `--batches`: Number of training batches per iteration (default: 1000)

#### Loading
- `--customdir`: Path to load a model from a custom directory (default: '')
- `--latest-checkpoint`: Load the latest checkpoint (default: false)

#### Training
- `--iterations`: Number of training iterations (default: 128)
- `--episodes`: Number of episodes per iteration (default: 32)
- `--evaluations`: Number of evaluation episodes (default: 16)
- `--eval-interval`: Interval for evaluation in terms of iterations (default: 4)

#### Disk Data
- `--replay-path`: Path to save replay data (default: `bin/replay.pickle'`)
- `--checkdir`: Directory to save checkpoints (default: `bin/checkpoints/{today}/`)

#### Model
- `--model`: Model type (`ac` (default), `rand`, `human`)

#### Hardware
- `--device`: Device for PyTorch models (`cuda` (default), or `cpu`)

#### Visual
- `--render`: Rendering type (`terminal` (default) or `human`)
