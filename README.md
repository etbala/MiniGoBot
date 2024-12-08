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


> **Note:** To visualize the games in the replay buffer, run `visualize.py` while there is a replay.pickle file in bin.
