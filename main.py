import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from go_bot.actor_critic import ActorCriticNet, ActorCriticPolicy
from go_bot.mcts import mcts_search
from tqdm import tqdm
import gym
from go_bot.eval import EloEvaluator
from go_bot.self_play import play_games

def main():
    # Configuration
    BOARD_SIZE = 9  # Adjusted for training stability
    EPISODES = 5  # Increased number of episodes
    MCTS_SIMULATIONS = 5  # Adjusted for better search
    TEMPERATURE = 1.0
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 5
    CHECKPOINT_DIR = "checkpoints"
    OUTPUT_PLOTS_DIR = "plots"

    # Determine the device (CPU or CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Ensure directories exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

    # Initialize environment, model, and evaluator
    env = gym.make("gym_go:go-v0", size=BOARD_SIZE)
    GoGame = env.gogame

    # Move the model to the device
    model = ActorCriticNet(BOARD_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    evaluator = EloEvaluator()
    evaluator.add_model("random_policy")

    # Initialize policy wrappers
    black_policy = ActorCriticPolicy(model, MCTS_SIMULATIONS, TEMPERATURE)
    white_policy = ActorCriticPolicy(model, MCTS_SIMULATIONS, TEMPERATURE)

    # Training loop
    print("Starting training...")
    for epoch in range(1, 11):
        print(f"\n=== Epoch {epoch} ===")
        print("Generating self-play data...")

        # Self-play to generate training data
        win_rate, _, replay, _ = play_games(env, black_policy, white_policy, EPISODES)
        training_data = [event for traj in replay for event in traj.get_events()]

        # Prepare training data
        states = [event[0] for event in training_data]  # List of numpy arrays of shape [6, board_size, board_size]
        valid_moves_list = [event[1] for event in training_data]  # List of numpy arrays of shape [action_size]
        policies = [event[6] for event in training_data]  # List of numpy arrays of shape [action_size]
        values = [event[3] for event in training_data]  # List of scalar values (rewards)

        # Convert lists to numpy arrays
        states_array = np.array(states)
        valid_moves_array = np.array(valid_moves_list)
        policies_array = np.array(policies)
        values_array = np.array(values)

        # Convert to tensors
        states_tensor = torch.tensor(states_array, dtype=torch.float32).to(device)
        valid_moves_tensor = torch.tensor(valid_moves_array, dtype=torch.float32).to(device)
        policies_tensor = torch.tensor(policies_array, dtype=torch.float32).to(device)
        values_tensor = torch.tensor(values_array, dtype=torch.float32).to(device)

        # Train the model
        print("Training the model...")
        dataset = torch.utils.data.TensorDataset(states_tensor, policies_tensor, values_tensor, valid_moves_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch_idx in range(EPOCHS):
            total_loss = 0
            for batch_states, batch_policies, batch_values, batch_valid_moves in tqdm(dataloader, desc=f"Epoch {epoch_idx + 1}/{EPOCHS}"):
                optimizer.zero_grad()
                loss, _, _ = model.compute_loss(batch_states, None, batch_values, batch_policies, batch_valid_moves)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch_idx + 1}: Total Loss = {total_loss:.4f}")

        # Save model checkpoint
        checkpoint_name = f"model_epoch_{epoch}.pth"
        checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved: {checkpoint_path}")

        evaluator.add_model(checkpoint_name)

        # Evaluate against the previous checkpoint
        if epoch > 1:
            prev_checkpoint_name = f"model_epoch_{epoch - 1}.pth"
            prev_model = ActorCriticNet(BOARD_SIZE).to(device)
            prev_checkpoint_path = os.path.join(CHECKPOINT_DIR, prev_checkpoint_name)
            prev_model.load_state_dict(torch.load(prev_checkpoint_path, map_location=device))

            prev_policy = ActorCriticPolicy(prev_model, MCTS_SIMULATIONS, TEMPERATURE)
            win_rate, _, _, _ = play_games(env, black_policy, prev_policy, EPISODES)

            evaluator.update_ratings(checkpoint_name, prev_checkpoint_name, win_rate)
            print(f"Win rate against {prev_checkpoint_name}: {win_rate:.2f}")

    # Save Elo progression plot
    plot_elo_progression(evaluator, OUTPUT_PLOTS_DIR)
    print(f"Elo progression plot saved in {OUTPUT_PLOTS_DIR}")

def plot_elo_progression(evaluator, output_dir):
    model_names = list(evaluator.ratings.keys())
    ratings = [evaluator.get_rating(name) for name in model_names]

    plt.figure(figsize=(10, 6))
    plt.plot(model_names, ratings, marker="o", label="Elo Rating")
    plt.xlabel("Model Checkpoint")
    plt.ylabel("Elo Rating")
    plt.title("Elo Rating Progression")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "elo_progression.png")
    plt.savefig(plot_path)
    print(f"Elo progression plot saved at: {plot_path}")

if __name__ == "__main__":
    main()
