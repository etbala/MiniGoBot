import os
import torch
import matplotlib.pyplot as plt
from go_bot.actor_critic import ActorCriticNet, ActorCriticPolicy
from go_bot.mcts import mcts_search
from tqdm import tqdm
from go_bot.eval import EloEvaluator
from go_bot.self_play import play_games

def main():
    # Configuration
    BOARD_SIZE = 19
    EPISODES = 20
    MCTS_SIMULATIONS = 50
    TEMPERATURE = 1.0
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 5
    CHECKPOINT_DIR = "checkpoints"
    OUTPUT_PLOTS_DIR = "plots"

    # Ensure directories exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

    # Initialize environment, model, and evaluator
    import gym
    env = gym.make("gym_go:go-v0", size=BOARD_SIZE)

    model = ActorCriticNet(BOARD_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    evaluator = EloEvaluator()

    # Add initial benchmark policy (random policy)
    evaluator.add_model("random_policy")

    # Initialize policy wrappers
    black_policy = ActorCriticPolicy(model, MCTS_SIMULATIONS, TEMPERATURE)
    white_policy = ActorCriticPolicy(model, MCTS_SIMULATIONS, TEMPERATURE)

    # Training loop
    print("Starting training...")
    for epoch in range(1, 11):  # Example: 10 epochs
        print(f"\n=== Epoch {epoch} ===")
        print("Generating self-play data...")

        # Self-play to generate training data
        win_rate, _, replay, _ = play_games(env, black_policy, white_policy, EPISODES)
        training_data = [event for traj in replay for event in traj.get_events()]

        # Prepare training data
        states = torch.tensor([event[0] for event in training_data], dtype=torch.float32)
        policies = torch.tensor([event[6] for event in training_data], dtype=torch.float32)
        values = torch.tensor([event[5] for event in training_data], dtype=torch.float32)

        # Train the model
        print("Training the model...")
        dataset = torch.utils.data.TensorDataset(states, policies, values)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch_idx in range(EPOCHS):
            total_loss = 0
            for batch_states, batch_policies, batch_values in tqdm(dataloader, desc=f"Epoch {epoch_idx + 1}/{EPOCHS}"):
                optimizer.zero_grad()
                loss, _, _ = model.compute_loss(batch_states, None, batch_values, batch_policies)
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
            prev_model = ActorCriticNet(BOARD_SIZE)
            prev_checkpoint_path = os.path.join(CHECKPOINT_DIR, prev_checkpoint_name)
            prev_model.load_state_dict(torch.load(prev_checkpoint_path))

            prev_policy = ActorCriticPolicy(prev_model, MCTS_SIMULATIONS, TEMPERATURE)
            win_rate, _, _, _ = play_games(env, black_policy, prev_policy, EPISODES)

            evaluator.update_ratings(checkpoint_name, prev_checkpoint_name, win_rate)
            print(f"Win rate against {prev_checkpoint_name}: {win_rate:.2f}")

    # Save Elo progression plot
    plot_elo_progression(evaluator, OUTPUT_PLOTS_DIR)
    print(f"Elo progression plot saved in {OUTPUT_PLOTS_DIR}")

def plot_elo_progression(evaluator, output_dir):
    """
    Save a plot showing the Elo progression over time.
    """
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
