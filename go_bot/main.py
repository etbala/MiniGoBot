import gym
import os
from training.train import GoTrainer

if __name__ == "__main__":
    import gym

    BOARD_SIZE = 19
    MCTS_SIMULATIONS = 50
    TEMPERATURE = 1.0
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 5
    NUM_SELF_PLAY_GAMES = 10
    MODEL_PATH = "actor_critic_model.pth"

    env = gym.make("gym_go:go-v0", size=BOARD_SIZE)
    trainer = GoTrainer(BOARD_SIZE, MCTS_SIMULATIONS, TEMPERATURE, LEARNING_RATE, BATCH_SIZE, EPOCHS, MODEL_PATH)

    print("Generating self-play data...")
    training_data = trainer.self_play(NUM_SELF_PLAY_GAMES, env)

    print("Training the model...")
    trainer.train(training_data)

    print("Evaluating the model...")
    trainer.evaluate(env)

    trainer.save_model()