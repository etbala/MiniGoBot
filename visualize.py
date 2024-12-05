import pickle
import gym
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

from go_bot.data import load_replay, GoVars

def action_to_move(action, board_size):
    """
    Converts an action index to a board move (row, col).
    """
    if action == board_size ** 2:
        # Pass action
        return None  # Some environments may expect a specific pass representation
    else:
        row = action // board_size
        col = action % board_size
        return (row, col)

def render_board(state, board_size, step_index, output_dir):
    """
    Renders the board state and saves it as an image.
    """
    # Extract the board positions
    black_positions = state[GoVars.BLACK]
    white_positions = state[GoVars.WHITE]

    # Create a matplotlib figure
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Set the limits to always show the entire board
    ax.set_xlim(-0.5, board_size - 0.5)
    ax.set_ylim(-0.5, board_size - 0.5)
    ax.invert_yaxis()  # Invert y-axis to match the Go board orientation

    # Set grid lines
    ax.set_xticks(np.arange(board_size + 1))
    ax.set_yticks(np.arange(board_size + 1))
    ax.grid(True)

    # Ensure grid lines are below the stones
    ax.set_axisbelow(True)

    # Plot black stones
    black_coords = np.argwhere(black_positions == 1)
    if black_coords.size > 0:
        ax.scatter(black_coords[:, 1], black_coords[:, 0], c='black', s=200)

    # Plot white stones
    white_coords = np.argwhere(white_positions == 1)
    if white_coords.size > 0:
        ax.scatter(white_coords[:, 1], white_coords[:, 0], c='white', edgecolors='black', s=200)

    # Remove margins
    plt.tight_layout()

    # Save the figure
    filename = f"step_{step_index:03d}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return filepath

def main():
    # Load the replay data
    replay_path = 'bin/replay.pickle'  # Update this path if necessary
    replay = load_replay(replay_path)

    # Iterate over each game in the replay
    for game_index, traj in enumerate(replay):
        print(f"Game {game_index + 1}:")
        # Create a directory to store images for this game
        output_dir = f"bin/game_{game_index + 1}_frames"
        os.makedirs(output_dir, exist_ok=True)

        # Initialize the Go environment for each game
        board_size = traj.states[0].shape[1]
        go_env = gym.make('gym_go:go-v0', size=board_size, komi=0, reward_method='heuristic', disable_env_checker=True)
        go_env.reset()
        
        # Retrieve the actions from the trajectory
        actions = traj.actions
        pis = traj.pis
        rewards = traj.rewards
        won = traj.get_winner()
        
        # List to store image file paths
        image_paths = []

        # Reconstruct the game by applying actions
        for step_index, (action, pi) in enumerate(zip(actions, pis)):
            # Determine whose turn it is
            current_player = 'Black' if go_env.turn() == GoVars.BLACK else 'White'
            
            # Convert action index to board move
            move = action_to_move(action, board_size)
    
            # Apply the move to the environment
            state, reward, done, info = go_env.step(move)
    
            # Render the board state and save the image
            image_path = render_board(go_env.state(), board_size, step_index, output_dir)
            image_paths.append(image_path)
    
            # Optionally, print out additional information
            print(f"Step {step_index + 1}: Player {current_player} moved {move if move is not None else 'pass'}")
            if done:
                break  # Exit the loop if the game has ended

        # After the game, create a GIF from the images
        gif_filename = f"game_{game_index + 1}.gif"
        gif_filepath = os.path.join(output_dir, gif_filename)
        with imageio.get_writer(gif_filepath, mode='I', duration=250, loop=0) as writer:
            for filename in image_paths:
                image = imageio.imread(filename)
                writer.append_data(image)

        print(f"Game {game_index + 1} ended. Winner: {'Black' if won == 1 else 'White' if won == -1 else 'Draw'}")
        print(f"GIF saved as {gif_filepath}")
        print("=" * 50)

if __name__ == "__main__":
    main()
