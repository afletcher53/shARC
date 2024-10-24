from collections import Counter
import os

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from classes.data_loader import DataLoader
from utils.find_similar_grid import (
    find_similar_solutions,
)
from utils.generate_aug_training import get_augmented_training_examples, subtract_colour_maps


def apply_delta_to_grid(grid: np.ndarray, training_example_delta: tuple) -> np.ndarray:
    """
    Apply geometric and color map transformations to a grid based on the provided delta.
    
    Args:
        grid (np.ndarray): The input grid to transform
        training_example_delta (tuple): A tuple containing (colour_map_delta, geometric_delta)
            where geometric_delta is a string describing the transformations
            and colour_map_delta is a dictionary mapping original colors to new colors
    
    Returns:
        np.ndarray: The transformed grid
    """
    colour_map_delta, geometric_delta = training_example_delta
    
    # First apply geometric transformations
    transformed_grid = apply_geometric_delta(grid, geometric_delta)
    
    # Then apply color map transformations
    if colour_map_delta is not None:
        transformed_grid = apply_colour_map_delta(transformed_grid, colour_map_delta)
    
    return transformed_grid

def apply_geometric_delta(grid: np.ndarray, geometric_delta: str) -> np.ndarray:
    """
    Apply geometric transformations to a grid based on the transformation description.
    
    Args:
        grid (np.ndarray): The input grid to transform
        geometric_delta (str): String describing the transformations
            Format: "[flip_description], rotated [X]°"
    
    Returns:
        np.ndarray: The geometrically transformed grid
    """
    transformed_grid = grid.copy()
    
    # Parse the geometric delta string
    flip_part, rotation_part = geometric_delta.split(", rotated ")
    rotation_degrees = int(rotation_part.rstrip("°"))
    
    # Apply flips first
    if "vertical flip" in flip_part:
        transformed_grid = np.flipud(transformed_grid)
    if "horizontal flip" in flip_part:
        transformed_grid = np.fliplr(transformed_grid)
    
    # Apply rotation
    rotation_k = (rotation_degrees // 90) % 4
    transformed_grid = np.rot90(transformed_grid, k=rotation_k)
    
    return transformed_grid

def apply_colour_map_delta(grid: np.ndarray, colour_map_delta: dict) -> np.ndarray:
    """
    Apply color transformations to a grid based on the color map delta.
    
    Args:
        grid (np.ndarray): The input grid to transform
        colour_map_delta (dict): Dictionary mapping original colors to new colors
    
    Returns:
        np.ndarray: The color-transformed grid
    """
    transformed_grid = grid.copy()
    
    # Create a vectorized function to apply the color mapping
    def map_color(x):
        return colour_map_delta.get(x, x)
    
    transformed_grid = np.vectorize(map_color)(transformed_grid)
    return transformed_grid
def plot_transformation_set(dl, challenge_id, training_examples, test_input, solution, transformed_test, transformed_solution, index):
    """
    Plots all training examples (inputs and outputs) along with test input and solution.
    
    :param dl: DataLoader instance
    :param challenge_id: ID of the challenge
    :param training_examples: List of 4 (input_dict, output_dict) pairs
    :param test_input: Original test input
    :param solution: Original solution
    :param transformed_test: Transformed test input
    :param transformed_solution: Transformed solution
    :param index: Index of the transformation
    """
    fig, axs = plt.subplots(4, 2, figsize=(10, 20))
    plt.suptitle(f"Transformation Set {index} for Challenge {challenge_id}", fontsize=16)
    
    # Plot the 4 training example pairs (input/output)
    for i, (input_dict, output_dict) in enumerate(training_examples):
        # Training input
        axs[i, 0].imshow(
            np.array(input_dict['original']).squeeze(),
            cmap=matplotlib.colors.ListedColormap(dl.strings.CUSTOM_COLORS),
            interpolation='nearest'
        )
        axs[i, 0].grid(True, which='both', color='grey', linewidth=0.5, alpha=0.5)
        axs[i, 0].set_title(f"Training Input {i+1}")
        
        # Training output
        axs[i, 1].imshow(
            np.array(output_dict['original']).squeeze(),
            cmap=matplotlib.colors.ListedColormap(dl.strings.CUSTOM_COLORS),
            interpolation='nearest'
        )
        axs[i, 1].grid(True, which='both', color='grey', linewidth=0.5, alpha=0.5)
        axs[i, 1].set_title(f"Training Output {i+1}")
    
    # Add description from first example
    plt.figtext(0.02, 0.98, f"Transformation: {training_examples[0][0]['description']}", 
                wrap=True, horizontalalignment='left', fontsize=10)
    
    # Adjust layout and add gridlines
    for row in axs:
        for ax in row:
            ax.set_xticks(np.arange(-0.5, 30, 1))
            ax.set_yticks(np.arange(-0.5, 30, 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    
    plt.tight_layout()
    
    # Create a second figure for test input and solution
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
    plt.suptitle("Test Input and Solution", fontsize=16)
    
    # Test input
    axs2[0].imshow(
        np.array(test_input).squeeze(),
        cmap=matplotlib.colors.ListedColormap(dl.strings.CUSTOM_COLORS),
        interpolation='nearest'
    )
    axs2[0].grid(True, which='both', color='grey', linewidth=0.5, alpha=0.5)
    axs2[0].set_title("Test Input")
    
    # Solution
    axs2[1].imshow(
        np.array(solution).squeeze(),
        cmap=matplotlib.colors.ListedColormap(dl.strings.CUSTOM_COLORS),
        interpolation='nearest'
    )
    axs2[1].grid(True, which='both', color='grey', linewidth=0.5, alpha=0.5)
    axs2[1].set_title("Solution")
    
    # Adjust layout for second figure
    for ax in axs2:
        ax.set_xticks(np.arange(-0.5, 30, 1))
        ax.set_yticks(np.arange(-0.5, 30, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    plt.tight_layout()
    
    # Save the figures
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save training examples
    output_path1 = os.path.join(output_dir, f"{challenge_id}_training_examples_{index}.png")
    fig.savefig(output_path1, bbox_inches='tight', dpi=300)
    
    # Save test and solution
    output_path2 = os.path.join(output_dir, f"{challenge_id}_test_and_solution_{index}.png")
    fig2.savefig(output_path2, bbox_inches='tight', dpi=300)
    
    plt.close(fig)
    plt.close(fig2)

def main():
    dl = DataLoader()
    training_data = dl.load_dataset("training")

    total_training_inputs = 0
    for key, value in training_data.items():
        total_training_inputs += len(value["train_examples"])
    print(f"Total training inputs: {total_training_inputs}")

    unique_keys = set(training_data.keys())
    print(f"Number of training challenges: {len(unique_keys)}")

    challenge_id = "00d62c1b"
    challenge_data = dl.get_specific_sample(challenge_id)

    print("Test Input:", challenge_data["test_input"])
    print("Training Examples:", challenge_data["train_examples"])
    print("Solution:", challenge_data["solution"])

    # dl.plot_train_and_test_examples({challenge_id: challenge_data})

    # dl.plot_solution(challenge_data["test_input"], f"{challenge_id}_test_input")

    training_examples_cid = get_augmented_training_examples(dl, challenge_id, visualize=False)
    # TODO: what exactly does training_examples_cid look like?
    #  from the code, it looks like this below - is this correct? what does pair[0] and pair[1] correspond to? input/output?
    # training_examples_cid = [
    #   [
    #       {
    #           "colour_map": {int1: int2, int2: int3, ...},
    #           "description": "str",
    #       },
    #       {
    #           "colour_map": {int1: int2, int2: int3, ...},
    #           "description": "str"},
    #   ],
    # ]


    # for each training_example_cid, find the consistent colour map delta

    training_example_deltas = []
    for training_example in training_examples_cid:
        colour_map_deltas = []
        geometric_deltas = []
        for pair in training_example:
            colour_map_deltas.append(subtract_colour_maps(pair[0]["colour_map"], pair[1]["colour_map"]))
            geometric_deltas.append(pair[0]["description"])
        # assert that all the geometric deltas are the same
        # JC: I see, so these asserts are ensuring that the colour maps are aligned across different i/o pairs?
        assert all(delta == geometric_deltas[0] for delta in geometric_deltas)
        # assert that all the colour map deltas are the same
        assert all(delta == colour_map_deltas[0] for delta in colour_map_deltas)
        training_example_deltas.append((colour_map_deltas[0], geometric_deltas[0]))
    # get the most common colour map delta
    test_input = challenge_data["test_input"]
    solution = challenge_data["solution"]

   # Apply deltas to test input and solution
    test_input_deltas = []
    solution_deltas = []
    for idx, training_example_delta in enumerate(training_example_deltas):
        # Apply geometric transformation to test input
        transformed_test = apply_delta_to_grid(
            test_input, 
            (None, training_example_delta[1])  # Only geometric delta for test input
        )
        test_input_deltas.append(transformed_test)
        
        # Apply both transformations to solution
        transformed_solution = apply_delta_to_grid(
            solution,
            training_example_delta  # Both color and geometric deltas
        )
        solution_deltas.append(transformed_solution)
        
        # Plot comprehensive visualization
        plot_transformation_set(
            dl,
            challenge_id,
            training_examples_cid[idx],  # Pass all 4 training examples
            test_input,
            solution,
            transformed_test,
            transformed_solution,
            idx
        )
            



    similar_grids, original_idx = find_similar_solutions(solution, dl, 5)

    for idx, grid in enumerate(similar_grids):
        dl.plot_solution(grid, f"similar_grid_{idx}")

    dl.plot_solution(solution, "original_solution")


if __name__ == "__main__":
    main()
