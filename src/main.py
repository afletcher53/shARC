import random
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.base import defaultdict

from classes import ProjectStrings
from classes.data_loader import DataLoader
from utils.find_similar_grid import (
    find_similar_solutions,
    generate_augmentations,
)


def plot_grid_pair(input_grid, output_grid, pair_index, custom_colors):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Grid Pair {pair_index + 1}")

    cmap = ListedColormap(custom_colors)

    norm = plt.Normalize(0, len(custom_colors) - 1)

    ax1.imshow(input_grid, cmap=cmap, norm=norm)
    ax1.set_title("Input Grid")
    ax1.axis("off")

    ax2.imshow(output_grid, cmap=cmap, norm=norm)
    ax2.set_title("Output Grid")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(f"output/grid_pair_{pair_index}.png")
    plt.close(fig)


def visualize_grid_pairs(valid_pairs):
    project_strings = ProjectStrings()
    custom_colors = project_strings.CUSTOM_COLORS
    for i, pair in enumerate(valid_pairs[:40]):
        input_grid = pair[0]["permutation"]
        output_grid = pair[1]["permutation"]
        plot_grid_pair(input_grid, output_grid, i, custom_colors)


def get_consistent_color_mapping(pairs):
    # Get all unique colors from inputs
    input_colors = set()
    for pair in pairs:
        input_colors.update(pair[0]["colour_map"].values())

    # Get all unique colors from outputs
    output_colors = set()
    for pair in pairs:
        output_colors.update(pair[1]["colour_map"].values())

    # Find new colors in outputs
    new_colors = output_colors - input_colors

    # Create a consistent mapping for new colors
    consistent_mapping = {color: i for i, color in enumerate(new_colors)}

    return consistent_mapping, input_colors

def apply_consistent_mapping(pair, consistent_mapping, input_colors):
    input_map = pair[0]["colour_map"]
    output_map = pair[1]["colour_map"]

    new_output_map = {}
    for key, color in output_map.items():
        if color in input_colors:
            new_output_map[key] = color
        else:
            new_output_map[key] = consistent_mapping[color]

    return (pair[0], {"colour_map": new_output_map})

def get_augmented_training_examples(dataloader: DataLoader, id: str):
    sample = dataloader.get_specific_sample(id)
    train_examples = sample["train_examples"]

    valid_pairs = []
    train_example_count = 0
    for train_example in train_examples:
        train_example_input = train_example["input"]
        train_example_output = train_example["output"]

        if not isinstance(train_example_input[0], list):
            train_example_input = [train_example_input]
        if not isinstance(train_example_output[0], list):
            train_example_output = [train_example_output]

        aug_input_dict = generate_augmentations(
            [train_example_input], id, "input", train_example_count
        )
        aug_output_dict = generate_augmentations(
            [train_example_output], id, "output", train_example_count
        )

        final_pairs = []
        for input_augmentation in aug_input_dict:
            for output_augmentation in aug_output_dict:
                if (
                    aug_input_dict[input_augmentation]["description_hash"]
                    == aug_output_dict[output_augmentation]["description_hash"]
                ):
                    final_pairs.append(
                        (
                            aug_input_dict[input_augmentation],
                            aug_output_dict[output_augmentation],
                        )
                    )

        for pair in final_pairs:
            set_input = pair[0]["colour_map"]
            set_output = pair[1]["colour_map"]

            lowest_cardinality_set = (
                set_input if len(set_input) < len(set_output) else set_output
            )
            highest_cardinality_set = (
                set_output if len(set_output) > len(set_input) else set_input
            )

            if all(
                item in highest_cardinality_set.values()
                for item in lowest_cardinality_set.values()
            ):
                if all(
                    highest_cardinality_set[key] == lowest_cardinality_set[key]
                    for key in lowest_cardinality_set.keys()
                ):
                    valid_pairs.append(pair)

        # visualize_grid_pairs(valid_pairs)
        train_example_count += 1

    def subtract_colour_maps(colour_map_1, colour_map_2):
        return {key: value for key, value in colour_map_2.items() if key not in colour_map_1 or colour_map_1[key] != value}

    grouped_by_description_hash = {}
    for pair in valid_pairs:
        description_hash = pair[0]["description_hash"]
        if description_hash not in grouped_by_description_hash:
            grouped_by_description_hash[description_hash] = []
        grouped_by_description_hash[description_hash].append(pair)

    delta_maps = []
    for description_hash, pairs in grouped_by_description_hash.items():
        for pair in pairs:
            delta_map = subtract_colour_maps(pair[0]["colour_map"], pair[1]["colour_map"])
            delta_maps.append((description_hash, delta_map, pair))

    # Group by both description_hash and delta_map
    grouped_by_description_and_delta = defaultdict(list)
    for description_hash, delta_map, original_pair in delta_maps:
        # Convert delta_map to a frozenset of items for hashability
        delta_map_key = frozenset(delta_map.items())
        grouped_by_description_and_delta[(description_hash, delta_map_key)].append(original_pair)

    # Create training examples
    training_examples = []
    for group in grouped_by_description_and_delta.values():
        if len(group) >= 4:
            # Randomly select 4 pairs from the group
            selected_pairs = random.sample(group, 4)
            training_examples.append(selected_pairs)
    # shuffle training examples within each group
    for group in training_examples:
        random.shuffle(group)
    project_strings = ProjectStrings()
    custom_colors = project_strings.CUSTOM_COLORS   

    visualize_all_training_examples(training_examples, custom_colors)


    return valid_pairs

def visualize_training_example(training_example, custom_colors, example_index):
    fig, axes = plt.subplots(4, 2, figsize=(12, 20))
    fig.suptitle(f"Training Example {example_index + 1}")

    cmap = ListedColormap(custom_colors)
    norm = plt.Normalize(0, len(custom_colors) - 1)

    for i, pair in enumerate(training_example):
        input_grid = pair[0]["permutation"]
        output_grid = pair[1]["permutation"]

        axes[i, 0].imshow(input_grid, cmap=cmap, norm=norm)
        axes[i, 0].set_title(f"Input Grid {i + 1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(output_grid, cmap=cmap, norm=norm)
        axes[i, 1].set_title(f"Output Grid {i + 1}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(f"output/training_example_{example_index}.png")
    plt.close(fig)

def visualize_all_training_examples(training_groups, custom_colors):
    for i, training_example in enumerate(training_groups):
        visualize_training_example(training_example, custom_colors, i)

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

    dl.plot_train_and_test_examples({challenge_id: challenge_data})

    dl.plot_solution(challenge_data["test_input"], f"{challenge_id}_test_input")

    get_augmented_training_examples(dl, challenge_id)

    solution = challenge_data["solution"]

    similar_grids, original_idx = find_similar_solutions(solution, dl, 5)

    for idx, grid in enumerate(similar_grids):
        dl.plot_solution(grid, f"similar_grid_{idx}")

    dl.plot_solution(solution, "original_solution")


if __name__ == "__main__":
    main()
