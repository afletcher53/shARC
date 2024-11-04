import random
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.base import defaultdict
from classes.data_loader import DataLoader
from classes.project_strings import ProjectStrings
from utils.find_similar_grid import generate_augmentations


def prepare_example(train_example):
    """Prepare a single training example by ensuring inputs and outputs are in list format."""
    input_data = train_example["input"]
    output_data = train_example["output"]
    prepared_input = [input_data] if not isinstance(input_data[0], list) else input_data
    prepared_output = [output_data] if not isinstance(output_data[0], list) else output_data
    return prepared_input, prepared_output

def generate_augmentation_pairs(input_data, output_data, id, train_example_count):
    """Generate and pair augmentations for input and output data."""
    aug_input_dict = generate_augmentations([input_data], id, "input", train_example_count)
    aug_output_dict = generate_augmentations([output_data], id, "output", train_example_count)
    
    final_pairs = []
    for input_aug in aug_input_dict:
        for output_aug in aug_output_dict:
            if aug_input_dict[input_aug]["description_hash"] == aug_output_dict[output_aug]["description_hash"]:
                final_pairs.append((aug_input_dict[input_aug], aug_output_dict[output_aug]))
    return final_pairs

def validate_pair(pair):
    """Validate a single augmentation pair.
    TODO: it looks like this is checking that, within an input/output pair, the input and output colour maps align
     not sure where is the function that checks whether the colour maps across i/o pairs align?
    """
    set_input = pair[0]["colour_map"]
    set_output = pair[1]["colour_map"]
    lowest_card_set = set_input if len(set_input) < len(set_output) else set_output
    highest_card_set = set_output if len(set_output) > len(set_input) else set_input
    
    if not all(item in highest_card_set.values() for item in lowest_card_set.values()):
        return False
    if not all(highest_card_set[key] == lowest_card_set[key] for key in lowest_card_set.keys()):
        return False
    return True

def subtract_colour_maps(colour_map_1, colour_map_2):
    """Calculate the difference between two color maps.
    JC note: this looks like colour_map_2 - colour_map_1
    TODO: why also remove the elements from colour_map_2 that colour_map_1 maps to a different value?
     from where it's used (main.py line 230) it looks like, for each pair of i/o training examples,
     the colour map of the input will be subtracted from the colour map of the output
    """
    return {key: value for key, value in colour_map_2.items() 
            if (key not in colour_map_1) or (colour_map_1[key] != value)}

def group_pairs_by_description(valid_pairs):
    """Group pairs by their description hash."""
    grouped = {}
    for pair in valid_pairs:
        description_hash = pair[0]["description_hash"]
        if description_hash not in grouped:
            grouped[description_hash] = []
        grouped[description_hash].append(pair)
    return grouped

def create_delta_maps(grouped_pairs):
    """Create delta maps for each pair."""
    delta_maps = []
    for description_hash, pairs in grouped_pairs.items():
        for pair in pairs:
            delta_map = subtract_colour_maps(pair[0]["colour_map"], pair[1]["colour_map"])
            delta_maps.append((description_hash, delta_map, pair))
    return delta_maps

def group_by_description_and_delta(delta_maps):
    """Group pairs by both description hash and delta map."""
    grouped = defaultdict(list)
    for description_hash, delta_map, original_pair in delta_maps:
        delta_map_key = frozenset(delta_map.items())
        grouped[(description_hash, delta_map_key)].append(original_pair)
    return grouped

def select_training_examples(grouped_pairs):
    """Select and shuffle training examples from grouped pairs."""
    training_examples = []
    for group in grouped_pairs.values():
        if len(group) >= 4:
            selected_pairs = random.sample(group, 4)
            training_examples.append(selected_pairs)
    
    for group in training_examples:
        random.shuffle(group)
    return training_examples

def get_augmented_training_examples(dataloader: DataLoader, id: str, visualize: bool = False):
    """Main function to get augmented training examples."""
    # Step 1: Get the sample and train examples
    #print("Step 1: Get the sample and train examples")
    sample = dataloader.get_specific_sample(id)
    train_examples = sample["train_examples"]

    # Step 2: Process each training example
    #print("Step 2: Process each training example")

    valid_pairs = []
    for train_example_count, train_example in enumerate(train_examples):
        # Prepare input and output
        prepared_input, prepared_output = prepare_example(train_example)
        
        # Generate augmentation pairs
        final_pairs = generate_augmentation_pairs(
            prepared_input, prepared_output, id, train_example_count
        )
        
        # Validate pairs
        valid_pairs.extend([pair for pair in final_pairs if validate_pair(pair)])
    
    # Step 3: Group and process pairs
    #print("Step 3: Group and process pairs")
    grouped_by_hash = group_pairs_by_description(valid_pairs)
    delta_maps = create_delta_maps(grouped_by_hash)
    grouped_final = group_by_description_and_delta(delta_maps)
    
    # Step 4: Create training examples
    #print("Step 4: Create training examples")
    training_examples = select_training_examples(grouped_final)
    
    # Step 5: Visualize results
    if visualize:
        project_strings = ProjectStrings()
        custom_colors = project_strings.CUSTOM_COLORS 
        visualize_all_training_examples(training_examples, custom_colors)
    
    return training_examples

def visualize_training_example(training_example, custom_colors, example_index):
    """Visualize a single training example."""
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
    """Visualize all training examples."""
    for i, training_example in enumerate(training_groups):
        visualize_training_example(training_example, custom_colors, i)


