from itertools import permutations, product
import math
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
from typing import List, Tuple
import pandas as pd

from classes.project_strings import ProjectStrings


def find_similar_solutions(
    solution: np.ndarray, dataset, n: int = 5
) -> List[np.ndarray]:
    """
    Find n similar solution grids in the dataset, handling various input shapes and float values.
    Then add augmented versions of the original solution.

    Args:
        solution (np.ndarray): The input solution grid.
        dataset: The dataset containing all output grids.
        n (int): The number of similar solutions to find before adding augmentations.

    Returns:
        List[np.ndarray]: A list of similar solution grids and augmented versions of the original.
    """
    solution = preprocess_grid(solution)
    all_grids = [
        preprocess_grid(grid) for grid in dataset.get_all_output_grids()
    ]

    solution_features = extract_features(solution)
    similarities = [
        (grid, calculate_similarity(solution_features, extract_features(grid)))
        for grid in all_grids
        if not np.array_equal(grid, solution)
    ]

    similarities.sort(key=lambda x: x[1], reverse=True)
    similar_grids = [grid for grid, _ in similarities[:n]]

    augmented_solutions = generate_augmentations(solution)

    combined_grids = similar_grids.copy()
    for aug_solution in augmented_solutions:
        if not any(
            np.array_equal(aug_solution, grid) for grid in combined_grids
        ):
            combined_grids.append(aug_solution)

    colour_variations =[create_colour_blind_grids(solution) for _ in range(5)]
    combined_grids.extend([variation for variations in colour_variations for variation in variations])

    np.random.shuffle(combined_grids)

    original_solution_index = next(
        (
            i
            for i, grid in enumerate(combined_grids)
            if np.array_equal(grid, solution)
        ),
        None,
    )

    return combined_grids, original_solution_index

def create_colour_blind_grids(grid: np.ndarray) -> List[np.ndarray]:
    variation = grid.copy()
    mask = grid != 0
    variation[mask] = np.random.randint(1, 10, size=variation[mask].shape)
    return [variation]

def preprocess_grid(grid: np.ndarray) -> np.ndarray:
    """Preprocess the input grid to ensure consistent shape and data type."""
    grid = np.array(grid)
    if grid.shape == (1, 9, 9):
        grid = grid[0]
    return grid.astype(int)


def extract_features(grid: np.ndarray) -> Tuple[np.ndarray, int]:
    """Extract color distribution and size features from a grid."""
    flat_grid = grid.flatten()
    size = len(flat_grid)

    if np.issubdtype(flat_grid.dtype, np.floating):
        hist, _ = np.histogram(
            flat_grid,
            bins=np.arange(np.min(flat_grid), np.max(flat_grid) + 2, 1),
        )
        color_dist = hist / size
    else:
        color_dist = np.bincount(flat_grid.astype(int)) / size

    return color_dist, size


def calculate_similarity(
    features1: Tuple[np.ndarray, int], features2: Tuple[np.ndarray, int]
) -> float:
    """Calculate similarity between two grids based on their features."""
    color_dist1, size1 = features1
    color_dist2, size2 = features2

    max_colors = max(len(color_dist1), len(color_dist2))
    color_dist1_padded = np.pad(color_dist1, (0, max_colors - len(color_dist1)))
    color_dist2_padded = np.pad(color_dist2, (0, max_colors - len(color_dist2)))

    color_similarity = 1 - cosine(color_dist1_padded, color_dist2_padded)
    size_similarity = 1 - (abs(size1 - size2) / max(size1, size2))

    return color_similarity * size_similarity

def generate_color_permutations(grid: np.ndarray, available_colors: List[int]) -> List[np.ndarray]:
    unique_colors = set(np.unique(grid)) - {0, 1}

    color_permutations = list(permutations(available_colors, len(unique_colors)))
    
    permuted_grids = []
    
    colour_maps = []
    for color_perm in color_permutations:
        color_map = {int(k): v for k, v in zip(unique_colors, color_perm)}
        color_map[0] = 0 # 0 is black
        color_map[5] = 5 # 5 is the grey
        colour_maps.append(color_map)

        permuted_grid = np.vectorize(lambda x: color_map.get(x, x))(grid)
        permuted_grids.append(permuted_grid)
    
    return permuted_grids, colour_maps


def generate_augmentations(grid: List[np.ndarray], challenge_id: str, grid_type: str, train_example_count: int, plot: bool = False) -> dict:
    def augment_single_grid(g):
        augmentations = []
        for v_flip in [False, True]:
            for h_flip in [False, True]:
                for rotation in range(4):
                    aug = g.copy()
                    if v_flip:
                        aug = np.flipud(aug)
                    if h_flip:
                        aug = np.fliplr(aug)
                    
                    flip_status = []
                    if v_flip:
                        flip_status.append("vertical flip")
                    if h_flip:
                        flip_status.append("horizontal flip")
                    flip_description = " and ".join(flip_status) if flip_status else "no flip"
                    
                    aug = np.rot90(aug, k=rotation)
                    augmentations.append((f"{flip_description}, rotated {rotation * 90}°", aug))
        return augmentations


    augmented_grids = [augment_single_grid(g) for g in grid]
    
    unique_augmentations = []
    for i, aug in enumerate(augmented_grids):
        for description, grid in aug:
            if not any(np.array_equal(grid, unique[1]) for unique in unique_augmentations):
                unique_augmentations.append((description, grid))

    
        
    available_colors = [2, 3, 4, 5, 6, 7, 8, 9]
    lookup_list = {}
    
    for description, unique_grid in unique_augmentations:
        color_permuted_grids, colour_maps = generate_color_permutations(unique_grid, available_colors)
        lookup_list[description] = {
            "original": unique_grid,
            "color_permutations": color_permuted_grids,
            "colour_maps": colour_maps
        }
        lookup_list[description]["colour_maps"] = colour_maps

    colour_dict = {}
    for description, data in lookup_list.items():
        for i, colour_map in enumerate(data["colour_maps"]):
            identifier = hash(str(colour_map)+str(description))
            colour_dict[identifier] = {
                "description": description,
                "description_hash": hash(str(description)),
                "challenge_id": challenge_id,
                "grid_type": grid_type,
                "original": data["original"],
                "permutation": data["color_permutations"][i],
                "colour_map": colour_map, 
                "colour_map_hash": hash(str(colour_map))
            }
    
    if plot:
        save_augmentations_on_same_plot(grid, augmented_grids, f"{challenge_id}_augmentations_{grid_type}_{train_example_count}.png")
        save_augmentations_on_same_plot(grid, [unique_augmentations], f"{challenge_id}_unique_augmentations_{grid_type}_{train_example_count}.png")
        save_colour_dict_to_png(colour_dict, f"{challenge_id}_colour_dict_{grid_type}_{train_example_count}.png")

    return colour_dict

import matplotlib.colors as mcolors

def save_colour_dict_to_png(colour_dict, save_path, max_cols=5, max_rows=20):
    ps = ProjectStrings()
    num_grids = len(colour_dict)
    grids_per_figure = max_cols * max_rows
    num_figures = math.ceil(num_grids / grids_per_figure)

    color_map = mcolors.ListedColormap(ps.CUSTOM_COLORS)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, 10, 1), color_map.N)

    for fig_num in range(num_figures):
        start_index = fig_num * grids_per_figure
        end_index = min((fig_num + 1) * grids_per_figure, num_grids)
        
        num_grids_this_fig = end_index - start_index
        num_cols = min(num_grids_this_fig, max_cols)
        num_rows = min(max_rows, math.ceil(num_grids_this_fig / num_cols))
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
        if num_rows == 1 and num_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, (identifier, data) in enumerate(list(colour_dict.items())[start_index:end_index]):
            if i < len(axes):
                ax = axes[i]
                im = ax.imshow(data['permutation'], cmap=color_map, norm=norm)
                ax.set_title(f"{data['description']}\nID: {identifier}", fontsize=8)
                ax.axis('off')
        
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax, ticks=range(10))
        cbar.set_ticklabels(range(10))
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        
        if num_figures > 1:
            fig_save_path = f"{save_path[:-4]}_{fig_num+1}.png"
        else:
            fig_save_path = save_path
        
        plt.savefig(fig_save_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Saved colour_dict visualization to {fig_save_path}")


def save_augmentations_on_same_plot(grid: np.ndarray, augmented_grids: List[List[Tuple[str, np.ndarray]]], save_path: str):
    num_augmentations = len(augmented_grids[0])
    fig, axes = plt.subplots(1, num_augmentations + 1, figsize=(20, 5))
    
    ps = ProjectStrings()
    color_map = mcolors.ListedColormap(ps.CUSTOM_COLORS)
    
    norm = mcolors.BoundaryNorm(np.arange(-0.5, 10, 1), color_map.N)
    
    im = axes[0].imshow(grid, cmap=color_map, norm=norm)
    axes[0].set_title("Original Grid")
    axes[0].axis('off')

    # Plot augmented grids
    for j, (description, aug) in enumerate(augmented_grids[0], 1):
        axes[j].imshow(aug, cmap=color_map, norm=norm)
        axes[j].set_title(f"{description}\n{aug.shape}", fontsize=8)
        axes[j].axis('off')

    # Add a colorbar to the right of the entire plot
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=range(10))
    cbar.set_ticklabels(range(10))

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the rect to make room for the colorbar
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved augmentations visualization to {save_path}")


def create_dataframe_from_lookup(lookup_list: dict) -> pd.DataFrame:
    rows = []
    for description, data in lookup_list.items():
        original_input = data["original"]
        color_permutations = data["color_permutations"]
        colour_maps = data["colour_maps"]
        
        for permuted_grid, colour_map in zip(color_permutations, colour_maps):
            rows.append({
                "Description (Geometric Augmentation)": description,
                "Original Input": original_input,
                "Permutation": permuted_grid,
                "Colour Map": colour_map
            })

    df = pd.DataFrame(rows)
    return df


def mask_grid(grid: np.ndarray, mask_ratio: float = 0.2) -> np.ndarray:
    """Apply random masking to the grid."""
    mask = np.random.choice(
        [True, False], size=grid.shape, p=[mask_ratio, 1 - mask_ratio]
    )
    masked_grid = grid.copy()
    masked_grid[mask] = 0
    return masked_grid


def jitter_grid(grid: np.ndarray) -> np.ndarray:
    """Apply color jittering to the grid."""
    return np.clip(grid + np.random.randint(-1, 2, size=grid.shape), 0, None)


def scale_grid(grid: np.ndarray, scale: int = 2) -> np.ndarray:
    """Scale up the grid."""
    return np.kron(grid, np.ones((scale, scale)))


def crop_grid(grid: np.ndarray) -> np.ndarray:
    """Crop the grid by removing the last row and column."""
    return grid[:-1, :-1] if grid.shape[0] > 1 and grid.shape[1] > 1 else grid

# colour augmentations
# Original colour map
# dictionary remapping of colours
# don't change black or grey
# flip x rotate x colour is the possible combinations