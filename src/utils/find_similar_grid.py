from itertools import permutations, product
import numpy as np
from scipy.spatial.distance import cosine
from typing import List, Tuple
import pandas as pd


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
    # Gather unique colors in the grid excluding 0 and 1
    unique_colors = set(np.unique(grid)) - {0, 1}
    
    # Generate permutations of the unique colors from the available colors
    color_permutations = list(permutations(available_colors, len(unique_colors)))
    
    permuted_grids = []
    
    colour_maps = []
    for color_perm in color_permutations:
        color_map = dict(zip(unique_colors, color_perm))
        # Add mappings for colors 0 and 1 to stay unchanged
        color_map[0] = 0
        color_map[1] = 1
        colour_maps.append(color_map)

        # Create a new grid by replacing the colors using the color_map
        permuted_grid = np.vectorize(lambda x: color_map.get(x, x))(grid)
        permuted_grids.append(permuted_grid)
    
    return permuted_grids, colour_maps


def generate_augmentations(grid: List[np.ndarray]) -> dict:
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
    
    # limit to unique augmentations
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
        # append the colour maps to the lookup list

    df_lookup = create_dataframe_from_lookup(lookup_list)
    return df_lookup


def create_dataframe_from_lookup(lookup_list: dict) -> pd.DataFrame:
    rows = []
    
    # Iterate over each description in the lookup list
    for description, data in lookup_list.items():
        original_input = data["original"]
        color_permutations = data["color_permutations"]
        colour_maps = data["colour_maps"]
        
        # For each color permutation, we add a row
        for permuted_grid, colour_map in zip(color_permutations, colour_maps):
            rows.append({
                "Description (Geometric Augmentation)": description,
                "Original Input": original_input,
                "Permutation": permuted_grid,
                "Colour Map": colour_map
            })
    
    # Create the DataFrame
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