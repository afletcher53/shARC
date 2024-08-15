import numpy as np
from scipy.spatial.distance import cosine
from typing import List, Tuple


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


def generate_augmentations(grid: List[np.ndarray]) -> List[np.ndarray]:
    """Generate augmented versions of the input grids."""
    augmented = []
    augmented.extend(
        [
            grid,
            np.flipud(grid),
            np.fliplr(grid),
            np.rot90(grid),
            np.rot90(grid, k=2),
            np.rot90(grid, k=3),
            mask_grid(grid),
            jitter_grid(grid),
            scale_grid(grid),
            crop_grid(grid),
        ]
    )
    return augmented


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
