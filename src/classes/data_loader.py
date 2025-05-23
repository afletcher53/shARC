import os
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from .base_class_with_logger import BaseClassWithLogger
from classes.project_strings import ProjectStrings


class DataLoader(BaseClassWithLogger):
    def __init__(self) -> None:
        super().__init__("data_loader.log")
        self.logger.info("Data Loader initialized.")
        self.strings = ProjectStrings()
        self.dataset = None

    def show_available_datasets(self) -> list:
        self.logger.info(
            f"Available datasets: {self.strings.available_datasets}"
        )
        return self.strings.available_datasets

    def _validate_dataset_type(self, dataset_type: str) -> bool:
        if dataset_type not in self.strings.available_datasets:
            self.logger.error(f"Invalid dataset type: {dataset_type}")
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        return True

    def load_dataset(self, dataset_type: str, dataset_locations_override=()) -> dict:
        self._validate_dataset_type(dataset_type)
        self.logger.info(f"Loading dataset of type: {dataset_type}")
        self.dataset_type = dataset_type

        if dataset_type == "test":
            self.logger.info("Fetching test dataset without solutions.")
            return self._load_test_dataset()

        if not dataset_locations_override:
            challenges_path, solutions_path = self._get_dataset_locations()
        else:
            challenges_path, solutions_path = dataset_locations_override

        self.logger.info(
            f"Loading dataset from {challenges_path} and {solutions_path}"
        )

        with open(challenges_path, "r") as f:
            challenges = json.load(f)
            self.logger.info(f"Challenges loaded from {challenges_path}")

        with open(solutions_path, "r") as f:
            solutions = json.load(f)
            self.logger.info(f"Solutions loaded from {solutions_path}")
        self.logger.info("Flattening dataset.")
        self.dataset = self._flatten_dataset(challenges, solutions)
        self.logger.info("Dataset loaded and flattened.")
        self.logger.info(
            f"{len(self.dataset)} {dataset_type} challenges loaded and flattened."
        )
        return self.dataset

    def _get_dataset_locations(self):
        if self.dataset_type == "training":
            self.logger.info("Fetching training dataset locations.")
            return (
                os.path.join(
                    os.getcwd(), self.strings.get_training_challenges_data_path
                ),
                os.path.join(
                    os.getcwd(), self.strings.get_training_solutions_data_path
                ),
            )
        elif self.dataset_type == "evaluation":
            self.logger.info("Fetching evaluation dataset locations.")
            return (
                os.path.join(
                    os.getcwd(),
                    self.strings.get_evaluation_challenges_data_path,
                ),
                os.path.join(
                    os.getcwd(), self.strings.get_evaluation_solutions_data_path
                ),
            )
        elif self.dataset_type == "test":
            self.logger.info("Fetching test dataset location.")
            return (
                os.path.join(
                    os.getcwd(), self.strings.get_test_challenges_data_path
                ),
            )

    def _flatten_dataset(self, challenges: dict, solutions: dict) -> dict:
        flattened_dataset = {}
        for uuid, challenge_data in challenges.items():
            if uuid in solutions:
                test_input = challenge_data.get("test", [])[0]["input"]
                train_examples = [
                    {"input": ex["input"], "output": ex["output"]}
                    for ex in challenge_data["train"]
                ]
                flattened_dataset[uuid] = {
                    "test_input": test_input,
                    "train_examples": train_examples,
                    "solution": solutions[uuid],
                }
        return flattened_dataset

    def _load_test_dataset(self) -> dict:
        test_challenges_path = self._get_dataset_locations()[0]
        self.logger.info(f"Loading test dataset from {test_challenges_path}")

        with open(test_challenges_path, "r") as f:
            challenges = json.load(f)
            self.logger.info(
                f"Test challenges loaded from {test_challenges_path}"
            )
        self.logger.info("Flattening test dataset.")
        self.dataset = {
            uuid: {"test_input": challenge_data.get("test", [])[0]["input"]}
            for uuid, challenge_data in challenges.items()
        }
        self.logger.info(
            f"{len(self.dataset)} test challenges loaded and flattened."
        )
        return self.dataset

    def randomly_sample_datapoints(self, sample_size: int) -> dict:
        if sample_size < 1:
            self.logger.error("Sample size should be greater than 0.")
            raise ValueError("Sample size should be greater than 0.")
        if sample_size > len(self.dataset):
            self.logger.error(
                "Sample size should be less than the size of the dataset."
            )
            raise ValueError(
                "Sample size should be less than the size of the dataset."
            )

        if not hasattr(self, "dataset_type"):
            self.logger.error(
                "Dataset type not set, needs to be set before sampling. Try loading a dataset first."
            )
            raise ValueError(
                "Dataset type not set, needs to be set before sampling. Try loading a dataset first."
            )

        unique_keys = set(self.dataset.keys())
        self.logger.info(f"len({self.dataset_type}): {len(unique_keys)}")

        import random

        random.seed(self.strings.get_seed)
        unique_keys_list = sorted(unique_keys)
        selected_challenges = random.sample(unique_keys_list, sample_size)
        self.logger.info(f"Randomly selected challenges: {selected_challenges}")

        return {uuid: self.dataset[uuid] for uuid in selected_challenges}

    def get_specific_sample(self, challenge_id: str) -> dict:
        if not hasattr(self, "dataset_type"):
            self.logger.error(
                "Dataset type not set, needs to be set before sampling. Try loading a dataset first."
            )
            raise ValueError(
                "Dataset type not set, needs to be set before sampling. Try loading a dataset first."
            )

        if challenge_id not in self.dataset:
            self.logger.error(
                f"Challenge with id {challenge_id} not found in the dataset."
            )
            raise ValueError(
                f"Challenge with id {challenge_id} not found in the dataset."
            )
        return self.dataset[challenge_id]

    def plot_train_and_test_examples(
        self, sample_data: dict, output_dir: str = "output"
    ):
        """
        Plots the train and test examples for a given sample.

        :param sample_data: A dictionary with 'train_examples' and 'test_input'.
        :param output_dir: Directory where the plot image will be saved.
        """
        if self.dataset is None:
            self.logger.error(
                "Dataset not loaded. Load a dataset before plotting."
            )
            raise ValueError(
                "Dataset not loaded. Load a dataset before plotting."
            )
        if self.dataset_type == "test":
            self.logger.error(
                "Cannot plot train and test examples for test dataset. "
            )
            raise ValueError(
                "Cannot plot train and test examples for test dataset."
            )

        for sample_id, data in sample_data.items():
            train_examples = data["train_examples"]
            num_train_examples = len(train_examples)

            fig, axs = plt.subplots(
                num_train_examples + 1,
                2,
                figsize=(10, 5 * (num_train_examples + 1)),
            )

            for idx, ex in enumerate(train_examples):
                input_grid = np.array(ex["input"])
                output_grid = np.array(ex["output"])

                axs[idx, 0].imshow(
                    input_grid, cmap="viridis", interpolation="nearest"
                )
                axs[idx, 0].set_title(f"Train Input {idx+1}")
                axs[idx, 0].axis("off")

                axs[idx, 1].imshow(
                    output_grid, cmap="viridis", interpolation="nearest"
                )
                axs[idx, 1].set_title(f"Train Output {idx+1}")
                axs[idx, 1].axis("off")

            axs[num_train_examples, 0].imshow(
                np.array(data["test_input"]),
                cmap="viridis",
                interpolation="nearest",
            )
            axs[num_train_examples, 0].set_title("Test Input")
            axs[num_train_examples, 0].axis("off")

            axs[num_train_examples, 1].axis("off")

            fig.suptitle(f"Sample: {sample_id}")

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_file_path = os.path.join(
                output_dir, f"{sample_id}_train_and_test_example.png"
            )
            plt.savefig(output_file_path)
            plt.close(fig)

    def plot_inference_results(
        self, instance: dict, generated_output: np.ndarray, output_dir: str = "output"
    ):
        """
        Plots the train examples, test input, and generated output for a single instance.

        :param instance: Dictionary containing sample data with 'train_examples' and 'test_input'.
        :param generated_output: The generated output grid (numpy array).
        :param output_dir: Directory where the plot images will be saved.
        """
        sample_id = "current_sample"  # Default ID since we're plotting one at a time

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        train_examples = instance["train_examples"]
        num_train_examples = len(train_examples)
        generated_output = np.array(generated_output)

        fig, axs = plt.subplots(
            num_train_examples + 2,
            2,
            figsize=(10, 5 * (num_train_examples + 2)),
        )

        # Plot train examples
        for train_idx, ex in enumerate(train_examples):
            input_grid = np.array(ex["input"])
            output_grid = np.array(ex["output"])

            axs[train_idx, 0].imshow(
                input_grid, cmap="viridis", interpolation="nearest"
            )
            axs[train_idx, 0].set_title(f"Train Input {train_idx + 1}")
            axs[train_idx, 0].axis("off")

            axs[train_idx, 1].imshow(
                output_grid, cmap="viridis", interpolation="nearest"
            )
            axs[train_idx, 1].set_title(f"Train Output {train_idx + 1}")
            axs[train_idx, 1].axis("off")

        # Plot test input
        test_input_grid = np.array(instance["test_input"])
        axs[num_train_examples, 0].imshow(
            test_input_grid, cmap="viridis", interpolation="nearest"
        )
        axs[num_train_examples, 0].set_title("Test Input")
        axs[num_train_examples, 0].axis("off")
        axs[num_train_examples, 1].axis("off")  # Hide the unused subplot

        # Plot generated output
        axs[num_train_examples + 1, 0].imshow(
            generated_output, cmap="viridis", interpolation="nearest"
        )
        axs[num_train_examples + 1, 0].set_title("Generated Output")
        axs[num_train_examples + 1, 0].axis("off")
        axs[num_train_examples + 1, 1].axis("off")  # Hide the unused subplot

        fig.suptitle(f"Sample: {sample_id}")

        output_file_path = os.path.join(
            output_dir, f"{sample_id}_inference_result.png"
        )
        plt.savefig(output_file_path)
        plt.close(fig)
        self.logger.info(f"Inference results plot saved as {output_file_path}")
    def plot_solution(
        self, grid: np.ndarray, file_name: str, output_dir: str = "output"
    ):
        """
        Given a grid as a numpy array, plots the grid as an image using ARC custom colors.
        
        :param grid: A numpy array representing the grid to be plotted.
        :param file_name: Name for the output file
        :param output_dir: Directory where the plot image will be saved.
        """
        # Create discrete colormap using ProjectStrings custom colors
        colors = self.strings.CUSTOM_COLORS
        n_colors = len(colors)
        
        # Create figure and axis
        plt.figure(figsize=(6, 6))
        
        # Create a masked array to handle values outside our color range
        masked_grid = np.ma.masked_where(grid >= n_colors, grid)
        
        # Plot the grid using custom colors
        plt.imshow(
            masked_grid,
            cmap=matplotlib.colors.ListedColormap(colors),
            interpolation='nearest',
            vmin=0,
            vmax=len(colors)-1
        )
        
        # Add gridlines
        plt.grid(True, which='both', color='grey', linewidth=0.5, alpha=0.5)
        plt.xticks(np.arange(-0.5, grid.shape[1], 1), [])
        plt.yticks(np.arange(-0.5, grid.shape[0], 1), [])
        
        plt.title(f"Grid for {file_name}")
        plt.axis("on")  # Show the axis for grid lines
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save and show the plot
        output_file_path = os.path.join(output_dir, f"{file_name}.png")
        plt.savefig(output_file_path, bbox_inches='tight', dpi=300)
        # plt.show()
        plt.close()
        
        self.logger.info(f"Solution grid saved as {output_file_path}")

    def plot_multiple_solutions(self, grids: list, output_dir: str = "output"):
        """
        Given a list of grids as numpy arrays, plots all the grids as images.

        :param grids: A list of numpy arrays representing the grids to be plotted.
        :param output_dir: Directory where the plot images will be saved.
        """
        output_dir = f"./output/{output_dir}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, grid in enumerate(grids):
            plt.figure(figsize=(6, 6))
            plt.imshow(grid, cmap="viridis", interpolation="nearest")
            plt.axis("off")

            plt.title(f"Grid {idx}")

            output_file_path = os.path.join(output_dir, f"grid_{idx}.png")

            plt.savefig(output_file_path)
            plt.close()

        self.logger.info(f"{len(grids)} solution grids saved in {output_dir}")

    def get_all_output_grids(self):
        if self.dataset is None:
            self.logger.error(
                "Dataset not loaded. Load a dataset before plotting."
            )
            raise ValueError(
                "Dataset not loaded. Load a dataset before plotting."
            )

        all_output_grids = []
        for _, data in self.dataset.items():
            all_output_grids.append(data["solution"][0])

        self.logger.info(
            f"Total number of output grids: {len(all_output_grids)}"
        )

        return all_output_grids


    def plot_single_train_examples(
        self, sample_data: dict, output_dir: str = "output"
    ):
        """
        Plots the train and test examples for a given sample.

        :param sample_data: A dictionary with 'train_examples' and 'test_input'.
        :param output_dir: Directory where the plot image will be saved.
        """
        if self.dataset is None:
            self.logger.error(
                "Dataset not loaded. Load a dataset before plotting."
            )
            raise ValueError(
                "Dataset not loaded. Load a dataset before plotting."
            )
        if self.dataset_type == "test":
            self.logger.error(
                "Cannot plot train and test examples for test dataset. "
            )
            raise ValueError(
                "Cannot plot train and test examples for test dataset."
            )

        for sample_id, data in sample_data.items():
            train_examples = data["train_examples"]
            num_train_examples = len(train_examples)
            
           
            for idx, ex in enumerate(train_examples):
                output_file_path = os.path.join(
                    output_dir, f"{sample_id}_train_{idx}_input.png"
                )
                input_grid = np.array(ex["input"])
                output_grid = np.array(ex["output"])



                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(
                    input_grid, cmap="viridis", interpolation="nearest"
                )
                axs[0].set_title(f"Train Input {idx+1}")
                axs[0].axis("off")

                axs[1].imshow(
                    output_grid, cmap="viridis", interpolation="nearest"
                )
                axs[1].set_title(f"Train Output {idx+1}")
                axs[1].axis("off")
                
               
                plt.savefig(output_file_path)
                plt.close(fig)
