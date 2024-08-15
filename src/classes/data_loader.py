import os
import json
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

    def load_dataset(self, dataset_type: str) -> dict:
        self._validate_dataset_type(dataset_type)
        self.logger.info(f"Loading dataset of type: {dataset_type}")
        self.dataset_type = dataset_type

        if dataset_type == "test":
            self.logger.info("Fetching test dataset without solutions.")
            return self._load_test_dataset()

        challenges_path, solutions_path = self._get_dataset_locations()
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

    def plot_solution(
        self, grid: np.ndarray, file_name: str, output_dir: str = "output"
    ):
        """
        Given a grid as a numpy array, plots the grid as an image.

        :param grid: A numpy array representing the grid to be plotted.
        :param output_dir: Directory where the plot image will be saved.
        """

        plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap="viridis", interpolation="nearest")
        plt.axis("off")

        plt.title(f"Grid for {file_name}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file_path = os.path.join(output_dir, f"{file_name}.png")

        plt.savefig(output_file_path)
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
