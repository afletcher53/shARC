"""This file contains the ProjectStrings class, which is used to store and retrieve project-specific strings and paths.

These are immutable and are not expected to change.
"""

from .base_class_with_logger import BaseClassWithLogger
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="keys.env")

class ProjectStrings(BaseClassWithLogger):
    """This class is used to store and retrieve project-specific strings and paths.
    
    These are immutable and are not expected to change.
    """
    def __init__(self) -> None:
        """Initializes the ProjectStrings class.
        
        Args:
            None
        
        Returns:
            None
        """
        super().__init__("project_strings.log")
        self.logger.info("ProjectStrings initialized.")
        self._base_path = "data/arc-agi_"
        self._datasets = {
            "training_challenges": "training_challenges.json",
            "training_solutions": "training_solutions.json",
            "evaluation_challenges": "evaluation_challenges.json",
            "evaluation_solutions": "evaluation_solutions.json",
            "test_challenges": "test_challenges.json",
            "test_solutions": "test_solutions.json",
        }

    @property
    def available_datasets(self) -> list:
        """Returns the available datasets for the project.
        
        Returns:
            list: The available datasets for the project.
        """
        return ["training", "evaluation", "test"]

    def get_data_path(self, dataset_type: str, data_kind: str) -> str:
        """Returns the data path for the given dataset type and data kind.

        Args:
            dataset_type (str): The type of dataset.
            data_kind (str): The kind of data.
        
        Returns:
            str: The data path for the given dataset type and data kind.
        """
        key = f"{dataset_type}_{data_kind}"
        return self._base_path + self._datasets.get(key, "")

    @property
    def get_training_challenges_data_path(self) -> str:
        """Returns the data path for the training challenges dataset.
        
        Returns:
            str: The data path for the training challenges dataset.
        """
        return self.get_data_path("training", "challenges")

    @property
    def get_training_solutions_data_path(self) -> str:
        """Returns the data path for the training solutions dataset.
        
        Returns:
            str: The data path for the training solutions dataset.
        """
        return self.get_data_path("training", "solutions")

    @property
    def get_evaluation_challenges_data_path(self) -> str:
        """Returns the data path for the evaluation challenges dataset.
        
        Returns:
            str: The data path for the evaluation challenges dataset.
        """
        return self.get_data_path("evaluation", "challenges")

    @property
    def get_evaluation_solutions_data_path(self) -> str:
        """Returns the data path for the evaluation solutions dataset.

        Returns:
            str: The data path for the evaluation solutions dataset.
        """
        return self.get_data_path("evaluation", "solutions")

    @property
    def get_test_challenges_data_path(self) -> str:
        """Returns the data path for the test challenges dataset.
        
        Returns:
            str: The data path for the test challenges dataset.
        """
        return self.get_data_path("test", "challenges")

    @property
    def get_test_solutions_data_path(self) -> str:
        """Returns the data path for the test solutions dataset.
        
        Returns:
            str: The data path for the test solutions dataset.
        """
        return self.get_data_path("test", "solutions")

    @property
    def get_seed(self) -> int:
        """Returns the seed for the project.
        
        Returns:
            int: The seed for the project.
        """
        return 42

    def __str__(self) -> str:
        """Returns a string representation of the ProjectStrings class.
        
        Returns:
            str: A string representation of the ProjectStrings class.
        """
        return f"ProjectStrings: {self.available_datasets}, {self.get_seed}, {self.get_training_challenges_data_path}"

    @property
    def CUSTOM_COLORS(self) -> list:
        """Returns the custom colors for the project.
        
        Returns:
            list: The custom colors for the project.
        """
        return  [
        'black',    # 0
        'blue',     # 1
        'red',      # 2
        'green',    # 3
        'yellow',   # 4
        'grey',     # 5
        'purple',   # 6
        'orange',   # 7
        'lightblue', # 8
        'brown'     # 9
    ]

    @property
    def HF_TOKEN(self) -> str:
        """Returns the Hugging Face token for the project.
        
        Returns:
            str: The Hugging Face token for the project.
        """
        return os.environ.get('HUGGING_FACE_TOKEN', '')

    @property
    def HF_HOME(self) -> str:
        """Returns the HF_HOME for the project.
        
        Returns:
            str: The HF_HOME for the project.
        """
        return 'hf_cache_dir/'
