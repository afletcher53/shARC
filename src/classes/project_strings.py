from .base_class_with_logger import BaseClassWithLogger


class ProjectStrings(BaseClassWithLogger):
    def __init__(self) -> None:
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
        return ["training", "evaluation", "test"]

    def get_data_path(self, dataset_type: str, data_kind: str) -> str:
        key = f"{dataset_type}_{data_kind}"
        return self._base_path + self._datasets.get(key, "")

    @property
    def get_training_challenges_data_path(self) -> str:
        return self.get_data_path("training", "challenges")

    @property
    def get_training_solutions_data_path(self) -> str:
        return self.get_data_path("training", "solutions")

    @property
    def get_evaluation_challenges_data_path(self) -> str:
        return self.get_data_path("evaluation", "challenges")

    @property
    def get_evaluation_solutions_data_path(self) -> str:
        return self.get_data_path("evaluation", "solutions")

    @property
    def get_test_challenges_data_path(self) -> str:
        return self.get_data_path("test", "challenges")

    @property
    def get_test_solutions_data_path(self) -> str:
        return self.get_data_path("test", "solutions")

    @property
    def get_seed(self) -> int:
        return 42

    def __str__(self) -> str:
        return f"ProjectStrings: {self.available_datasets}, {self.get_seed}, {self.get_training_challenges_data_path}"
