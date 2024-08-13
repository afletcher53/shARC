from .base_class_with_logger import BaseClassWithLogger  # Adjust the import to your file structure

class ProjectStrings(BaseClassWithLogger):
    def __init__(self) -> None:
        super().__init__("project_strings.log")
        self.logger.info("ProjectStrings initialized.")