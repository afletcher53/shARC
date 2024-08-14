import logging
from typing import Optional
import os


class LoggerSetup:
    def __init__(
        self, name: str, log_file: Optional[str] = None, level: int = logging.DEBUG
    ) -> None:
        self.name = name
        self.log_file = f"./logs/{log_file}" if log_file else None
        self.level = level
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)

        logger.handlers.clear()

        if self.log_file:
            log_dir = os.path.dirname(self.log_file)
            os.makedirs(log_dir, exist_ok=True)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(self.level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def get_logger(self) -> logging.Logger:
        return self.logger


class BaseClassWithLogger:
    def __init__(self, log_file: str) -> None:
        logger_name = self.__class__.__name__
        self.logger = LoggerSetup(name=logger_name, log_file=log_file).get_logger()


from classes.data_loader import DataLoader


def main():
    dl = DataLoader()
    print(dl.show_available_datasets())
    print(dl)


if __name__ == "__main__":
    main()
