import logging
from typing import Optional

class LoggerSetup:
    def __init__(self, name: str, log_file: Optional[str] = None, level: int = logging.DEBUG) -> None:
        self.name = name
        self.log_file = f'logs/{log_file}' if log_file else None
        self.level = level
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        if not logger.hasHandlers():
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(self.level)
            
            if self.log_file:
                file_handler = logging.FileHandler(self.log_file)
                file_handler.setLevel(self.level)
                logger.addHandler(file_handler)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            stream_handler.setFormatter(formatter)
            
            if self.log_file:
                file_handler.setFormatter(formatter)
            
            logger.addHandler(stream_handler)
            if self.log_file:
                logger.addHandler(file_handler)
        
        return logger

    def get_logger(self) -> logging.Logger:
        return self.logger
