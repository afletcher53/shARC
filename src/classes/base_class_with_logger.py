class BaseClassWithLogger:
    def __init__(self, log_file: str) -> None:
        from .logger_setup import LoggerSetup

        logger_name = self.__class__.__name__  # or __name__ to use the module name
        self.logger = LoggerSetup(name=logger_name, log_file=log_file).get_logger()
