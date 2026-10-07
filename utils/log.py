import logging
from pathlib import Path


class Logger(object):
    """Small file logger used by the training and forecasting scripts."""

    LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    def __init__(self, root_path, log_name, level="info"):
        self.root_path = Path(root_path)
        self.log_name = log_name
        self.level = self.LEVELS.get(level, logging.INFO)

    def get_logger(self):
        log_dir = self.root_path / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / (self.log_name + ".log")

        logger = logging.getLogger(self.log_name)
        logger.setLevel(self.level)
        logger.propagate = False

        # PyCharm can rerun a script in the same process. Clearing old handlers
        # prevents the same line from being written more than once.
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

        handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        return logger
