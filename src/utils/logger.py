# src/utils/logger.py

import logging
import os

def setup_logger(name: str, log_file: str = "logs/app.log", level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Stream (console) handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Avoid duplicate logs
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
