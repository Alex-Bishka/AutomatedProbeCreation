from datetime import datetime
import logging
import os
import uuid

from paths import LOGS_DIR


def setup_logger(log_to_file=True):
    logger = logging.getLogger("probe-factory")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        os.makedirs(LOGS_DIR, exist_ok=True)
        unique_id = uuid.uuid4()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(LOGS_DIR, f"{timestamp}_{unique_id.hex[:8]}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()