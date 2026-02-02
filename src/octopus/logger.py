import os
import logging

from typing import Optional

def get_logger(log_dir: Optional[str] = None):
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "log.txt")
    else:
        log_file = None
    logging.basicConfig(
        level=logging.INFO,
        format="[octopus] %(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()] if log_file else [logging.StreamHandler()],
    )
    return logging.getLogger(__name__)