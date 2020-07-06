import os
import logging


def check_path(*paths):
    path = os.path.join(*paths)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_logger(name):
    LOGGER = logging.getLogger(name)

    # Create handlers
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
    c_handler.setFormatter(c_format)

    # Add handlers to the logger
    LOGGER.addHandler(c_handler)

    return LOGGER
