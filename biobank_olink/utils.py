import logging

import colorlog


def get_color_logger(logger_name="experiment"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    ch = colorlog.StreamHandler()
    ch.setLevel(logging.INFO)
    log_colors = {
        "DEBUG": "cyan",
        "INFO": "yellow",
        "WARNING": "red",
        "ERROR": "red",
        "CRITICAL": "red",
    }

    header = "[%(levelname)1.1s %(asctime)s]"
    message = "%(message)s"
    formatter = colorlog.ColoredFormatter(
        f"%(green)s{header}%(reset)s %(log_color)s{message}%(reset)s", log_colors=log_colors
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
