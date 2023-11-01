import logging
from multiprocessing import current_process

import colorlog


def get_color_logger(logger_name=None):
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


def get_gpu_id(num_gpus: int, num_outer_processes: int):
    name = current_process().name
    values = name.split("-")[-1].split(":")
    if len(values) == 2:
        outer_process, inner_process = values
    else:
        outer_process, inner_process = values[0], 0
    proc_num = int(outer_process) * num_outer_processes + int(inner_process)
    gpu_id = proc_num % num_gpus
    return gpu_id
