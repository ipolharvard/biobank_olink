import logging
from multiprocessing import current_process
from typing import Optional

import colorlog

_default_handler: Optional[logging.Handler] = None
_log_colors = {
    "DEBUG": "cyan",
    "INFO": "yellow",
    "WARNING": "red",
    "ERROR": "red",
    "CRITICAL": "red",
}


def get_logger():
    global _default_handler

    logger_name = __name__.split(".")[0]
    logger = logging.getLogger(logger_name)
    if _default_handler is None:
        _setup_handler()
        logger.addHandler(_default_handler)
        logger.setLevel(logging.INFO)
    return logger


def _setup_handler():
    global _default_handler

    _default_handler = colorlog.StreamHandler()
    _default_handler.setLevel(logging.INFO)

    header = "[%(levelname)1.1s %(asctime)s]"
    message = "%(message)s"
    formatter = colorlog.ColoredFormatter(
        f"%(green)s{header}%(reset)s %(log_color)s{message}%(reset)s", log_colors=_log_colors
    )
    _default_handler.setFormatter(formatter)


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
