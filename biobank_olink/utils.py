import logging
from multiprocessing import current_process
from typing import Optional

import colorlog

from .constants import Model, Panel

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
    if name == "MainProcess":
        return 0
    values = name.split("-")[-1].split(":")
    outer_process, inner_process = values if len(values) == 2 else (values[0], 0)
    proc_num = num_outer_processes * int(inner_process) + int(outer_process)
    gpu_id = proc_num % num_gpus
    return gpu_id


def get_study_name(
        exp_name: str,
        model: Optional[Model] = None,
        panel: Optional[Panel] = None,
        nan_th: Optional[float] = None,
        corr_th: Optional[float] = None,
):
    study_name = exp_name
    if model is not None:
        study_name += f"_{model.value}"
    if nan_th:
        study_name += f"_nan{nan_th}"
    if corr_th:
        study_name += f"_corr{corr_th}"
    if panel != Panel.ALL:
        study_name += f"_{panel.value.lower()}"
    return study_name
