import json
from types import SimpleNamespace
from typing import Optional

from click import command, option, Choice

from biobank_olink.constants import SEED
from biobank_olink.exp_two_extremes import run_two_extremes_experiment, get_data
from biobank_olink.exp_two_extremes.constants import RESULTS_DIR, Target, Model, Panel
from biobank_olink.utils import get_logger

logger = get_logger()


@command
@option("--target", type=Choice([t.value for t in Target]))
@option("--model", type=Choice([m.value for m in Model]))
@option("--panel", default=Panel.ALL.value, show_default=True,
        type=Choice([p.value for p in Panel]))
@option(
    "--threshold",
    type=float,
    default=0.35,
    show_default=True,
    help="Upper and lower percentiles to create the two extremes of the target variable.",
)
@option(
    "--nan_th",
    type=float,
    help="Threshold of the maximal NaN ratio, columns and rows above this level are discarded.",
)
@option(
    "--corr_th",
    type=float,
    help="Threshold of the maximal correlation, columns that correlate stronger are discarded.",
)
@option(
    "--outer_splits",
    type=int,
    default=2,
    show_default=True,
    metavar="N",
    help="The number of outer splits of dataset (>1).",
)
@option(
    "--inner_splits",
    type=int,
    default=2,
    show_default=True,
    metavar="N",
    help="The number of inner splits in each outer split, the average score is used as Optuna "
         "objective (>1)."
)
@option("--n_trials", type=int, default=5, metavar="N", show_default=True,
        help="The number of Optuna trials.")
@option(
    "--no_opt",
    is_flag=True,
    default=False,
    help="do not optimize hyperparameters, evaluate best trials only",
)
@option(
    "-w",
    "--optuna_n_workers",
    type=int,
    default=1,
    metavar="N",
    show_default=True,
    help="The number of Optuna workers.",
)
@option("--num_gpus", type=int, default=1, metavar="N", show_default=True,
        help="The number of GPUs to use.")
def two_extremes(
        target: str,
        model: str,
        panel: str,
        threshold: float,
        nan_th: Optional[float],
        corr_th: Optional[float],
        **exp_kwargs
):
    target = Target(target)
    model = Model(model)
    panel = Panel(panel)

    study_name = get_study_name(target, model, panel, threshold, nan_th, corr_th)
    logger.info(f"Study started: '{study_name}'")

    x, y = get_data(target, model, panel, threshold, nan_th, corr_th)
    logger.info(f"Data loaded x: {x.shape}, y: {y.shape}")

    exp_props = SimpleNamespace(
        study_name=study_name,
        target=target,
        model=model,
        panel=panel,
        threshold=threshold,
        nan_th=nan_th,
        corr_th=corr_th,
        **exp_kwargs
    )
    experiment_results = run_two_extremes_experiment(x, y, exp_props)

    message = f"Completed the study '{study_name}'"
    if len(experiment_results) != exp_props.outer_splits:
        logger.warning(
            message + f", {len(experiment_results)} out of {exp_props.outer_splits} folds")
    else:
        logger.info(message)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{study_name}.json", "w") as f:
        json.dump(
            experiment_results,
            f,
            indent=4,
            default=lambda obj: obj.name if hasattr(obj, "name") else "unknown",
        )


def get_study_name(
        target: Target,
        model: Model,
        panel: Panel,
        threshold: float,
        nan_th: Optional[float] = None,
        corr_th: Optional[float] = None,
):
    study_name = "two_extremes_{}_{}_th{}".format(target.value, model.value, threshold)
    if nan_th:
        study_name += f"_nan{nan_th}"
    if corr_th:
        study_name += f"_corr{corr_th}"
    if panel != Panel.ALL:
        study_name += f"_{panel.value.lower()[:5]}"
    return study_name + f"_s{SEED}"
