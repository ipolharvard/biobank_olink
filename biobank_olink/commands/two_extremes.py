from types import SimpleNamespace
from typing import Optional

from click import command, option, Choice

from biobank_olink.base import run_optuna_pipeline
from biobank_olink.constants import Target, Model, Panel
from biobank_olink.exp_two_extremes import get_data
from biobank_olink.utils import get_logger, get_study_name

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
@option("--no_ren", is_flag=True, default=False)
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
        no_ren: bool,
        **exp_kwargs
):
    target = Target(target)
    model = Model(model)
    panel = Panel(panel)
    study_name = get_study_name(exp_name="two_extremes", model=model, panel=panel, nan_th=nan_th,
                                corr_th=corr_th)
    study_name += f"_{target.value}"
    if threshold:
        study_name += f"_th{threshold}"
    if no_ren:
        study_name += "_noren"
    logger.info(f"Study started: '{study_name}'")

    x, y = get_data(target, threshold, model, panel, nan_th, corr_th)
    if no_ren:
        x.drop(columns=["REN"], inplace=True)
    logger.info(f"Data loaded x: {x.shape}, y: {y.shape} [positives: {y.sum() / len(y):.2%}]")

    exp_props = SimpleNamespace(
        study_name=study_name,
        target=target,
        model=model,
        panel=panel,
        threshold=threshold,
        nan_th=nan_th,
        corr_th=corr_th,
        no_ren=no_ren,
        x_shape=x.shape,
        **exp_kwargs
    )
    run_optuna_pipeline(x, y, exp_props)
