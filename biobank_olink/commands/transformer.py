from types import SimpleNamespace
from typing import Optional

from click import command, option, Choice

from biobank_olink.base import run_optuna_pipeline
from biobank_olink.constants import Model, Panel
from biobank_olink.dataset import load_olink_and_covariates, get_olink_panel_mapping
from biobank_olink.utils import get_logger, get_study_name

logger = get_logger()


@command
@option("--panel", default=Panel.ALL.value, show_default=True,
        type=Choice([p.value for p in Panel]))
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
def transformer(
        panel: str,
        nan_th: Optional[float],
        corr_th: Optional[float],
        **exp_kwargs
):
    panel = Panel(panel)
    study_name = get_study_name(exp_name="transformer", panel=panel, nan_th=nan_th,
                                corr_th=corr_th)
    logger.info(f"Study started: '{study_name}'")

    ol_df, cov_df = load_olink_and_covariates(nan_th, nan_th, corr_th)
    x = ol_df
    if panel != Panel.ALL:
        x = x.loc[:, x.columns.isin(get_olink_panel_mapping()[panel.value])]
    y = cov_df.sbp > 130
    y = y.values.reshape(-1, 1)

    logger.info(f"Data loaded x: {x.shape}, y: {y.shape} [positives: {y.sum() / len(y):.2%}]")

    exp_props = SimpleNamespace(
        study_name=study_name,
        model=Model.TRANSFORMER,
        panel=panel,
        nan_th=nan_th,
        corr_th=corr_th,
        x_shape=x.shape,
        **exp_kwargs
    )
    run_optuna_pipeline(x, y, exp_props)
