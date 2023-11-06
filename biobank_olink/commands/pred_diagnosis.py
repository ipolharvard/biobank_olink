from types import SimpleNamespace
from typing import Optional

import pandas as pd
from click import command, option, Choice

from biobank_olink.base import run_optuna_pipeline
from biobank_olink.constants import Model, Panel
from biobank_olink.dataset import load_olink_and_covariates, get_olink_panel_mapping
from biobank_olink.utils import get_logger, get_study_name

logger = get_logger()


@command
@option("--model", type=Choice([m.value for m in Model]))
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
def pred_diagnosis(
        model: str,
        panel: str,
        nan_th: Optional[float],
        corr_th: Optional[float],
        **exp_kwargs
):
    model = Model(model)
    panel = Panel(panel)
    study_name = get_study_name(exp_name="pred_diagnosis", model=model, panel=panel, nan_th=nan_th,
                                corr_th=corr_th)

    ol_df, cov_df = load_olink_and_covariates(nan_th, nan_th, corr_th)
    # remove those who take HTN medication at baseline
    cov_df = cov_df.loc[cov_df.HTNgroup < 2].copy()
    # remove those who had I20 before baseline
    cov_df = cov_df.loc[cov_df.timeHTNcoxfinal >= 0].copy()

    x = ol_df.loc[cov_df.index].copy()
    # positive cases are patients who developed HTN within 5 years
    y = ((cov_df.HTNcaseCoxfinal == 1) & (cov_df.timeHTNcoxfinal < 365.25 * 5)).values.reshape(-1,
                                                                                               1)
    if panel != Panel.ALL:
        assays_mapping = get_olink_panel_mapping()
        x = x.loc[:, x.columns.isin(assays_mapping[panel])]

    cov_cols = ["age", "sex", "bmi", "sbp", "dbp", "pp"]
    x = pd.concat([cov_df[cov_cols], x], axis=1, verify_integrity=True)

    if model == Model.LOG_REG:
        x.fillna(x.median(), inplace=True)
        x = (x - x.mean()) / x.std()

    logger.info(f"Data loaded x: {x.shape}, y: {y.shape}")

    exp_props = SimpleNamespace(
        study_name=study_name,
        model=model,
        panel=panel,
        nan_th=nan_th,
        corr_th=corr_th,
        **exp_kwargs
    )
    run_optuna_pipeline(x, y, exp_props)
