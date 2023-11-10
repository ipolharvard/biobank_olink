from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
from click import command, option, Choice

from biobank_olink.base import run_optuna_pipeline
from biobank_olink.constants import Model, Panel, PROJECT_DATA
from biobank_olink.dataset import load_olink_and_covariates, get_olink_panel_mapping
from biobank_olink.utils import get_logger, get_study_name

logger = get_logger()


@command
@option("--model", type=Choice([m.value for m in Model]))
@option("--panel", default=Panel.ALL.value, show_default=True,
        type=Choice([p.value for p in Panel]))
@option("--years", type=int, default=5, show_default=True)
@option("--lifestyle", is_flag=True, default=False)
@option("--ext", is_flag=True, default=False)
@option("--olink", is_flag=True, default=False)
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
        years: int,
        lifestyle: bool,
        ext: bool,
        olink: bool,
        **exp_kwargs
):
    model = Model(model)
    panel = Panel(panel)
    study_name = get_study_name(exp_name="pred_diagnosis", model=model, panel=panel, nan_th=nan_th,
                                corr_th=corr_th)

    ol_df, cov_df = load_olink_and_covariates(nan_th, nan_th, corr_th)
    # take those patients who had no HTN at the baseline
    cov_df = cov_df.loc[cov_df.HTNgroup == 0].copy()
    # remove those who had I10 before the baseline
    cov_df = cov_df.loc[cov_df.timeHTNcoxfinal >= 0].copy()
    # positive cases are patients who developed HTN within n years
    y = (cov_df.HTNcaseCoxfinal == 1) & (cov_df.timeHTNcoxfinal < 365.25 * years)
    y = y.values.reshape(-1, 1)
    study_name += f"_y{years}"

    ol_df = ol_df.loc[cov_df.index].copy()
    if panel != Panel.ALL:
        ol_df = ol_df.loc[:, ol_df.columns.isin(get_olink_panel_mapping()[panel.value])]

    x = cov_df[["age", "sex"]]
    if lifestyle:
        cov_cols_num = ["bmi", "fastingtime", "deprivation", "educationuni"]
        cov_cols_cat = ["smoking", "alcohol", "physactivity"]
        one_hot_chunks = []
        for cat_col in cov_cols_cat:
            one_hot_col = pd.get_dummies(cov_df[cat_col], columns=cat_col).astype(float)
            one_hot_col.loc[cov_df[cat_col].isna()] = np.nan
            one_hot_col.columns = [f"{cat_col}_{col}" for col in one_hot_col.columns]
            one_hot_chunks.append(one_hot_col)
        x = pd.concat([x, cov_df[cov_cols_num], *one_hot_chunks], axis=1, verify_integrity=True)
        study_name += "_lifestyle"

    if ext:
        ext_df = pd.read_csv(PROJECT_DATA / "biomarker_clinical.csv.gz", index_col="eid")
        x = pd.concat([x, ext_df], axis=1, verify_integrity=True)

    if olink:
        x = pd.concat([x, ol_df], axis=1, verify_integrity=True)
        study_name += "_olink"

    if model == Model.LOG_REG:
        x.fillna(x.median(), inplace=True)
        x = (x - x.mean()) / x.std()

    logger.info(f"Study started: '{study_name}'")
    logger.info(f"Data loaded x: {x.shape}, y: {y.shape} [positives: {y.sum() / len(y):.2%}]")

    exp_props = SimpleNamespace(
        study_name=study_name,
        model=model,
        panel=panel,
        nan_th=nan_th,
        corr_th=corr_th,
        years=years,
        lifestyle=lifestyle,
        ext=ext,
        olink=olink,
        x_shape=x.shape,
        **exp_kwargs
    )
    run_optuna_pipeline(x, y, exp_props)
