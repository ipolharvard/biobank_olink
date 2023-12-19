import json
from multiprocessing import set_start_method
from typing import Optional

import numpy as np
import pandas as pd
import scipy
from click import command, option, Choice
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from biobank_olink.constants import Model, Panel, RESULTS_DIR
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
    default=1,
    show_default=True,
    metavar="N",
    help="The number of inner splits in each outer split, the average score is used as Optuna "
         "objective (>1)."
)
@option("--n_trials", type=int, default=1, metavar="N", show_default=True,
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
def cross_on_prospective(
        model: str,
        panel: str,
        nan_th: Optional[float],
        corr_th: Optional[float],
        **exp_kwargs
):
    set_start_method("spawn")
    model = Model(model)
    panel = Panel(panel)
    study_name = get_study_name(exp_name="cross_on_prospective", model=model, panel=panel,
                                nan_th=nan_th, corr_th=corr_th)

    ol_df, cov_df = load_olink_and_covariates(nan_th, nan_th, corr_th)
    if panel != Panel.ALL:
        ol_df = ol_df.loc[:, ol_df.columns.isin(get_olink_panel_mapping()[panel.value])].copy()

    htn_cov_df = cov_df.loc[cov_df.HTNgroup == 1].copy()
    ntn_cov_df = cov_df.loc[cov_df.HTNgroup == 0].copy()
    ntn_cov_df = ntn_cov_df.loc[ntn_cov_df.timeHTNcoxfinal >= 0].copy()
    ntn_y = ntn_cov_df.HTNcaseCoxfinal == 1

    logger.info(f"Study started: '{study_name}'")
    results = []
    for i, seed in enumerate(range(0, 100), 1):
        ntn_cov_df1, ntn_cov_df2 = train_test_split(ntn_cov_df, test_size=0.5, stratify=ntn_y,
                                                    random_state=seed)

        correction_df = pd.concat([htn_cov_df, ntn_cov_df1])[["sex", "age", "bmi"]]
        # standardize selected data
        correction_df = (correction_df - correction_df.mean()) / correction_df.std()
        # use Euclidean distance
        similarities = squareform(pdist(correction_df, metric="euclidean"))
        np.fill_diagonal(similarities, np.inf)
        similarities_df = pd.DataFrame(
            similarities, index=correction_df.index, columns=correction_df.index
        )
        similarities_sub_df = similarities_df.loc[htn_cov_df.index, ntn_cov_df1.index]
        # find the most similar pairs
        paired_up_df = similarities_sub_df.idxmin().to_frame("p2_id")
        paired_up_df["dist"] = similarities_df.min()
        paired_up_df2 = similarities_sub_df.T.idxmin().to_frame("p2_id")
        paired_up_df2["dist"] = similarities_df.T.min()
        paired_up_df = pd.concat([paired_up_df, paired_up_df2]);
        paired_up_df.sort_values(by="dist", inplace=True)

        chosen = set()
        for p1_idx, (p2_idx, _) in paired_up_df.iterrows():
            if p1_idx in chosen or p2_idx in chosen:
                continue
            chosen.add(p1_idx)
            chosen.add(p2_idx)
        chosen_cov_df = cov_df.loc[list(chosen)]

        x = ol_df.loc[chosen_cov_df.index].copy()
        y = chosen_cov_df.index.isin(htn_cov_df.index).reshape(-1, 1)

        if model == Model.LOG_REG:
            x.fillna(x.median(), inplace=True)
            x = (x - x.mean()) / x.std()

        logger.info(f"[{i}/{100}] x: {x.shape}, y: {y.shape} [positives: {y.sum() / len(y):.2%}]")
        # exp_props = SimpleNamespace(study_name=study_name, model=model, **exp_kwargs)
        # logger.info(f"Running optuna pipeline for {study_name}")
        # exp_results = run_optuna_pipeline(x, y, exp_props, dump_results=False)
        # best_params = max(exp_results, key=lambda x: x["auc_score"])
        #
        # logger.info("Removing optuna dbs...")
        # for optuna_db_path in OPTUNA_DB_DIR.glob(f"{study_name}_*"):
        #     optuna_db_path.unlink()

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y,
                                                            random_state=seed)
        logger.info(f"Training final model with the best params...")
        estimator = XGBClassifier(tree_method="hist", random_state=42)
        estimator.fit(x_train, y_train)

        y_pred = estimator.predict_proba(x_test)[:, 1]
        val_auc_score = roc_auc_score(y_test, y_pred)
        result = {"val_auc_score": val_auc_score}

        x = ol_df.loc[ntn_cov_df2.index]
        y_pred = estimator.predict_proba(x)[:, 1]

        y = ntn_cov_df2.timeHTNcoxfinal.to_frame()
        y.rename(columns={"timeHTNcoxfinal": "duration"}, inplace=True)
        y["event"] = (ntn_cov_df2.HTNcaseCoxfinal == 1).astype(int)
        y.loc[~y.event.astype(bool), "duration"] = y.duration.max()
        y["group"] = pd.qcut(y_pred, q=2, labels=False)

        for g, num_event in y.groupby("group").event.sum().items():
            result[f"pred_group_{g}"] = num_event

        target_cols = ["age", "bmi", "sbp", "dbp", "sex"]
        target_props = ntn_cov_df2.groupby(y["group"])[target_cols].mean()
        df = ntn_cov_df2[target_cols]
        g1_index, g2_index = train_test_split(df.index.values, train_size=0.5, random_state=seed)
        n_iters = 5_000
        p = 0.9

        def swap_indices(idx1, idx2):
            temp = g2_index[idx2]
            g2_index[idx2] = g1_index[idx1]
            g1_index[idx1] = temp

        def calc_score(current_props, target_props):
            return scipy.spatial.distance.euclidean(current_props, target_props)

        logger.info("Coming up with a random group assignment...")
        score1 = calc_score(df.loc[g1_index].mean(), target_props.loc[0])
        score2 = calc_score(df.loc[g2_index].mean(), target_props.loc[1])

        for _ in range(n_iters):
            idx1 = np.random.randint(low=0, high=len(g1_index))
            idx2 = np.random.randint(low=0, high=len(g2_index))
            swap_indices(idx1, idx2)

            new_score1 = calc_score(df.loc[g1_index].mean(), target_props.loc[0])
            new_score2 = calc_score(df.loc[g2_index].mean(), target_props.loc[1])

            if (new_score1 > score1 or new_score2 > score2) and np.random.rand() < p:
                swap_indices(idx1, idx2)
            else:
                score1 = new_score1
                score2 = new_score2

        y["random_group"] = ntn_cov_df2.index.isin(g2_index).astype(int)
        for g, num_event in y.groupby("random_group").event.sum().items():
            result[f"random_group_{g}"] = num_event

        results.append(result)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{study_name}.json", "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Experiment finished, results dump to {RESULTS_DIR / f'{study_name}.json'}")
