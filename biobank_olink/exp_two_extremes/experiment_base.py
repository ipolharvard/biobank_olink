"""
Contains the base code for the experiment that is invoked by the CLI. Check commands dir for more
info.
"""
import time
import warnings
from copy import deepcopy
from datetime import datetime
from functools import partial
from itertools import combinations
from multiprocessing import Process, current_process, Manager

import numpy as np
import optuna
import pandas as pd
from optuna._callbacks import MaxTrialsCallback
from optuna.storages import JournalStorage, JournalFileStorage
from scipy.spatial.distance import squareform, pdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from biobank_olink.constants import PROJECT_DATA
from biobank_olink.dataset import load_olink_and_covariates, get_olink_panel_mapping
from biobank_olink.exp_two_extremes.constants import (
    ModelType,
    OPTUNA_DB_DIR,
    OPTUNA_STATE_CHECKED,
    PanelType,
    TargetType,
    logger,
)

warnings.filterwarnings("ignore")


def get_model(params, args):
    if args.model == ModelType.XGBOOST:
        name = current_process().name
        values = name.split("-")[-1].split(":")
        if len(values) == 2:
            outer_process, inner_process = values
        else:
            outer_process = values[0]
            inner_process = 0
        proc_num = int(outer_process) * args.outer_splits + int(inner_process)
        gpu_id = proc_num % args.num_gpus

        return XGBClassifier(
            tree_method="gpu_hist", gpu_id=gpu_id, random_state=args.seed, **params
        )
    elif args.model == ModelType.LOGISTICREGRESSION:
        return LogisticRegression(solver="saga", max_iter=10_000, random_state=args.seed, **params)


def cross_validation_loop(dataset, model_params, args, trial):
    skf = StratifiedKFold(n_splits=args.inner_splits, shuffle=True, random_state=args.seed)
    x, y = dataset
    intermediate_scores = []

    for i, (train_index, test_index) in enumerate(skf.split(x, y)):
        train_x, train_y = x.iloc[train_index], y[train_index]
        test_x, test_y = x.iloc[test_index], y[test_index]

        model = get_model(model_params, args)
        model.fit(train_x, train_y)
        y_proba = model.predict_proba(test_x)[:, 1]
        score = roc_auc_score(test_y, y_proba)

        trial.report(score, i)
        if trial.should_prune():
            raise optuna.TrialPruned()

        intermediate_scores.append(score)

    return sum(intermediate_scores) / len(intermediate_scores)


def run_optuna_search(trial: optuna.Trial, dataset, args):
    if args.model == ModelType.XGBOOST:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 3000),
            "max_depth": trial.suggest_int("max_depth", 1, 20),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 10),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.01, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1000),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1000),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "max_bin": trial.suggest_int("max_bin", 2, 1024),
        }
    else:
        params = {
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "tol": trial.suggest_float("tol", 1e-4, 1e-1, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", None]),
        }
        if params["penalty"] == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0, 1)
        if params["penalty"] is not None:
            params["C"] = trial.suggest_float("C", 1e-4, 1000, log=True)
    for prev_trial in trial.study.trials:
        if prev_trial.state in OPTUNA_STATE_CHECKED and prev_trial.params == trial.params:
            raise optuna.TrialPruned()

    return cross_validation_loop(dataset, params, args, trial)


def get_optuna_optimized_params(study_name, dataset, args):
    OPTUNA_DB_DIR.mkdir(exist_ok=True)
    storage = JournalStorage(JournalFileStorage(str(OPTUNA_DB_DIR / f"{study_name}.db")))
    callbacks = [MaxTrialsCallback(args.n_trials, states=OPTUNA_STATE_CHECKED)]

    study = optuna.create_study(
        study_name=study_name, direction="maximize", storage=storage, load_if_exists=True
    )

    def run_optuna_worker():
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=optuna.samplers.TPESampler(seed=None),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=100, n_warmup_steps=2),
        )
        objective = partial(run_optuna_search, dataset=dataset, args=deepcopy(args))
        study.optimize(func=objective, callbacks=callbacks)

    if not args.no_opt:
        processes = []
        for i in range(args.optuna_n_workers - 1):
            p = Process(target=run_optuna_worker)
            p.start()
            processes.append(p)
            # prevent workers from accessing the same db file at the beginning
            time.sleep(3)

        run_optuna_worker()

        for p in processes:
            p.join()

    study_stats = {
        "best_trial_no": study.best_trial.number,
        "num_trials": len(study.get_trials(deepcopy=False, states=OPTUNA_STATE_CHECKED)),
    }
    return study.best_trial.params, study_stats


def one_fold_experiment_run(sh_dict, temp_dataset, test_dataset, fold_num, args):
    study_name = args.study_name + f"_f{fold_num}"

    best_params, study_stats = get_optuna_optimized_params(study_name, temp_dataset, args)
    logger.info(f"Got best params for '{study_name}': {dict(best_params)}")

    x_train, y_train = temp_dataset
    x_test, y_test = test_dataset

    model = get_model(best_params, args)
    model.fit(x_train, y_train)

    y_proba = model.predict_proba(x_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_proba)

    summary = {
        "auc_score": auc_score,
        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "fold_num": fold_num,
        **study_stats,
        **args.__dict__,
        **best_params,
        "y_test": y_test.tolist(),
        "y_proba": y_proba.tolist(),
    }

    if args.model == ModelType.XGBOOST:
        feat_values = model.feature_importances_
    elif args.model == ModelType.LOGISTICREGRESSION:
        feat_values = np.abs(model.coef_[0])
        feat_values = feat_values / np.sum(feat_values)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    summary["feat_importance"] = dict(zip(x_train.columns, feat_values.tolist()))

    sh_dict[fold_num] = summary
    logger.info(f"Fold completed - '{study_name}', auc_score: {summary['auc_score']:.4f}")


def execute_outer_cross_validation_loop(x, y, args):
    skf = StratifiedKFold(n_splits=args.outer_splits, shuffle=True, random_state=args.seed)
    processes, mgr = [], Manager()
    d = mgr.dict()
    for fold_num, (train_index, test_index) in enumerate(skf.split(x, y)):
        temp_dataset = (x.iloc[train_index], y[train_index])
        test_dataset = (x.iloc[test_index], y[test_index])

        p = Process(
            target=one_fold_experiment_run, args=(d, temp_dataset, test_dataset, fold_num, args)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    experiment_results = list(d.values())
    mgr.shutdown()
    return experiment_results


def get_data(args):
    ol_df, cov_df = load_olink_and_covariates(args.nan_th, args.nan_th, args.corr_th)
    # 0 - NTN, 1 - HTN no meds, 2 - HTN meds (we consider only 0 and 1 in the experiment)
    cov_df = cov_df.loc[cov_df.HTNgroup < 2].copy()
    ol_df = ol_df.loc[cov_df.index].copy()

    target = args.target
    lower_bound, upper_bound = cov_df[target].quantile([args.threshold, 1 - args.threshold]).values
    high_cov_df = cov_df[upper_bound <= cov_df[target]]
    low_cov_df = cov_df[cov_df[target] <= lower_bound]

    correction_df = pd.concat([low_cov_df, high_cov_df])
    correction_cols = ["sex", "age", "bmi"]
    correction_df = correction_df[correction_cols]
    correction_df = (correction_df - correction_df.mean()) / correction_df.std()

    similarities = squareform(pdist(correction_df, metric="euclidean"))
    np.fill_diagonal(similarities, np.inf)
    similarities_df = pd.DataFrame(
        similarities, index=correction_df.index, columns=correction_df.index
    )
    del similarities, correction_df
    similarities_sub_df = similarities_df.loc[low_cov_df.index, high_cov_df.index]

    paired_up_df = similarities_sub_df.idxmin().to_frame("p2_id")
    paired_up_df["dist"] = similarities_df.min()
    paired_up_df2 = similarities_sub_df.T.idxmin().to_frame("p2_id")
    paired_up_df2["dist"] = similarities_df.T.min()
    del similarities_df
    paired_up_df = pd.concat([paired_up_df, paired_up_df2])
    paired_up_df.sort_values(by="dist", inplace=True)

    chosen = set()
    for p1_idx, (p2_idx, _) in paired_up_df.iterrows():
        if p1_idx in chosen or p2_idx in chosen:
            continue
        chosen.add(p1_idx)
        chosen.add(p2_idx)

    chosen_cov_df = cov_df.loc[list(chosen)]
    high_cov_df = chosen_cov_df[upper_bound < chosen_cov_df[target]]
    x = ol_df.loc[chosen_cov_df.index]
    y = chosen_cov_df.index.isin(high_cov_df.index)

    if args.panel != PanelType.WHOLE:
        assays_mapping = get_olink_panel_mapping()
        x = x.loc[:, x.columns.isin(assays_mapping[args.panel])]

    if args.n_best_feats:
        standard_x = (x - x.mean()) / x.std()
        is_high = standard_x.index.isin(high_cov_df.index)
        ol_mean_diffs = (standard_x.loc[is_high].mean() - standard_x.loc[~is_high].mean()).abs()
        best_feats = ol_mean_diffs.sort_values(ascending=False)[: args.n_best_feats].index
        x = x.loc[:, best_feats]

    if args.interactions:
        feat_imps = pd.read_csv(PROJECT_DATA / "feat_importances" / f"{args.study_name}.csv")
        most_imp_feats = feat_imps.sort_values("importance", ascending=False)[
            : args.interactions
        ].feature
        inter_combs = list(combinations(most_imp_feats, 2))
        new_feats = pd.concat([x[feat1] * x[feat2] for feat1, feat2 in inter_combs], axis=1)
        new_feats.columns = [f"{feat1}*{feat2}" for feat1, feat2 in inter_combs]
        x = pd.concat([x, new_feats], axis=1)

    if args.model == ModelType.LOGISTICREGRESSION:
        x.fillna(x.median(), inplace=True)
        x = (x - x.mean()) / x.std()

    return x, y
