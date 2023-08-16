"""
The whole experiment is defined in this file. It is a bit long, but it is not that complicated.
"""
import argparse
import json
import random
import time
import warnings
from copy import deepcopy
from datetime import datetime
from functools import partial
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna._callbacks import MaxTrialsCallback
from optuna.exceptions import ExperimentalWarning
from optuna.storages import JournalStorage, JournalFileStorage
from optuna.trial import TrialState
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

from biobank_olink.dataset import load_datasets
from biobank_olink.utils import get_color_logger

warnings.filterwarnings("ignore", category=ExperimentalWarning)

THREADS_PER_WORKER = 2

RESULTS_DIR = "results"
OPTUNA_DB_DIR = "optuna_dbs"

OPTUNA_STATE_CHECKED = (TrialState.PRUNED, TrialState.COMPLETE)

logger = get_color_logger()


class Target:
    SBP = "sbp"
    DBP = "dbp"
    ALL = (SBP, DBP)


class NanHandling:
    REMOVE = "remove"
    IGNORE = "ignore"
    ALL = (REMOVE, IGNORE)


class CorrHandling:
    IGNORE = "ignore"
    DROP = "drop"
    ALL = (IGNORE, DROP)


def get_model(params, args):
    gpu_id = random.randint(0, 1)
    return XGBClassifier(tree_method="gpu_hist", gpu_id=gpu_id, random_state=args.seed,
                         **params)


def cross_validation_loop(dataset, model_params, args, trial):
    kf = KFold(n_splits=args.inner_splits, shuffle=True, random_state=args.seed)
    x, y = dataset
    intermediate_scores = []

    for i, (train_index, test_index) in enumerate(kf.split(x)):
        train_x, train_y = x.iloc[train_index], y[train_index]
        test_x, test_y = x.iloc[test_index], y[test_index]

        model = get_model(model_params, args)
        model.fit(train_x, train_y)
        y_proba = model.predict_proba(test_x)[:, 1]
        auc_score = roc_auc_score(test_y, y_proba)

        trial.report(auc_score, i)
        if trial.should_prune():
            raise optuna.TrialPruned()

        intermediate_scores.append(auc_score)

    return sum(intermediate_scores) / len(intermediate_scores)


def run_optuna_search(trial: optuna.Trial, dataset, args):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 10, 2000),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.01, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
    }
    for prev_trial in trial.study.trials:
        if prev_trial.state in OPTUNA_STATE_CHECKED and prev_trial.params == trial.params:
            raise optuna.TrialPruned()

    return cross_validation_loop(dataset, params, args, trial)


def get_optuna_optimized_params(study_name, dataset, args):
    optuna_dbs_dir = args.working_dir / OPTUNA_DB_DIR
    optuna_dbs_dir.mkdir(exist_ok=True)
    storage = JournalStorage(JournalFileStorage(str(optuna_dbs_dir / f"{study_name}.db")))
    callbacks = [MaxTrialsCallback(args.n_trials, states=OPTUNA_STATE_CHECKED)]

    study = optuna.create_study(study_name=study_name, direction="maximize", storage=storage,
                                load_if_exists=True)

    def run_optuna_worker():
        study = optuna.load_study(study_name=study_name, storage=storage,
                                  sampler=optuna.samplers.TPESampler(seed=None),
                                  pruner=optuna.pruners.MedianPruner(n_startup_trials=20,
                                                                     n_warmup_steps=3))
        objective = partial(run_optuna_search, dataset=dataset, args=deepcopy(args))
        study.optimize(func=objective, callbacks=callbacks)

    processes = []
    for i in range(args.optuna_n_workers - 1):
        p = Process(target=run_optuna_worker)
        p.start()
        processes.append(p)
        time.sleep(1)

    run_optuna_worker()

    for p in processes:
        p.join()

    return study.best_trial.params


def one_fold_experiment_run(queue, temp_dataset, test_dataset, fold_num, args):
    study_name = args.study_name + f"_f{fold_num}"
    start_time = time.time()

    best_params = get_optuna_optimized_params(study_name, temp_dataset, args)
    logger.info(f"Got best params for '{study_name}': {dict(best_params)}")

    train_x, train_y = temp_dataset
    test_x, test_y = test_dataset

    model = get_model(best_params, args)
    model.fit(train_x, train_y)
    y_proba = model.predict_proba(test_x)[:, 1]
    auc_score = roc_auc_score(test_y, y_proba)

    summary = {
        "auc_score": auc_score,
        **best_params,
        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "elapsed_time": time.time() - start_time,
        "fold_num": fold_num,
        **args.__dict__,
        "feat_importance": pd.Series(model.feature_importances_, index=train_x.columns).nlargest(
            50).to_dict()
    }

    queue.put(summary)
    logger.info(f"Fold completed '{study_name}', auc_score: {summary['auc_score']:.4f}")


def execute_outer_cross_validation_loop(x, y, args):
    kf = KFold(n_splits=args.outer_splits, shuffle=True, random_state=args.seed)
    processes, queue = [], Queue()
    for fold_num, (train_index, test_index) in enumerate(kf.split(x)):
        temp_dataset = (x.iloc[train_index], y[train_index])
        test_dataset = (x.iloc[test_index], y[test_index])

        p = Process(target=one_fold_experiment_run,
                    args=(queue, temp_dataset, test_dataset, fold_num, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    experiment_results = []
    while not queue.empty():
        experiment_results.append(queue.get())
    return experiment_results


def get_data(args):
    if args.nan_handling == NanHandling.REMOVE:
        ol_df, cov_df = load_datasets(cols_na_th=0.3, rows_na_th=0.3)
    elif args.nan_handling == NanHandling.IGNORE:
        ol_df, cov_df = load_datasets(cols_na_th=0, rows_na_th=0)

    cov_df = cov_df.loc[cov_df.HTNgroup < 2]
    ol_df = ol_df.loc[cov_df.index]

    if args.corr_handling == CorrHandling.DROP:
        ol_df_corr = ol_df.corr()
        mask = np.triu(np.ones(ol_df_corr.shape), k=1).astype(bool)
        high_corr = ol_df_corr.where(mask)
        cols_to_remove = [column for column in high_corr.columns if any(high_corr[column] > 0.9)]
        ol_df.drop(columns=cols_to_remove, inplace=True)

    if args.target == Target.SBP:
        target = "SBP"
    else:
        target = "DBP"

    lower_bound, upper_bound = cov_df[target].quantile([args.threshold, 1 - args.threshold]).values
    low_cov_df = cov_df[cov_df[target] < lower_bound]
    high_cov_df = cov_df[upper_bound < cov_df[target]]

    correction_df = pd.concat([low_cov_df, high_cov_df])
    correction_cols = ["Sex", "age", "BMI"]
    correction_df = correction_df[correction_cols]
    correction_df = (correction_df - correction_df.mean()) / correction_df.std()

    similarities = squareform(pdist(correction_df, metric="euclidean"))
    np.fill_diagonal(similarities, np.inf)
    similarities_df = pd.DataFrame(similarities, index=correction_df.index,
                                   columns=correction_df.index)
    del similarities
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
    return x, y


def run_experiment(args):
    args.study_name = "two_extremes_exp_{}_th{}_nan_{}_corr_{}_s{}".format(
        args.target, args.threshold, args.nan_handling, args.corr_handling, args.seed
    )
    logger.info(f"Study started: '{args.study_name}'")
    args.working_dir = Path(args.working_dir)

    x, y = get_data(args)
    logger.info(f"Data loaded, x: {x.shape}, y: {y.shape}")

    experiment_results = execute_outer_cross_validation_loop(x, y, args)

    message = f"Completed the study '{args.study_name}'"
    if len(experiment_results) != args.outer_splits:
        logger.warning(message + f", {len(experiment_results)} out of {args.outer_splits} folds")
    else:
        logger.info(message)

    result_dir = args.working_dir / RESULTS_DIR
    result_dir.mkdir(exist_ok=True)
    with open(result_dir / f"{args.study_name}.json", "w") as f:
        json.dump(experiment_results, f, indent=4,
                  default=lambda obj: obj.__name__ if hasattr(obj, "__name__") else "unknown")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.35, metavar='N')
    parser.add_argument('--target', type=str, choices=Target.ALL, default=Target.ALL[0])
    parser.add_argument('--nan_handling', type=str, choices=NanHandling.ALL,
                        default=NanHandling.ALL[0])
    parser.add_argument('--corr_handling', type=str, choices=CorrHandling.ALL,
                        default=CorrHandling.ALL[0])
    parser.add_argument('--outer_splits', type=int, default=1, metavar='N',
                        help="number of outer splits of dataset")
    parser.add_argument('--inner_splits', type=int, default=2, metavar='N',
                        help="number of inner splits in each outer split, and also this is the "
                             "score Optuna bases on")
    parser.add_argument('--n_trials', type=int, default=5, metavar='N',
                        help="number of trials for Optuna")
    parser.add_argument('--optuna_n_workers', type=int, default=1, metavar='N',
                        help="number of workers for Optuna")
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed')
    parser.add_argument('--working_dir', type=str, default=".", metavar='N')

    args = parser.parse_args()
    run_experiment(args)
