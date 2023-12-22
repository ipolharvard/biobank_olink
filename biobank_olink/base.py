import json
import time
import traceback
import warnings
from copy import deepcopy
from datetime import datetime
from functools import partial
from multiprocessing import Process, Manager

import numpy as np
import optuna
import pandas as pd
import shap
from optuna._callbacks import MaxTrialsCallback
from optuna.storages import JournalStorage, JournalFileStorage
from optuna.trial import TrialState
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.cuda import OutOfMemoryError
from torchtuples.callbacks import EarlyStopping
from xgboost import XGBClassifier

from .constants import OPTUNA_DB_DIR, OPTUNA_STATE_CHECKED, Model, MAX_OUTER_SPLITS, \
    MAX_INNER_SPLITS, RESULTS_DIR, MODEL_DUMP_DIR
from .constants import SEED
from .transformer import get_transformer, Tokenizer
from .utils import get_gpu_id, get_logger

warnings.filterwarnings("ignore")

logger = get_logger()


def get_model_params(model: Model, trial: optuna.Trial):
    if model == Model.XGBOOST:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 5000),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10, log=True),
        }
    elif model == Model.LOG_REG:
        params = {
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "tol": trial.suggest_float("tol", 1e-4, 1e-1, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", None]),
        }
        if params["penalty"] == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0, 1)
        if params["penalty"] is not None:
            params["C"] = trial.suggest_float("C", 1e-4, 1000, log=True)
        return params
    elif model == Model.TRANSFORMER:
        return {
            "vocab_size": trial.suggest_int("vocab_size", 3, 21),
            "epochs": trial.suggest_int("epochs", 10, 200),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "batch_size": trial.suggest_int("batch_size", 8, 64, 8),
            "n_layer": trial.suggest_int("n_layer", 1, 2),
            "n_embd": trial.suggest_int("n_embd", 8, 128, 8),
            "n_head": trial.suggest_categorical("n_head", [1, 2, 4]),
            "dropout": trial.suggest_float("dropout", 0, 0.5),
            "bias": trial.suggest_categorical("bias", [True, False]),
            "d_ff": trial.suggest_int("d_ff", 8, 128, 8),
        }
    else:
        raise NotImplementedError()


def get_fitted_model(model_name: Model, params, dataset: tuple, gpu_id: int):
    if model_name == Model.XGBOOST:
        model = XGBClassifier(
            tree_method="hist", device=f"cuda:{gpu_id}", random_state=SEED, **params
        )
        model.fit(*dataset)
    elif model_name == Model.LOG_REG:
        model = LogisticRegression(solver="saga", max_iter=10_000, random_state=SEED, **params)
        model.fit(*dataset)
    elif model_name == Model.TRANSFORMER:
        model = get_transformer(**params, device=gpu_id)
        x, y = dataset
        model.fit(input=x, target=y.astype(np.float32), batch_size=params["batch_size"],
                  epochs=params["epochs"], verbose=False, torch_ds_dl=True, shuffle=True,
                  callbacks=[EarlyStopping(patience=5, dataset="train")])
    else:
        raise NotImplementedError()
    return model


def cross_validation_loop(dataset, model_params, args, trial):
    x, y = dataset
    splits = StratifiedKFold(n_splits=MAX_INNER_SPLITS, shuffle=True, random_state=SEED).split(x, y)
    splits = list(splits)[: args.inner_splits]
    intermediate_scores = []
    for i, (train_index, test_index) in enumerate(splits):
        x_train, y_train = x.iloc[train_index], y[train_index]
        x_test, y_test = x.iloc[test_index], y[test_index]

        if args.model == Model.TRANSFORMER:
            tokenizer = Tokenizer(n_bins=model_params["vocab_size"] - 1)
            x_train = tokenizer.fit_transform(x_train).values
            x_test = tokenizer.transform(x_test).values
            model_params["in_feats"] = x_train.shape[1]

        model = get_fitted_model(
            args.model,
            model_params,
            dataset=(x_train, y_train),
            gpu_id=get_gpu_id(args.num_gpus, args.outer_splits)
        )
        y_proba = model.predict_proba(x_test)[:, -1]
        score = roc_auc_score(y_test, y_proba)
        intermediate_scores.append(score)

        trial.report(score, i)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return sum(intermediate_scores) / len(intermediate_scores)


def run_optuna_search(trial: optuna.Trial, dataset, args):
    params = get_model_params(args.model, trial)
    # prevent optuna from running the same trial twice
    for prev_trial in trial.study.trials:
        if prev_trial.state in OPTUNA_STATE_CHECKED and prev_trial.params == trial.params:
            logger.info(f"Params have been evaluated already: {params}")
            raise optuna.TrialPruned()
    try:
        return cross_validation_loop(dataset, params, args, trial)
    except OutOfMemoryError as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Cuda OOM: {params}, {stack_trace}")
        raise optuna.TrialPruned()


def run_optuna_worker(study_name, dataset, storage, callbacks, args):
    pruner = optuna.pruners.MedianPruner(n_startup_trials=25, n_warmup_steps=2)
    if args.model == Model.TRANSFORMER:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)

    _study = optuna.load_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=None),
        pruner=pruner,
    )
    objective = partial(run_optuna_search, dataset=dataset, args=deepcopy(args))
    _study.optimize(func=objective, callbacks=callbacks)


def get_optuna_optimized_params(study_name, dataset, args):
    OPTUNA_DB_DIR.mkdir(exist_ok=True)
    storage = JournalStorage(JournalFileStorage(str(OPTUNA_DB_DIR / f"{study_name}.db")))
    callbacks = [MaxTrialsCallback(args.n_trials, states=OPTUNA_STATE_CHECKED)]
    study = optuna.create_study(
        study_name=study_name, direction="maximize", storage=storage, load_if_exists=True
    )
    if not args.no_opt:
        processes = []
        for i in range(args.optuna_n_workers - 1):
            p = Process(target=run_optuna_worker,
                        args=(study_name, dataset, storage, callbacks, args))
            p.start()
            processes.append(p)
            # prevent workers from accessing the same db file at the beginning
            time.sleep(3)
        run_optuna_worker(study_name, dataset, storage, callbacks, args)
        for p in processes:
            p.join()
    study_stats = {
        "best_trial_no": study.best_trial.number,
        "best_trial_score": study.best_trial.value,
        "num_trials": len(study.get_trials(deepcopy=False, states=OPTUNA_STATE_CHECKED)),
        "num_failed_trials": len(study.get_trials(deepcopy=False, states=[TrialState.FAIL])),
    }
    return study.best_trial.params, study_stats


def one_fold_experiment_run(sh_dict, temp_dataset, test_dataset, fold_num, args):
    study_name = args.study_name + f"_f{fold_num}"
    best_params, study_stats = get_optuna_optimized_params(study_name, temp_dataset, args)
    logger.info(f"Got best params for '{study_name}': {dict(best_params)}")

    x_train, y_train = temp_dataset
    x_test, y_test = test_dataset

    if args.model == Model.TRANSFORMER:
        tokenizer = Tokenizer(n_bins=best_params["vocab_size"] - 1)
        x_train = tokenizer.fit_transform(x_train).values
        x_test = tokenizer.transform(x_test).values
        best_params["in_feats"] = x_train.shape[1]

    model = get_fitted_model(
        args.model,
        best_params,
        dataset=(x_train, y_train),
        gpu_id=get_gpu_id(args.num_gpus, args.outer_splits)
    )
    y_proba = model.predict_proba(x_test)[:, -1]

    summary = {
        "auc_score": roc_auc_score(y_test, y_proba),
        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "fold_num": fold_num,
        **study_stats,
        **args.__dict__,
        **best_params,
        "y_test": y_test.tolist(),
        "y_proba": y_proba.tolist(),
    }
    if args.model == Model.XGBOOST:
        feat_values = model.feature_importances_
        # in addition, we compute SHAP values
        shap_values = shap.TreeExplainer(model)(x_test)
        shap_sum = np.abs(shap_values.values).sum(axis=0).tolist()
        summary["shap_importance"] = dict(zip(x_test.columns, shap_sum))
        summary["feat_importance"] = dict(zip(x_test.columns, feat_values.tolist()))
    elif args.model == Model.LOG_REG:
        feat_values = np.abs(model.coef_[0])
        feat_values = feat_values / np.sum(feat_values)
        summary["feat_importance"] = dict(zip(x_test.columns, feat_values.tolist()))

    sh_dict[fold_num] = summary
    logger.info(f"Fold completed - '{study_name}', auc_score: {summary['auc_score']:.4f}")

    if args.__dict__.get("dump_model") and args.model == Model.XGBOOST:
        MODEL_DUMP_DIR.mkdir(exist_ok=True)
        model.save_model(MODEL_DUMP_DIR / f"{study_name}_model.ubj")
        x_train.to_csv(MODEL_DUMP_DIR / f"{study_name}_x_train.csv.gz", index=False,
                       compression="gzip")
        x_test.to_csv(MODEL_DUMP_DIR / f"{study_name}_x_test.csv.gz", index=False,
                      compression="gzip")
        pd.Series(y_train, name="y_train").to_csv(
            MODEL_DUMP_DIR / f"{study_name}_y_train.csv.gz", index=False, compression="gzip")
        pd.Series(y_test, name="y_test").to_csv(
            MODEL_DUMP_DIR / f"{study_name}_y_test.csv.gz", index=False, compression="gzip")


def run_optuna_pipeline(x, y, exp_props, dump_results: bool = True):
    processes, mgr = [], Manager()
    d = mgr.dict()
    splits = StratifiedKFold(n_splits=MAX_OUTER_SPLITS, shuffle=True, random_state=SEED).split(x, y)
    splits = list(splits)[: exp_props.outer_splits]
    for fold_num, (train_index, test_index) in enumerate(splits):
        temp_dataset = (x.iloc[train_index], y[train_index])
        test_dataset = (x.iloc[test_index], y[test_index])
        p = Process(
            target=one_fold_experiment_run,
            args=(d, temp_dataset, test_dataset, fold_num, exp_props)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    experiment_results = list(d.values())
    mgr.shutdown()

    message = f"Completed the study '{exp_props.study_name}'"
    if len(experiment_results) != exp_props.outer_splits:
        logger.warning(
            message + f", {len(experiment_results)} out of {exp_props.outer_splits} folds")
    else:
        logger.info(message)

    if dump_results:
        RESULTS_DIR.mkdir(exist_ok=True)
        with open(RESULTS_DIR / f"{exp_props.study_name}.json", "w") as f:
            json.dump(
                experiment_results,
                f,
                indent=4,
                default=lambda obj: obj.name if hasattr(obj, "name") else "unknown",
            )
    return experiment_results
