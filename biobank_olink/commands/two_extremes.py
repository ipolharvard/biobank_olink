import json
from types import SimpleNamespace

from click import command, option, Choice

from biobank_olink.exp_two_extremes.constants import (
    TargetType,
    PanelType,
    RESULTS_DIR,
    ModelType,
    logger,
)
from biobank_olink.exp_two_extremes.experiment_base import (
    execute_outer_cross_validation_loop,
    get_data,
)


@command
@option("--target", type=Choice(TargetType.ALL))
@option("--model", default=ModelType.XGBOOST, type=Choice(ModelType.ALL))
@option("--panel", default=PanelType.WHOLE, type=Choice(PanelType.ALL))
@option(
    "--threshold",
    type=float,
    default=0.35,
    help="Upper and lower percentiles to create the two extremes of the target variable.",
)
@option(
    "--nan_th",
    type=float,
    default=0.3,
    help="threshold for maximal NaN ratio, everything above is removed",
)
@option(
    "--corr_th",
    type=float,
    default=None,
    help="threshold for maximal correlation, columns that correlate stronger"
         " are removed,'None' means do not remove anything",
)
@option("--interactions", type=int, default=None)
@option("--n_best_feats", type=int, default=None)
@option(
    "--outer_splits",
    type=int,
    default=2,
    help="number of outer splits of dataset (>1)",
)
@option(
    "--inner_splits",
    type=int,
    default=2,
    help="number of inner splits in each outer split, the average score is Optuna's objective (>1)"
)
@option("--n_trials", type=int, default=5, metavar="N", help="number of trials for Optuna")
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
    help="number of workers for Optuna",
)
@option("--num_gpus", type=int, default=1, help="Number of GPUs to use.")
@option("--seed", type=int, default=42, help="Seed to use for RNG.")
def two_extremes(**kwargs):
    args = SimpleNamespace(**kwargs)
    args.study_name = "two_extremes_{}_{}_th{}".format(args.target, args.model, args.threshold)
    if args.nan_th:
        args.study_name += f"_nan{args.nan_th}"
    if args.corr_th:
        args.study_name += f"_corr{args.corr_th}"
    if args.panel != PanelType.WHOLE:
        args.study_name += f"_{args.panel.lower()[:5]}"
    if args.n_best_feats:
        args.study_name += f"_best{args.n_best_feats}"
    args.study_name += f"_s{args.seed}"
    study_name = (
        args.study_name + f"_inter{args.interactions}" if args.interactions else args.study_name
    )
    logger.info(f"Study started: '{study_name}'")

    x, y = get_data(args)
    logger.info(f"Data loaded x: {x.shape}, y: {y.shape}")
    args.study_name = study_name
    experiment_results = execute_outer_cross_validation_loop(x, y, args)

    message = f"Completed the study '{args.study_name}'"
    if len(experiment_results) != args.outer_splits:
        logger.warning(message + f", {len(experiment_results)} out of {args.outer_splits} folds")
    else:
        logger.info(message)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{args.study_name}.json", "w") as f:
        json.dump(
            experiment_results,
            f,
            indent=4,
            default=lambda obj: obj.__name__ if hasattr(obj, "__name__") else "unknown",
        )
