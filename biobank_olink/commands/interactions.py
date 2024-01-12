from multiprocessing import set_start_method
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from click import command, option, argument

from biobank_olink.interactions import generate_interactions
from biobank_olink.utils import get_logger

logger = get_logger()


@command
@argument("dump_path", type=str)
@option("--fold", type=int, default=0, show_default=True, help="Fold number.")
@option("--n_jobs", type=int, default=1, show_default=True, help="Number of parallel jobs.")
def interactions(dump_path: str, fold: int, n_jobs: int):
    """Interactions between Olink and other variables."""
    set_start_method("spawn")

    data_path = Path(dump_path) / f"f{fold}"
    study_name = data_path.parent.name
    logger.info(f"Loading model '{study_name}'")

    model = xgb.Booster()
    model.load_model(data_path / "model.ubj")

    x_train = pd.read_csv(data_path / "x_train.csv.gz")
    x_test = pd.read_csv(data_path / "x_test.csv.gz")
    x = pd.concat([x_train, x_test])

    logger.info(f"Generating interactions for {len(x):,} samples using {n_jobs} workers, "
                f"giving {len(x) // n_jobs:,} per worker.")
    results = generate_interactions(x, model, n_jobs)

    interactions_path = data_path / "interactions.npy"
    logger.info(f"Saving interactions to '{interactions_path}'")
    np.save(interactions_path, results)
