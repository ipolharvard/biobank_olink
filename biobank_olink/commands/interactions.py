from multiprocessing import set_start_method

import numpy as np
import pandas as pd
import xgboost as xgb
from click import command, option

from biobank_olink.constants import PROJECT_DATA, PROJECT_ROOT
from biobank_olink.interactions import generate_interactions
from biobank_olink.utils import get_logger

logger = get_logger()


@command
@option("--fold", type=int, default=0, show_default=True, help="Fold number.")
@option("--n_jobs", type=int, default=1, show_default=True, help="Number of parallel jobs.")
def interactions(fold: int, n_jobs: int):
    """Interactions between Olink and other variables."""
    set_start_method("spawn")
    study_name = "prospective_xgb_corr0.9_y10_lifestyle_ext_olink"

    logger.info(f"Loading model '{study_name}'")

    data_path = PROJECT_DATA / 'model_dump'
    model = xgb.Booster()
    fold_part = f"_f{fold}_"
    x_train = pd.read_csv(data_path / (study_name + fold_part + "x_train.csv.gz"))
    x_train = x_train.values.astype(np.float32)
    model.load_model(data_path / (study_name + fold_part + "model.ubj"))

    logger.info(f"Generating interactions for {len(x_train):,} samples using {n_jobs} workers, "
                f"giving {len(x_train) // n_jobs:,} per worker.")
    results = generate_interactions(x_train, model, n_jobs)

    interactions_path = data_path / (study_name + fold_part + "interactions.npz")
    logger.info(f"Saving interactions to '{interactions_path.relative_to(PROJECT_ROOT)}'")
    np.savez(interactions_path, results)
