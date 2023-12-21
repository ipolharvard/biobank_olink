from enum import Enum
from importlib.resources import files

from optuna.trial import TrialState

PROJECT_DATA = files("biobank_olink.data")
PROJECT_ROOT = (PROJECT_DATA / ".." / "..").resolve()

SEED = 42

RESULTS_DIR = PROJECT_ROOT / "results"
OPTUNA_DB_DIR = PROJECT_ROOT / "optuna_dbs"
OPTUNA_STATE_CHECKED = (TrialState.PRUNED, TrialState.COMPLETE)

FEAT_IMP_DIR = PROJECT_DATA / "feat_importances"

MAX_OUTER_SPLITS = 5
MAX_INNER_SPLITS = 5


class Model(Enum):
    XGBOOST = "xgb"
    LOG_REG = "lr"
    TRANSFORMER = "tfr"


class Target(Enum):
    SBP = "sbp"
    DBP = "dbp"
    PP = "pp"


class Panel(Enum):
    ALL = "all"
    CARDIOMETABOLIC = "cardiometabolic"
    INFLAMMATION = "inflammation"
    NEUROLOGY = "neurology"
    ONCOLOGY = "oncology"
    ENDOTHELIAL = "endothelial"
    IMMUNE = "immune"
    INFL2 = "infla2"
    IMMUNE_INFL2 = "immune_infla2"
    BP = "bp"
    RENAL = "renal"
