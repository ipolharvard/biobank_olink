from optuna.trial import TrialState

from biobank_olink.constants import PROJECT_ROOT
from biobank_olink.utils import get_color_logger


class ModelType:
    XGBOOST = "xgb"
    LOGISTICREGRESSION = "lr"
    ALL = (XGBOOST, LOGISTICREGRESSION)


class TargetType:
    SBP = "sbp"
    DBP = "dbp"
    PP = "pp"
    PP2 = "pp2"
    ALL = (SBP, DBP, PP, PP2)


class PanelType:
    WHOLE = "all"
    CARDIOMETABOLIC = "cardiometabolic"
    INFLAMMATION = "inflammation"
    NEUROLOGY = "neurology"
    ONCOLOGY = "oncology"
    ALL = (WHOLE, CARDIOMETABOLIC, INFLAMMATION, NEUROLOGY, ONCOLOGY)


RESULTS_DIR = PROJECT_ROOT / "results"
OPTUNA_DB_DIR = PROJECT_ROOT / "optuna_dbs"
OPTUNA_STATE_CHECKED = (TrialState.PRUNED, TrialState.COMPLETE)

logger = get_color_logger("two_extreme_experiment")
