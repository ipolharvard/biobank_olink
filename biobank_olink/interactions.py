import numpy as np
import shap
from joblib import Parallel, delayed


def get_shap_interactions(subset, model):
    explainer = shap.TreeExplainer(model)
    return explainer.shap_interaction_values(subset)


def generate_interactions(x, model, n_jobs):
    subsets = np.array_split(x, n_jobs)
    results = Parallel(n_jobs=n_jobs)(
        delayed(get_shap_interactions)(subset, model) for subset in subsets)
    result = np.concatenate(results, axis=0)
    return result
