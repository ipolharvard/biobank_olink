import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist

from ..constants import Target, Model, Panel
from ..dataset import load_olink_and_covariates, get_olink_panel_mapping


def get_data(
        target: Target,
        model: Model,
        panel: Panel,
        threshold: float,
        nan_th: float = 0,
        corr_th: float = 0,
):
    ol_df, cov_df = load_olink_and_covariates(nan_th, nan_th, corr_th)
    # 0 - NTN, 1 - HTN no meds, 2 - HTN meds (we consider only 0 and 1 in the experiment)
    cov_df = cov_df.loc[cov_df.HTNgroup < 2].copy()
    ol_df = ol_df.loc[cov_df.index].copy()

    target = target.value
    lower_bound, upper_bound = cov_df[target].quantile([threshold, 1 - threshold]).values
    high_cov_df = cov_df[upper_bound <= cov_df[target]]
    low_cov_df = cov_df[cov_df[target] < lower_bound]

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
    y = chosen_cov_df.index.isin(high_cov_df.index).reshape(-1, 1)

    if panel != Panel.ALL:
        x = x.loc[:, x.columns.isin(get_olink_panel_mapping()[panel.value])]

    if model == Model.LOG_REG:
        x.fillna(x.median(), inplace=True)
        x = (x - x.mean()) / x.std()

    return x, y
