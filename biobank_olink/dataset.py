from importlib.resources import files

import numpy as np
import pandas as pd

DATA_DIR = files("biobank_olink.data")


def load_datasets():
    cov_df = pd.read_csv(DATA_DIR / "BP_cov.zip", sep="\t", index_col="eid")
    ol_df = pd.read_csv(DATA_DIR / "olink_data_ins0.csv", index_col=0)

    eids = cov_df.index.intersection(ol_df.index)

    cov_df = cov_df.loc[eids]
    ol_df = ol_df.loc[cov_df.index]

    cov_df[cov_df == " "] = np.nan
    cov_df[cov_df.HTNgroup == 4] = np.nan

    def replace_comma_with_dot_if_str(x):
        return x.replace(",", ".") if isinstance(x, str) else x

    cols = ["BMI", "SBP", "DBP", "PP"]
    cov_df[cols] = cov_df[cols].applymap(replace_comma_with_dot_if_str)
    cov_df = cov_df.astype(float)

    return ol_df, cov_df
