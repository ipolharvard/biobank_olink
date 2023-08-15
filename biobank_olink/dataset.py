from importlib.resources import files

import numpy as np
import pandas as pd

DATA_DIR = files("biobank_olink.data")


def load_datasets(cols_na_th=0.3, rows_na_th=0.3):
    cov_df = pd.read_csv(DATA_DIR / "BP_cov.zip", sep="\t", index_col="eid")
    ol_df = pd.read_csv(DATA_DIR / "olink_data_ins0.csv", index_col=0)

    def replace_comma_with_dot_if_str(x):
        return x.replace(",", ".") if isinstance(x, str) else x

    cols = ["BMI", "SBP", "DBP", "PP"]
    cov_df[cols] = cov_df[cols].applymap(replace_comma_with_dot_if_str)
    cov_df.replace(" ", np.nan, inplace=True)
    cov_df = cov_df.astype(float)
    cov_df = cov_df.loc[~(cov_df.BMI.isna() | cov_df.SBP.isna())]
    cov_df.loc[cov_df.HTNgroup == 4, "HTNgroup"] = np.nan

    if cols_na_th:
        col_null_fracs = ol_df.isnull().sum().sort_values(ascending=False) / ol_df.shape[0]
        cols_to_drop = col_null_fracs.loc[col_null_fracs > cols_na_th].index.values
        print("dropping {} columns in OLINK".format(len(cols_to_drop)))
        ol_df.drop(columns=cols_to_drop, inplace=True)

    if rows_na_th:
        row_null_fracs = ol_df.isnull().sum(axis=1).sort_values(ascending=False) / ol_df.shape[1]
        rows_to_drop = row_null_fracs.loc[row_null_fracs > rows_na_th].index.values
        print("dropping {} rows in OLINK".format(len(rows_to_drop)))
        ol_df.drop(index=rows_to_drop, inplace=True)

    eids = cov_df.index.intersection(ol_df.index)
    cov_df = cov_df.loc[eids]
    ol_df = ol_df.loc[eids]

    return ol_df, cov_df
