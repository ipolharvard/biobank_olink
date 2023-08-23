import numpy as np
import pandas as pd
from joblib import Memory

from biobank_olink.constants import DATA_DIR

memory = Memory(DATA_DIR / "../../cache", verbose=2)


@memory.cache
def load_olink_and_covariates(cols_na_th=0.3, rows_na_th=0.3, corr_th=None):
    """
    Load Olink data and covariates, and preprocess them.
    :param cols_na_th: threshold for dropping columns with NaNs in OLINK
    :param rows_na_th: threshold for dropping rows with NaNs in OLINK
    :param corr_th: threshold for dropping columns with high correlation in OLINK
    :return: ol_df: Olink data (pd.DataFrame), cov_df: covariates (pd.DataFrame)
    """
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
        print("Dropped columns due to NaN threshold: {}".format(len(cols_to_drop)))
        ol_df.drop(columns=cols_to_drop, inplace=True)

    if rows_na_th:
        row_null_fracs = ol_df.isnull().sum(axis=1).sort_values(ascending=False) / ol_df.shape[1]
        rows_to_drop = row_null_fracs.loc[row_null_fracs > rows_na_th].index.values
        print("Dropped rows due to NaN threshold: {}".format(len(rows_to_drop)))
        ol_df.drop(index=rows_to_drop, inplace=True)

    if corr_th is not None:
        ol_df_corr = ol_df.corr()
        mask = np.triu(np.ones(ol_df_corr.shape), k=1).astype(bool)
        high_corr = ol_df_corr.where(mask)
        cols_to_remove = [column for column in high_corr.columns if
                          any(high_corr[column] > corr_th)]
        ol_df.drop(columns=cols_to_remove, inplace=True)
        print("Dropped columns due to Corr threshold: {}".format(len(cols_to_remove)))

    common_index = cov_df.index.intersection(ol_df.index)
    cov_df = cov_df.loc[common_index]
    ol_df = ol_df.loc[common_index]

    return ol_df, cov_df


def get_olink_panel_mapping():
    olink_assays = pd.read_csv(DATA_DIR / "olink-explore-3072-assay-list-2023-06-08.csv")
    olink_assays["Explore 384 panel"] = olink_assays.loc[:, "Explore 384 panel"].apply(
        lambda x: x.split("_")[0])
    assays_mapping = olink_assays.groupby("Explore 384 panel")["Gene name"].apply(list).to_dict()
    return assays_mapping
