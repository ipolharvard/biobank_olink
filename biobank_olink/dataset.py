import numpy as np
import pandas as pd
from joblib import Memory

from .constants import PROJECT_DATA, PROJECT_ROOT, Panel
from .utils import get_logger

logger = get_logger()
memory = Memory(PROJECT_ROOT / "cache", verbose=2)


@memory.cache
def load_olink_and_covariates(cols_na_th=0, rows_na_th=0, corr_th=None) -> tuple[
    pd.DataFrame, pd.DataFrame]:
    """Load olink data and corresponding covariate features, and preprocess them.
    :param cols_na_th: Threshold for dropping columns with NaNs in olink.
    :param rows_na_th: Threshold for dropping rows with NaNs in olink.
    :param corr_th: Threshold for dropping columns with high correlation in olink.
    :return: ol_df: Olink data (pd.DataFrame), cov_df: covariates (pd.DataFrame)
    """
    ol_df = pd.read_csv(PROJECT_DATA / "olink_data_ext_ins0.csv.gz", index_col="eid")

    if corr_th:
        ol_df_corr = compute_correlations_in_df(ol_df, ol_df.shape)
        mask = np.triu(np.ones(ol_df_corr.shape), k=1).astype(bool)
        high_corr = ol_df_corr.where(mask)
        cols_to_remove = [
            column for column in high_corr.columns if any(high_corr[column] > corr_th)
        ]
        ol_df.drop(columns=cols_to_remove, inplace=True)
        logger.info("Dropped columns due to corr threshold: {}".format(len(cols_to_remove)))

    if cols_na_th:
        col_null_fracs = ol_df.isnull().sum(axis=0) / ol_df.shape[0]
        cols_to_drop = col_null_fracs.loc[col_null_fracs > cols_na_th].index
        ol_df.drop(columns=cols_to_drop, inplace=True)
        logger.info(f"Dropped columns due to NaN threshold: {len(cols_to_drop):,}")

    if rows_na_th:
        row_null_fracs = ol_df.isnull().sum(axis=1) / ol_df.shape[1]
        rows_to_drop = row_null_fracs.loc[row_null_fracs > rows_na_th].index
        ol_df.drop(index=rows_to_drop, inplace=True)
        logger.info(f"Dropped rows due to NaN threshold: {len(rows_to_drop):,}")

    cov_df = pd.read_csv(PROJECT_DATA / "ukb_caucasian_cov.csv.gz", index_col="eid")
    cov_df = cov_df.loc[~cov_df[["sbp", "bmi", "HTNgroup"]].isna().any(axis=1)].copy()
    cov_df["HTNgroup"] = cov_df.HTNgroup.astype(np.int64)
    common_index = cov_df.index.intersection(ol_df.index)
    ol_df, cov_df = ol_df.loc[common_index].copy(), cov_df.loc[common_index].copy()
    return ol_df, cov_df


@memory.cache(ignore=["df"])
def compute_correlations_in_df(df, shape):
    """Shape is used for hashing purposes of the cache."""
    return df.corr()


def get_olink_panel_mapping():
    olink_assays = pd.read_csv(PROJECT_DATA / "olink-explore-3072-assay-list-2023-06-08.csv")
    olink_assays["Explore 384 panel"] = olink_assays.loc[:, "Explore 384 panel"].map(
        lambda x: x.split("_")[0].lower()
    )
    panel_mapping = olink_assays.groupby("Explore 384 panel")["Gene name"].apply(list).to_dict()
    custom_panels = pd.read_csv(PROJECT_DATA / "panels_gene_ontology.csv")
    panel_mapping.update({col: custom_panels[col].dropna().tolist() for col in custom_panels})
    panel_mapping[Panel.IMMUNE_INFL2.value] = panel_mapping[Panel.IMMUNE.value] + panel_mapping[
        Panel.INFL2.value]
    return panel_mapping
