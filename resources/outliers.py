import pandas as pd
import numpy as np

def outliers_mad(file_df, col_name, count_from=0):
    file_df_col = file_df[col_name].dropna()

    if count_from is not None:
        file_df_col = file_df_col[file_df_col > count_from]

    # MAD
    outliers_mad = mad_based_outlier(file_df_col)
    list_outlier = []
    for ax, func in zip(file_df_col, outliers_mad):
        if func:  # True is outlier
            list_outlier.append(ax)
    list_outlier = set(list_outlier)
    name = str(col_name) + '_mad_outlier'
    file_df[name] = pd.Series(0, index=file_df.index)
    file_df[name] = file_df.apply(
        lambda x: 1
        if x[col_name] in list_outlier
        else 0, axis=1)

    return file_df


def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:, None]

    median = np.median(points, axis=0)

    diff = np.sum((points - median) ** 2, axis=-1)

    diff = np.sqrt(diff)

    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
