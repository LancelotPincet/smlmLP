#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

"""
Save a DataFrame with quoted SMLM headers.

This function exports DataFrame data to CSV format with proper header quoting
for SMLM data files.
"""

import csv
from contextlib import nullcontext


# %% Main function


def save_df(df, path, head2save=None, printer=None):
    """
    Save a DataFrame with quoted SMLM headers.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to save.
    path : str or pathlib.Path
        CSV output path.
    head2save : list or None, optional
        Optional list of logical headers to include.
    printer : object or None, optional
        Optional object exposing a ``timeit`` context manager.
    """
    timeit = (
        nullcontext()
        if printer is None
        else printer.timeit(f"saving {df.__class__.__name__} into {path}")
    )

    with timeit:
        columns_list = list(df.columns)
        for pos, col in enumerate(columns_list):
            columns_list[pos] = '"' + col + '"'
        df.columns = columns_list

        if head2save is not None:
            columns_list = [
                col for col in df.columns if col[1:-1] in head2save
            ]
        df.to_csv(
            path, columns=columns_list, quoting=csv.QUOTE_NONE, float_format="%.3f"
        )

        columns_list = list(df.columns)
        for pos, col in enumerate(columns_list):
            columns_list[pos] = col[1:-1]
        df.columns = columns_list