#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

import csv
from contextlib import nullcontext

def save_df(df, path, head2save=None, printer=None) :
    """
    Save a dataframe with quoted SMLM headers.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to save.
    path : str or pathlib.Path
        CSV output path.
    head2save : list[str] or None, default=None
        Optional logical headers to include.
    printer : object or None, default=None
        Optional object exposing a ``timeit`` context manager.
    """
    timeit = nullcontext() if printer is None else printer.timeit(f"saving {df.__class__.__name__} into {path}")

    with timeit :
        columns = list(df.columns)
        for pos,col in enumerate(columns) :
            columns[pos] = '"' + col + '"'
        df.columns = columns

        if head2save is not None :
            columns = [col for col in df.columns if col[1:-1] in head2save]
        df.to_csv(path, columns=columns, quoting = csv.QUOTE_NONE, float_format="%.3f")

        columns = list(df.columns)
        for pos,col in enumerate(columns) :
            columns[pos] = col[1:-1]
        df.columns = columns
