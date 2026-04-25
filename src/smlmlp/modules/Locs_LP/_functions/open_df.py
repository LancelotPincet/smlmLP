#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

from contextlib import nullcontext

from warnings import warn

from smlmlp import columns

def open_df(locs, df, printer=None) :
    """
    Open one pandas dataframe into a Locs object.

    Parameters
    ----------
    locs : Locs
        Localization container to receive dataframe values.
    df : pandas.DataFrame
        Source dataframe. Its index name selects the destination dataframe.
    printer : object or None, default=None
        Optional object exposing a ``timeit`` context manager.

    Returns
    -------
    None
        Data are assigned to ``locs`` in place.
    """

    index = df.index.name
    if index is not None :
        index = index.replace('"', '')
        index = index.replace("'", "")
    else :
        index = 'detection'
    if index not in columns.headers :
        warn(f'Skipping opening unknown dataframe with index "{index}"')
        return None
    col_index = columns.headers[index]
    df_name = col_index.df_name
    dataframe = getattr(locs, df_name)

    timeit = nullcontext() if printer is None else printer.timeit(f"loading {df_name}")

    with timeit :
        # Apply column
        for header in df.columns :
            if header not in columns.headers :
                warn(f'Skipping opening unknown column with header "{header}"')
                continue
            col = columns.headers[header]
            if col is col_index : continue
            col_name = col.col
            mine = getattr(dataframe, f'{col_name}_mine')
            if not mine : continue
            dtype = col.dtype
            array = df[header].to_numpy().astype(dtype)
            setattr(dataframe, col_name, array)
