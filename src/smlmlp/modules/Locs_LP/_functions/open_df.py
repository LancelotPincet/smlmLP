#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import columns
from warnings import warn



def open_df(locs, df) :

    # Get dataframe instance
    index = df.index.name
    if index is not None :
        index = index.replace('"', '')
        index = index.replace("'", "")
    else :
        index = 'detection'
    if index not in columns.headers :
        warn(f'Skipping opening unknown dataframe with index "{index}"')
        return None
    col = columns.headers[index]
    df_name = col.df_name
    dataframe =  getattr(locs, df_name)

    # Apply column
    for header in df.columns :
        if header not in columns.headers :
            warn(f'Skipping opening unknown column with header "{header}"')
        col = columns.headers[header]
        col_name = col.col
        mine = getattr(dataframe, f'{col_name}_mine')
        if not mine : continue
        dtype = col.dtype
        array = df[header].to_numpy().astype(dtype)
        setattr(dataframe, col_name, array)