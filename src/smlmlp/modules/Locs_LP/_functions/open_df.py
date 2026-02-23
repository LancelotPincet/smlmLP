#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
import pandas as pd



def open_df(df) :
    index = df.index.name
    if index is not None :
        index = index.replace('"', '')
        index = index.replace("'", "")
    else :
        index = 'loc'
    idx = self.head2col.get(index)
    if idx is None or idx not in self.index2df :
        warn(f'Skipping opening unknown dataframe with index "{index}"')
        return

    #Set df
    df_name = self.index2df[idx]
    if self.col2df[idx] != 'df' :
        _df, self.df = self._df, df_name
    else :
        _df = self._df
    predf = getattr(self, df_name, None)
    if predf is None :
        predf = pd.DataFrame()
        dtype = self.col2dtype[idx]
        predf[index] = np.asarray(df.index.to_numpy(), dtype=dtype)
        predf.set_index(index, inplace=True)
        setattr(self, df_name, predf)

    #Copying columns
    for header in df.columns :
        header = header.replace('"', '')
        header = header.replace("'", "")
        if header not in self.head2col :
            warn(f'Skipping opening unknown column "{header}"')
            continue
        array = df[header].to_numpy()
        col = self.head2col[header]
        setattr(self, col, array)

    #reset df
    self.df = _df


