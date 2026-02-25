#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-20
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : Locs

"""
This class define objects corresponding to localizations sets for one experiment.
"""



# %% Libraries
from corelp import folder
from smlmlp import open_df, save_df, LocsDataFrame, DetsDataFrame
from pathlib import Path
import pandas as pd
import numpy as np



# %% Class
class Locs(LocsDataFrame) :
    '''
    This class define objects corresponding to localizations sets for one experiment.

    Parameters
    ----------
    a : int or float
        TODO.

    Attributes
    ----------
    _attr : int or float
        TODO.

    Examples
    --------
    >>> from smlmlp import Locs
    ...
    >>> instance = Locs(TODO)
    '''


    # Creates Locs objects
    def __new__(cls, source=None, /, **kwargs) :

        #If Locs object is given as input returns it directly
        if isinstance(source, cls) :
            return source

        #Creates a new Locs object
        return object.__new__(cls)



    # Initialize with any object
    df_dict = None
    def __init__(self, source=None, config=None) :
        self.df_dict = dict(detections=DetsDataFrame(self))

        #Changes attributes from kwargs values
        if isinstance(source, Locs) :
            return #break if already a Locs object

        #Opening data
        if source is not None :
            open(self, source)
    @property
    def detections(self) :
        return self.df_dict['detections']



    def filter(self, *filter_names, mask=None) :
        #Get filter mask
        filters = [getattr(self, name, None) for name in filter_names]
        if mask is not None :
            filters += [mask]
        mask = np.logical_and.reduce(filters)
        df = self.df[mask].drop(columns=[self.col2head[name] for name in filter_names])

        #filter all dataframe where mask can be spread
        dfs = [df]
        name = self._df
        supname = self._df
        subname = self.df2subdf[supname]
        while supname != subname :
            self.df = subname
            if self.df is None :
                break
            self.index = self.df2index[supname]
            mask = spread(mask, self.index, desorting=self.desorting, unique=self.unique, counts=self.counts, cumsum=self.cumsum, resort=self.resort)
            df = self.df[mask]
            supname = subname
            subname = self.df2subdf[supname]
            dfs.append(df)

        cls = self.__class__
        self.df, self.index = name, None
        return cls(dfs, **self.metadata, _df=self._df, parent=self)



    def open(self, locs) :
        if locs is None :
            return
        elif isinstance(locs, pd.DataFrame) :
            open_df(locs)
        elif isinstance(data, dict) :
            selfkwargs(self, data)
        elif isinstance(data,str) or isinstance(data,Path('').__class__) :
            open_file(data)
        elif isinstance(data, list) or isinstance(data, tuple) :
            for d in data :
                open_data(d)
        else :
            raise SyntaxError('data type was not recognized for Locs')



    def save(self, path, file=None) :
        path = Path(path) if file is None else Path(path) / file
        saving_folder = folder(path.with_suffix(''), warning=False)
        stem = saving_folder.stem
        mainpath = saving_folder / f'{stem}.csv'
        
        # Saving df
        for df_name in self.df_names :

            #Get df
            df = getattr(self, df_name, None)
            if df is None or len(df.columns) == 0 : continue
            path = mainpath.with_stem(f'{stem}_[{df_name}]')
            save_df(df, path, df_name, self.head2save)

        # Saving metadata
        mainpath = saving_folder / f'{stem}_[metadata].json'
        save_metadata(self, mainpath)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)