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
from corelp import prop, selfkwargs, folder
from smlmlp import open_locs, save_df, save_metadata
from pathlib import Path
import pandas as pd



# %% Class
class Locs() :
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
    def __new__(cls, locs=None, /, **kwargs) :

        #If Locs object is given as input returns it directly
        if isinstance(locs, cls) :
            return locs

        #Creates a new Locs object
        return object.__new__(cls)



    # Initialize with any object
    def __init__(self, locs=None, /, **kwargs) :

        #Changes attributes from kwargs values
        selfkwargs(self,kwargs)
        if isinstance(locs, Locs) :
            return #break if already a Locs object

        #Opening data
        if locs is not None :
            open_locs(self, locs)



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