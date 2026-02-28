#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-27
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : Hardware

"""
This class will be inherited by all the hardware objects used to define configuration.
"""



# %% Libraries
from corelp import prop, selfkwargs
from pathlib import Path
import pandas as pd
import numpy as np



# %% Class
class Hardware() :
    '''
    This class will be inherited by all the hardware objects used to define configuration.
    
    Parameters
    ----------
    name : str
        First and only non-keyword argument, will load parameters from preconfigured data.
    '''
    config = None
    constructor = None

    def __init__(self, name=None, /, **kwargs) :
        self.name = name
        kw = {} if name is None else self.models[name]
        selfkwargs(self, kwargs)

    def parameters(self) :
        name = 'Object' if self.name is None else self.name
        string = f'| {self.__class__.__name__.upper()}: {name} parameters |'
        print(f'\n\n*{"-" * (len(string)-2)}*')
        print(string)
        print(f'*{"-" * (len(string)-2)}*\n\n')
        for param in self.tosave :
            print(f'- {param}: {getattr(self, param)}')

    _wl = None
    def load_spectra(self, spectra_name, default=1.) :
        path = Path(__file__).parent / f'_functions/{spectra_name}.csv'
        df = None
        if self.wl is None :
            df = pd.read_csv(path, index_col=0)
            self.wl = df.index.to_numpy()
        value = getattr(self, f'_{spectra_name}', None)
        if self.name is not None and value is None :
            if df is None : df = pd.read_csv(path, index_col=0)
            if self.name in df.columns : value = df[self.name].to_numpy()
        value = 1. if value is None else value
        try :
            if len(value) != len(self.wl) : raise ValueError('Wavelength and Spectra arrays do not have the same length')
        except TypeError :
            value = np.full_like(self.wl, value, dtype=float)
        setattr(self, f'_{spectra_name}', value)
        return self.wl, value



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)