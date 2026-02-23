#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
import pandas as pd



# %% Function
class DataFrame(pd.DataFrame) :
    '''
    Localization dimensional dataframe
    '''

    def __init__(self, locs) :
        pd.DataFrame.__init__()
        self.locs = locs # Main localization dataframe

