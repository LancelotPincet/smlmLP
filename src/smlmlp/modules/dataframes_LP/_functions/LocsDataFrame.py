#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
import pandas as pd
from smlmlp import DataFrame



# %% Function
class LocsDataFrame(DataFrame) :
    '''
    Localization main dataframe
    '''

    def __init__(self) :
        pd.DataFrame.__init__()
        self.locs = self