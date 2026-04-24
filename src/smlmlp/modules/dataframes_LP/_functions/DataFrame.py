#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
import pandas as pd
import numpy as np
from smlmlp import BaseDataFrame



# %% Function
class DataFrame(BaseDataFrame) :
    '''
    Localization dimensional dataframe
    '''

    def __init__(self, locs) :
        super().__init__(locs)
        index = np.sort(self.locs.detections[self.index_header].unique())
        if index[0] == 0 :
            index = index[1:]
        pd.DataFrame.__init__(self, index=index)
        self.index.name = self.index_header
