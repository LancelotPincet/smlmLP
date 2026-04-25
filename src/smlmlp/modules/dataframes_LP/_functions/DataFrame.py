#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

import numpy as np
import pandas as pd

from smlmlp import BaseDataFrame



class DataFrame(BaseDataFrame) :
    """
    Intermediate localization dataframe indexed from detection columns.

    Parameters
    ----------
    locs : Locs
        Parent localization container.
    """

    def __init__(self, locs) :
        """Initialize the object."""
        super().__init__(locs)
        index = np.sort(self.locs.detections[self.index_header].unique())
        if index[0] == 0 :
            index = index[1:]
        pd.DataFrame.__init__(self, index=index)
        self.index.name = self.index_header
