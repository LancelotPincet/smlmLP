#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

import pandas as pd

from smlmlp import BaseDataFrame



class MainDataFrame(BaseDataFrame) :
    """
    Main detections dataframe hosting dynamic column descriptors.

    Parameters
    ----------
    locs : Locs
        Parent localization container.
    """

    def __init__(self, locs) :
        """Initialize the object."""
        super().__init__(locs)
        pd.DataFrame.__init__(self, index=pd.RangeIndex(start=1, stop=1, step=1))
        self.index.name = self.index_header

    head2save = []
