#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



import numpy as np

from smlmlp import DataFrame, column


class dyes(DataFrame) :
    """Dye-level dataframe aggregated from blinks."""

    @column(headers=['dye'], dtype=np.uint8, save=True, agg='min', index="blinks")
    def dye(self) :
        """Assign dye identifiers from demixing or a single-dye default."""
        if self.locs.config.ndyes == 1 :
            return np.ones(self.locs.ndetections, dtype=np.uint8)
        else :
            from smlmlp import demix_histogram

            return demix_histogram(self.demix, locs=self.locs)[0]
