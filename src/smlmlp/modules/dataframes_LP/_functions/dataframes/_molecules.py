#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



import numpy as np

from smlmlp import DataFrame, column


class molecules(DataFrame) :
    """Molecule-level dataframe aggregated from blinks."""

    @column(headers=['molecule'], dtype=np.uint64, fill=0, save=True, agg='min', index="blinks")
    def mol(self) :
        """Associate blinks into molecule identifiers."""
        from smlmlp import associate_molecules

        return associate_molecules(locs=self.locs)[0]
