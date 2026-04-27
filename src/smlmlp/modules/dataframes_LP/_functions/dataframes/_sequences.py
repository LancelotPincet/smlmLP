#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



import numpy as np

from smlmlp import DataFrame, column


class sequences(DataFrame) :
    """Sequence-level dataframe aggregated from points."""

    @column(headers=['sequence'], dtype=np.uint32, fill=0, save=True, agg='min', index="points")
    def seq(self) :
        """Assign sequence identifiers from frame numbers."""
        array = self.fr // self.locs.config.frames_per_sequence
        array[self.fr != 0] += 1
        return array
