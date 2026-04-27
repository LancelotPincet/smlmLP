#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



import numpy as np

from smlmlp import DataFrame, column


class elements(DataFrame) :
    """Element-level dataframe aggregated from blinks."""

    @column(headers=['element'], dtype=np.uint16, fill=0, save=True, agg='min', index="blinks")
    def elm(self) :
        """Assign element identifiers."""
        from smlmlp import clustering_dbscan

        return clustering_dbscan(locs=self.locs)[0]


    # Fuse

    @column(headers=['element x centroid [nm]'], dtype=np.float32, fill=np.nan, save=False, agg='mean')
    def elm_x0(self) :
        """Alias the element x centroid to the x coordinate."""
        return "x"

    @column(headers=['element y centroid [nm]'], dtype=np.float32, fill=np.nan, save=False, agg='mean')
    def elm_y0(self) :
        """Alias the element y centroid to the y coordinate."""
        return "y"
