#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



import numpy as np

from smlmlp import DataFrame, column


class tracks(DataFrame) :
    """Track-level dataframe aggregated from points."""

    @column(headers=['track'], dtype=np.uint64, save=True, agg='min', index="points")
    def trk(self) :
        """Associate consecutive frames into track identifiers."""
        from smlmlp import associate_consecutive_frames

        return associate_consecutive_frames(association_radius_nm=self.locs.config.track_association_radius_nm, z=None, locs=self.locs)[0]
