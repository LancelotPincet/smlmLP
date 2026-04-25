#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



import numpy as np

from smlmlp import DataFrame, column


class voxels(DataFrame) :
    """Voxel-level dataframe aggregated from points."""

    @column(headers=['voxel'], dtype=np.uint64, save=True, agg='min', index="points")
    def vox(self) :
        """Assign voxel identifiers."""
        from smlmlp import index_voxels

        return index_voxels(locs=self.locs)[0]



    # Density

    @column(headers=['density [loc.um-2]'], dtype=np.float32, save=True, agg='mean')
    def density(self) :
        """Estimate local density for each voxel."""
        from smlmlp import neighbors_density

        return neighbors_density(locs=self.locs)[0]
