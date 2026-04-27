#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



import numpy as np

from smlmlp import DataFrame, column


class voxels(DataFrame) :
    """Voxel-level dataframe aggregated from points."""

    @column(headers=['voxel'], dtype=np.uint64, fill=0, save=True, agg='min', index="points")
    def vox(self) :
        """Assign voxel identifiers."""
        return self.fr * (self.y_shape * self.x_shape) + np.round(self.y / self.y_pixel) * self.x_shape + np.round(self.x / self.x_pixel)



    # Density

    @column(headers=['density [loc.um-2]'], dtype=np.float32, fill=0, save=True, agg='mean')
    def density(self) :
        """Estimate local density for each voxel."""
        from smlmlp import associate_density

        return associate_density(locs=self.locs)[0]
