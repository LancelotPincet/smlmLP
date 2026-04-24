#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import DataFrame, column
import numpy as np



# %% Function
class voxels(DataFrame) :
    '''
    Voxels dataframe
    '''

    @column(headers=['voxel'], dtype=np.uint64, save=True, agg='min', index="points")
    def vox(self) :
        from smlmlp import index_voxels
        return index_voxels(locs=self.locs)[0]



    # Density

    @column(headers=['density [loc.um-2]'], dtype=np.float32, save=True, agg='mean')
    def density(self) :
        from smlmlp import neighbors_density
        return neighbors_density(locs=self.locs)[0]
