#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-20
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : Locs

"""
This file allows to test Locs

Locs : This class define objects corresponding to localizations sets for one experiment.
"""



# %% Libraries
from corelp import debug
import numpy as np
from smlmlp import Locs
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test Locs function
    '''
    
    # Init
    locs = Locs()

    # Simulate random data
    argunsort = np.arange(10000)
    np.random.shuffle(argunsort)
    array = np.hstack([np.full(100, i) for i in range(1, 101)]) # Stack of constants to apply to xdet and fr (should be paired)
    x_det = array.astype(np.float32)[argunsort]
    fr = array.astype(np.uint32)[argunsort]

    # Set data in locs object
    locs.detections.fr = fr
    locs.detections.x_det = x_det
    assert (locs.detections.x_det == x_det).all()
    assert (locs.detections.fr == fr).all()

    # Looking at frames dataframe with merging and spreads
    x_det_frames = locs.frames.x_det # merging
    assert (locs.frames.fr == x_det_frames).all()
    locs.frames.dx = x_det_frames # spreading
    assert (locs.detections.dx == x_det).all()

    # Saving and loading
    path = debug_folder / 'locs'
    locs.save(path)
    locs2 = Locs(path)

    # Filter
    mask = locs.detections.x_det > 50
    locs3 = locs.filter(mask=mask)
    


# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)