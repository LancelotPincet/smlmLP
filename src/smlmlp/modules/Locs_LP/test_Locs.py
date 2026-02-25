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
    array = np.hstack([np.full(100, i) for i in range(100)]) # Stack of constants to apply to xdet and fr (should be paired)
    xdet = array.astype(np.float32)[argunsort]
    fr = array.astype(np.uint32)[argunsort]

    # Set data in locs object
    locs.detections.fr = fr
    locs.detections.xdet = xdet
    assert (locs.detections.xdet == xdet).all()
    assert (locs.detections.fr == fr).all()

    # Looking at frames dataframe with merging and spreads
    xdet_frames = locs.frames.xdet # merging
    assert (locs.frames.fr == xdet_frames).all()
    locs.frames.dx = xdet_frames # spreading
    assert (locs.detections.dx == xdet).all()



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)