#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-08-28
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP

"""
A python library for Single Molecule Localization Microscopy.
"""



# %% Source import
sources = {
'Config': 'smlmlp.modules.Config_LP.Config',
'Locs': 'smlmlp.modules.Locs_LP.Locs',
'block': 'smlmlp.modules.block_LP.block',
'columns': 'smlmlp.modules.columns_LP.columns',
'computer': 'smlmlp.modules.computer_LP.computer',
'dataframes': 'smlmlp.modules.dataframes_LP.dataframes',
'metadatum': 'smlmlp.modules.Config_LP._functions.metadatum',
'Channel': 'smlmlp.modules.Config_LP._functions.Channel',
'Camera': 'smlmlp.modules.Config_LP._functions.Camera',
'save_df': 'smlmlp.modules.Locs_LP._functions.save_df',
'open_df': 'smlmlp.modules.Locs_LP._functions.open_df',
'load_data': 'smlmlp.modules.block_LP._functions.blocks.load.load_data',
'bkgd_spatialmean': 'smlmlp.modules.block_LP._functions.blocks.bkgd.bkgd_spatialmean',
'bkgd_spatialmini': 'smlmlp.modules.block_LP._functions.blocks.bkgd.bkgd_spatialmini',
'bkgd_temporalmedian': 'smlmlp.modules.block_LP._functions.blocks.bkgd.bkgd_temporalmedian',
'signal_spatialfilter': 'smlmlp.modules.block_LP._functions.blocks.signal.signal_spatialfilter',
'signal_temporalfilter': 'smlmlp.modules.block_LP._functions.blocks.signal.signal_temporalfilter',
'blink_psf': 'smlmlp.modules.block_LP._functions.blocks.blink.blink_psf',
'blink_on': 'smlmlp.modules.block_LP._functions.blocks.blink.blink_on',
'column': 'smlmlp.modules.columns_LP._functions.column',
'LocsDataFrame': 'smlmlp.modules.dataframes_LP._functions.LocsDataFrame',
'DetsDataFrame': 'smlmlp.modules.dataframes_LP._functions.DetsDataFrame',
'DataFrame': 'smlmlp.modules.dataframes_LP._functions.DataFrame',
'MainDataFrame': 'smlmlp.modules.dataframes_LP._functions.MainDataFrame'
}



# %% Lazy imports
from corelp import getmodule
__getattr__, __all__ = getmodule(sources)