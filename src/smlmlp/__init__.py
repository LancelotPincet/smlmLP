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
'detect_spatial_maxima': 'smlmlp.modules.block_LP._functions.blocks.detect.detect_spatial_maxima',
'detect_gain': 'smlmlp.modules.block_LP._functions.blocks.detect.detect_gain',
'detect_snr': 'smlmlp.modules.block_LP._functions.blocks.detect.detect_snr',
'bkgd_spatial_opening': 'smlmlp.modules.block_LP._functions.blocks.bkgd.bkgd_spatial_opening',
'bkgd_combination': 'smlmlp.modules.block_LP._functions.blocks.bkgd.bkgd_combination',
'bkgd_spatial_mean': 'smlmlp.modules.block_LP._functions.blocks.bkgd.bkgd_spatial_mean',
'bkgd_temporal_median': 'smlmlp.modules.block_LP._functions.blocks.bkgd.bkgd_temporal_median',
'signal_temporal_filter': 'smlmlp.modules.block_LP._functions.blocks.signal.signal_temporal_filter',
'signal_combination': 'smlmlp.modules.block_LP._functions.blocks.signal.signal_combination',
'signal_spatial_filter': 'smlmlp.modules.block_LP._functions.blocks.signal.signal_spatial_filter',
'crop_individual_extract': 'smlmlp.modules.block_LP._functions.blocks.crop.crop_individual_extract',
'blink_temporal_on': 'smlmlp.modules.block_LP._functions.blocks.blink.blink_temporal_on',
'blink_spatial_psf': 'smlmlp.modules.block_LP._functions.blocks.blink.blink_spatial_psf',
'column': 'smlmlp.modules.columns_LP._functions.column',
'LocsDataFrame': 'smlmlp.modules.dataframes_LP._functions.LocsDataFrame',
'DetsDataFrame': 'smlmlp.modules.dataframes_LP._functions.DetsDataFrame',
'DataFrame': 'smlmlp.modules.dataframes_LP._functions.DataFrame',
'MainDataFrame': 'smlmlp.modules.dataframes_LP._functions.MainDataFrame'
}



# %% Lazy imports
from corelp import getmodule
__getattr__, __all__ = getmodule(sources)