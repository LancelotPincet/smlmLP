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
'analysis': 'smlmlp.modules.analysis_LP.analysis',
'block': 'smlmlp.modules.block_LP.block',
'columns': 'smlmlp.modules.columns_LP.columns',
'computer': 'smlmlp.modules.computer_LP.computer',
'dataframes': 'smlmlp.modules.dataframes_LP.dataframes',
'Camera': 'smlmlp.modules.Config_LP._functions.Camera',
'metadatum': 'smlmlp.modules.Config_LP._functions.metadatum',
'Channel': 'smlmlp.modules.Config_LP._functions.Channel',
'save_df': 'smlmlp.modules.Locs_LP._functions.save_df',
'open_df': 'smlmlp.modules.Locs_LP._functions.open_df',
'analysis_det2blk': 'smlmlp.modules.analysis_LP._functions.analysis.analysis_det2blk',
'analysis_template': 'smlmlp.modules.analysis_LP._functions.analysis.analysis_template',
'registrate_pcc_shift': 'smlmlp.modules.block_LP._functions.registration.registrate_pcc_shift',
'registrate_solve_redundant': 'smlmlp.modules.block_LP._functions.registration.registrate_solve_redundant',
'registrate_optimize_images': 'smlmlp.modules.block_LP._functions.registration.registrate_optimize_images',
'bkgd_spatial_mean': 'smlmlp.modules.block_LP._functions.background.bkgd_spatial_mean',
'bkgd_spatial_opening': 'smlmlp.modules.block_LP._functions.background.bkgd_spatial_opening',
'bkgd_combination': 'smlmlp.modules.block_LP._functions.background.bkgd_combination',
'bkgd_temporal_median': 'smlmlp.modules.block_LP._functions.background.bkgd_temporal_median',
'crop_remove_bkgd': 'smlmlp.modules.block_LP._functions.crop.crop_remove_bkgd',
'crop_individual_extract': 'smlmlp.modules.block_LP._functions.crop.crop_individual_extract',
'load_data': 'smlmlp.modules.block_LP._functions.loading.load_data',
'load_chunking': 'smlmlp.modules.block_LP._functions.loading.load_chunking',
'signal_spatial_filter': 'smlmlp.modules.block_LP._functions.signal.signal_spatial_filter',
'signal_combination': 'smlmlp.modules.block_LP._functions.signal.signal_combination',
'signal_temporal_filter': 'smlmlp.modules.block_LP._functions.signal.signal_temporal_filter',
'locs_individual_gaussfit': 'smlmlp.modules.block_LP._functions.localization.locs_individual_gaussfit',
'locs_individual_splinefit': 'smlmlp.modules.block_LP._functions.localization.locs_individual_splinefit',
'locs_individual_isogaussfit': 'smlmlp.modules.block_LP._functions.localization.locs_individual_isogaussfit',
'locs_individual_barycenter': 'smlmlp.modules.block_LP._functions.localization.locs_individual_barycenter',
'blink_spatial_psf': 'smlmlp.modules.block_LP._functions.blink.blink_spatial_psf',
'blink_temporal_on': 'smlmlp.modules.block_LP._functions.blink.blink_temporal_on',
'detect_spatial_maxima': 'smlmlp.modules.block_LP._functions.detection.detect_spatial_maxima',
'detect_gain': 'smlmlp.modules.block_LP._functions.detection.detect_gain',
'detect_snr': 'smlmlp.modules.block_LP._functions.detection.detect_snr',
'column': 'smlmlp.modules.columns_LP._functions.column',
'MainDataFrame': 'smlmlp.modules.dataframes_LP._functions.MainDataFrame',
'DetsDataFrame': 'smlmlp.modules.dataframes_LP._functions.DetsDataFrame',
'LocsReceiver': 'smlmlp.modules.dataframes_LP._functions.LocsReceiver',
'DataFrame': 'smlmlp.modules.dataframes_LP._functions.DataFrame'
}



# %% Lazy imports
from corelp import getmodule
__getattr__, __all__ = getmodule(sources)