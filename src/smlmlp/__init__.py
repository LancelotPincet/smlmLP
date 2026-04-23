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
'timeloc_spadarray': 'smlmlp.modules.analysis_LP._functions.timeloc.timeloc_spadarray',
'timeloc_singlespad': 'smlmlp.modules.analysis_LP._functions.timeloc.timeloc_singlespad',
'flim_dpflim': 'smlmlp.modules.analysis_LP._functions.flim.flim_dpflim',
'flim_iflim': 'smlmlp.modules.analysis_LP._functions.flim.flim_iflim',
'flim_tcspc': 'smlmlp.modules.analysis_LP._functions.flim.flim_tcspc',
'metric_frc': 'smlmlp.modules.analysis_LP._functions.metric.metric_frc',
'metric_squirrel': 'smlmlp.modules.analysis_LP._functions.metric.metric_squirrel',
'metric_nena': 'smlmlp.modules.analysis_LP._functions.metric.metric_nena',
'metric_overloc': 'smlmlp.modules.analysis_LP._functions.metric.metric_overloc',
'modloc_sequential': 'smlmlp.modules.analysis_LP._functions.modloc.modloc_sequential',
'modloc_demodulated': 'smlmlp.modules.analysis_LP._functions.modloc.modloc_demodulated',
'zaxis_modloc': 'smlmlp.modules.analysis_LP._functions.zaxis.zaxis_modloc',
'zaxis_timeloc': 'smlmlp.modules.analysis_LP._functions.zaxis.zaxis_timeloc',
'zaxis_biplane': 'smlmlp.modules.analysis_LP._functions.zaxis.zaxis_biplane',
'zaxis_donald': 'smlmlp.modules.analysis_LP._functions.zaxis.zaxis_donald',
'zaxis_miet': 'smlmlp.modules.analysis_LP._functions.zaxis.zaxis_miet',
'zaxis_daisy': 'smlmlp.modules.analysis_LP._functions.zaxis.zaxis_daisy',
'zaxis_astig': 'smlmlp.modules.analysis_LP._functions.zaxis.zaxis_astig',
'zaxis_qtirf': 'smlmlp.modules.analysis_LP._functions.zaxis.zaxis_qtirf',
'orient_polar2d': 'smlmlp.modules.analysis_LP._functions.orient.orient_polar2d',
'orient_polar3d': 'smlmlp.modules.analysis_LP._functions.orient.orient_polar3d',
'drift_aim': 'smlmlp.modules.analysis_LP._functions.drift.drift_aim',
'drift_cc': 'smlmlp.modules.analysis_LP._functions.drift.drift_cc',
'drift_comet': 'smlmlp.modules.analysis_LP._functions.drift.drift_comet',
'drift_ms': 'smlmlp.modules.analysis_LP._functions.drift.drift_ms',
'demix_flux': 'smlmlp.modules.analysis_LP._functions.demix.demix_flux',
'demix_spectral': 'smlmlp.modules.analysis_LP._functions.demix.demix_spectral',
'index_blinks': 'smlmlp.modules.analysis_LP._functions.index.index_blinks',
'index_sequences': 'smlmlp.modules.analysis_LP._functions.index.index_sequences',
'index_molecules': 'smlmlp.modules.analysis_LP._functions.index.index_molecules',
'index_points': 'smlmlp.modules.analysis_LP._functions.index.index_points',
'index_frames': 'smlmlp.modules.analysis_LP._functions.index.index_frames',
'index_channels': 'smlmlp.modules.analysis_LP._functions.index.index_channels',
'index_elements': 'smlmlp.modules.analysis_LP._functions.index.index_elements',
'index_voxels': 'smlmlp.modules.analysis_LP._functions.index.index_voxels',
'index_pixels': 'smlmlp.modules.analysis_LP._functions.index.index_pixels',
'index_dyes': 'smlmlp.modules.analysis_LP._functions.index.index_dyes',
'image_densitymap': 'smlmlp.modules.analysis_LP._functions.image.image_densitymap',
'image_colmap': 'smlmlp.modules.analysis_LP._functions.image.image_colmap',
'image_smlm3d': 'smlmlp.modules.analysis_LP._functions.image.image_smlm3d',
'image_smlm': 'smlmlp.modules.analysis_LP._functions.image.image_smlm',
'image_vectors': 'smlmlp.modules.analysis_LP._functions.image.image_vectors',
'image_wf': 'smlmlp.modules.analysis_LP._functions.image.image_wf',
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
'BaseDataFrame': 'smlmlp.modules.dataframes_LP._functions.BaseDataFrame',
'MainDataFrame': 'smlmlp.modules.dataframes_LP._functions.MainDataFrame',
'DetsDataFrame': 'smlmlp.modules.dataframes_LP._functions.DetsDataFrame',
'LocsReceiver': 'smlmlp.modules.dataframes_LP._functions.LocsReceiver',
'DataFrame': 'smlmlp.modules.dataframes_LP._functions.DataFrame'
}



# %% Lazy imports
from corelp import getmodule
__getattr__, __all__ = getmodule(sources)