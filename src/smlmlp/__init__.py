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
'simulation': 'smlmlp.modules.simulation_LP.simulation',
'Camera': 'smlmlp.modules.Config_LP._functions.Camera',
'metadatum': 'smlmlp.modules.Config_LP._functions.metadatum',
'Channel': 'smlmlp.modules.Config_LP._functions.Channel',
'save_df': 'smlmlp.modules.Locs_LP._functions.save_df',
'open_df': 'smlmlp.modules.Locs_LP._functions.open_df',
'aggregate_ratio': 'smlmlp.modules.analysis_LP._functions.aggregate.aggregate_ratio',
'aggregate_flux': 'smlmlp.modules.analysis_LP._functions.aggregate.aggregate_flux',
'associate_consecutive_frames': 'smlmlp.modules.analysis_LP._functions.associate.associate_consecutive_frames',
'associate_molecules': 'smlmlp.modules.analysis_LP._functions.associate.associate_molecules',
'associate_different_channels': 'smlmlp.modules.analysis_LP._functions.associate.associate_different_channels',
'associate_density': 'smlmlp.modules.analysis_LP._functions.associate.associate_density',
'associate_consecutive_frames_radius': 'smlmlp.modules.analysis_LP._functions.associate.associate_consecutive_frames_radius',
'transform_locs': 'smlmlp.modules.analysis_LP._functions.transform.transform_locs',
'inv_transform_loc': 'smlmlp.modules.analysis_LP._functions.transform.inv_transform_loc',
'metric_frc': 'smlmlp.modules.analysis_LP._functions.metric.metric_frc',
'metric_squirrel': 'smlmlp.modules.analysis_LP._functions.metric.metric_squirrel',
'metric_nena': 'smlmlp.modules.analysis_LP._functions.metric.metric_nena',
'metric_overloc': 'smlmlp.modules.analysis_LP._functions.metric.metric_overloc',
'metric_photophysics': 'smlmlp.modules.analysis_LP._functions.metric.metric_photophysics',
'lost_frames': 'smlmlp.modules.analysis_LP._functions.lost.lost_frames',
'lost_channels': 'smlmlp.modules.analysis_LP._functions.lost.lost_channels',
'modloc_sequential': 'smlmlp.modules.analysis_LP._functions.modloc.modloc_sequential',
'modloc_demodulated': 'smlmlp.modules.analysis_LP._functions.modloc.modloc_demodulated',
'modloc_axial': 'smlmlp.modules.analysis_LP._functions.modloc.modloc_axial',
'modloc_transverse': 'smlmlp.modules.analysis_LP._functions.modloc.modloc_transverse',
'calibration_fuse': 'smlmlp.modules.analysis_LP._functions.calibration.calibration_fuse',
'calibration_zstacks': 'smlmlp.modules.analysis_LP._functions.calibration.calibration_zstacks',
'calibration_spheres': 'smlmlp.modules.analysis_LP._functions.calibration.calibration_spheres',
'calibration_flim': 'smlmlp.modules.analysis_LP._functions.calibration.calibration_flim',
'calibration_convert': 'smlmlp.modules.analysis_LP._functions.calibration.calibration_convert',
'orient_polar2d': 'smlmlp.modules.analysis_LP._functions.orient.orient_polar2d',
'orient_polar3d': 'smlmlp.modules.analysis_LP._functions.orient.orient_polar3d',
'drift_aim': 'smlmlp.modules.analysis_LP._functions.drift.drift_aim',
'drift_meanshift': 'smlmlp.modules.analysis_LP._functions.drift.drift_meanshift',
'drift_crosscorr': 'smlmlp.modules.analysis_LP._functions.drift.drift_crosscorr',
'drift_comet': 'smlmlp.modules.analysis_LP._functions.drift.drift_comet',
'clustering_dbscan': 'smlmlp.modules.analysis_LP._functions.clustering.clustering_dbscan',
'demix_histogram': 'smlmlp.modules.analysis_LP._functions.demix.demix_histogram',
'image_pixel': 'smlmlp.modules.analysis_LP._functions.image.image_pixel',
'image_picker': 'smlmlp.modules.analysis_LP._functions.image.image_picker',
'image_colmap': 'smlmlp.modules.analysis_LP._functions.image.image_colmap',
'image_smlm3d': 'smlmlp.modules.analysis_LP._functions.image.image_smlm3d',
'image_smlm': 'smlmlp.modules.analysis_LP._functions.image.image_smlm',
'image_vectors': 'smlmlp.modules.analysis_LP._functions.image.image_vectors',
'image_stackmap': 'smlmlp.modules.analysis_LP._functions.image.image_stackmap',
'globloc_fit': 'smlmlp.modules.block_LP._functions.globlocalization.globloc_fit',
'registrate_pcc_shift': 'smlmlp.modules.block_LP._functions.registration.registrate_pcc_shift',
'registrate_solve_redundant_shift': 'smlmlp.modules.block_LP._functions.registration.registrate_solve_redundant_shift',
'registrate_ecc_affine': 'smlmlp.modules.block_LP._functions.registration.registrate_ecc_affine',
'registrate_optimize_images': 'smlmlp.modules.block_LP._functions.registration.registrate_optimize_images',
'registrate_solve_redundant_affine': 'smlmlp.modules.block_LP._functions.registration.registrate_solve_redundant_affine',
'globdet_channel': 'smlmlp.modules.block_LP._functions.globdetection.globdet_channel',
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