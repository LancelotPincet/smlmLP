#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-25
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : Config

"""
This file allows to test Config

Config : This class stores configuration values in an object that can be saved and loaded.
"""
import numpy as np

from smlmlp import Config
from smlmlp.modules.Config_LP.Config import json_convert


def test_metadata_descriptor_converts_and_groups_values() :
    """Metadata descriptors store explicit values in grouped metadata."""
    config = Config(nframes=np.int64(12), cuda=True)

    assert config.nframes == 12
    assert config.cuda == 1
    assert config.metadata["Loads"]["nframes"] == 12
    assert config.metadata["Loads"]["cuda"] == 1


def test_dynamic_camera_metadata_descriptors() :
    """Camera metadata descriptors broadcast and collect camera values."""
    config = Config(ncameras=2)
    config.cameras_npixels = [(10, 20), (30, 40)]

    assert config.cameras_npixels == [[10, 20], [30, 40]]
    assert config.metadata["Cameras"]["cameras_npixels"] == [[10, 20], [30, 40]]


def test_default_camera_quantum_efficiency_is_numeric() :
    """Default camera QE metadata returns the documented numeric value."""
    config = Config()

    assert config.cameras_QE == [0.8]
    assert config.channels_QE == [0.8]


def test_channel_psf_wavelength_na_round_trip() :
    """Channel PSF wavelength/NA proxy reads and writes sigma metadata."""
    channel = Config().channels[0]

    channel.psf_wl_na_nm = 500.0

    assert channel.psf_wl_na_nm == 500.0
    assert channel.psf_sigma_nm == 105.0


def test_method_and_target_descriptor_typos_are_resolved() :
    """Demix2D and dye-count descriptors store independent values."""
    config = Config(demix_method="flux", demix2d_method="spectral", dyes=["a", "b"])

    assert config.demix_method == "flux"
    assert config.demix2d_method == "spectral"
    assert config.ndyes == 2


def test_json_convert_handles_numpy_values() :
    """JSON conversion normalizes numpy scalar and array values."""
    value = json_convert([np.bool_(True), np.array([1, 2], dtype=np.int64)])

    assert value == [True, [1, 2]]


if __name__ == "__main__":
    from corelp import test

    test(__file__)
