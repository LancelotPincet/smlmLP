#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-04-25
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : simulation

"""
Simulate SMLM (Single Molecule Localization Microscopy) data.

This module provides functionality to simulate various aspects of SMLM
experiments including photon statistics, blinking behavior, PSF generation,
and noise models.
"""

# %% Simulation module


def simulation(**kwargs):
    """
    Simulate SMLM data.

    Parameters
    ----------
    **kwargs
        Simulation parameters including:
        - nframes : int
            Number of frames to simulate.
        - nlocalizations : int
            Number of unique localizations.
        - photons : array-like
            Photon counts per localization.
        - x_nm, y_nm : array-like
            X/Y positions in nanometers.
        - z_nm : array-like, optional
            Z positions in nanometers.
        - frame_nm : array-like
            Frame numbers.
        - channels : array-like, optional
            Channel assignments.

    Returns
    -------
    Locs
        Simulated localization data.

    Raises
    ------
    SyntaxError
        Always raised until the simulator is implemented.

    Examples
    --------
    >>> from smlmlp import simulation
    >>> simulation()  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    SyntaxError: Not implemented yet.
    """
    raise SyntaxError("Not implemented yet.")


if __name__ == "__main__":
    from corelp import test

    test(__file__)