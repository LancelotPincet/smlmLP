#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-04-25
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : simulation
"""
This file allows to test simulation

simulation : This module simulates SMLM data.
"""

import pytest

from smlmlp import simulation


# %% test simulation


def test_simulation_raises_not_implemented():
    """Simulation module raises SyntaxError until implemented."""
    with pytest.raises(SyntaxError, match="Not implemented yet"):
        simulation()


def test_simulation_rejects_unknown_kwargs():
    """Simulation with unknown parameters still raises not implemented."""
    with pytest.raises(SyntaxError, match="Not implemented yet"):
        simulation(nframes=100, unknown_param=True)


if __name__ == "__main__":
    from corelp import test

    test(__file__)