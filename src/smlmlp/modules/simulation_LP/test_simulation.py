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


def test_simulation_is_not_implemented() :
    """The simulator placeholder raises the standardized error."""
    with pytest.raises(SyntaxError, match="Not implemented yet") :
        simulation()


if __name__ == "__main__":
    from corelp import test

    test(__file__)
