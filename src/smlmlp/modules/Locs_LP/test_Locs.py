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
import pytest
from smlmlp import Locs
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test Locs function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return Locs()

def test_instance(instance) :
    '''
    Test on fixture
    '''
    pass


# %% Returns test
@pytest.mark.parametrize("args, kwargs, expected, message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_returns(args, kwargs, expected, message) :
    '''
    Test Locs return values
    '''
    assert Locs(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test Locs error values
    '''
    with pytest.raises(error, match=error_message) :
        Locs(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)