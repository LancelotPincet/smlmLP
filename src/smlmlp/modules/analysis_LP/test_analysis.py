#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-04-21
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : analysis

"""
This file allows to test analysis

analysis : This decorator defines a function as an analysis on a Locs object.
"""



# %% Libraries
from corelp import debug
import pytest
from smlmlp import analysis
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test analysis function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return analysis()

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
    Test analysis return values
    '''
    assert analysis(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test analysis error values
    '''
    with pytest.raises(error, match=error_message) :
        analysis(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)