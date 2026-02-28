#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-27
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : Hardware

"""
This file allows to test Hardware

Hardware : This class will be inherited by all the hardware objects used to define configuration.
"""



# %% Libraries
from corelp import debug
import pytest
from smlmlp import Hardware
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test Hardware function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return Hardware()

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
    Test Hardware return values
    '''
    assert Hardware(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test Hardware error values
    '''
    with pytest.raises(error, match=error_message) :
        Hardware(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)