#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-25
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : block

"""
This file allows to test block

block : This function is a decorator to be used on block function, which allow to use config for default values.
"""



# %% Libraries
from corelp import debug
import pytest
from smlmlp import block
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test block function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return block()

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
    Test block return values
    '''
    assert block(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test block error values
    '''
    with pytest.raises(error, match=error_message) :
        block(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)