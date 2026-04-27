#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-04-27
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : start_server

"""
This file allows to test start_server

start_server : This function starts a server with a semaphore for managing (V)RAM usage.
"""



# %% Libraries
from corelp import debug
import pytest
from smlmlp import start_server
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test start_server function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return start_server()

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
    Test start_server return values
    '''
    assert start_server(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test start_server error values
    '''
    with pytest.raises(error, match=error_message) :
        start_server(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)