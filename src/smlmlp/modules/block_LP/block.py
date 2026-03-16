#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-25
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : block

"""
This function is a decorator to be used on block function, which allow to use config for default values.
"""



# %% Libraries
import functools
import time
import inspect



# %% Function
def block() :
    '''
    This function is a decorator to be used on block function, which allow to use config for default values.
    A decorated function can use a config object with config=config_object to defined default value of all the keyword only parameters.
    You can also use a Locs object via the locs=mylocsobject attribute for all parameters linked to localizations
    This decorator works for functions and generators.
    Computation time will be added for each call of the decorated function in block.times dictionary.
    
    Returns
    -------
    function : function
        Decorated function.

    Examples
    --------
    >>> from smlmlp import block
    ...
    >>> block()
    ... def myfunc() :
    ...     return long_process()
    ...
    >>> result = myfunc(config=config_object)
    '''

    def decorator(function) :
        name = function.__name__
        @functools.wraps(function)
        def wrapper(*args, config=None, locs=None, **kwargs) -> None :

            # Manages kwargs from config and locs
            if config is not None :
                signature = inspect.signature(function)
                kw = {}
                for pname, param in signature.parameters.items() :
                    if (param.kind is inspect.Parameter.KEYWORD_ONLY or param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD) and hasattr(config, pname) :
                        kw[pname] = getattr(config, pname)
                    elif (param.kind is inspect.Parameter.KEYWORD_ONLY or param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD) and hasattr(locs, pname) :
                        kw[pname] = getattr(locs, pname)
                kw.update(kwargs)
                kwargs = kw

            #Launch timed function
            tic = time.perf_counter()
            result = function(*args, **kwargs)
            toc = time.perf_counter()

            # Check if generator, if normal function just exit here
            if not inspect.isgenerator(result):
                if name in block.times :
                    block.times[name] += toc-tic
                else :
                    block.times[name] = toc-tic
                return result
            
            # If is a generator
            def generator_wrapper():
                while True:
                    try:
                        tic = time.perf_counter()
                        value = next(result)
                        toc = time.perf_counter()
                        if name in block.times :
                            block.times[name] += toc-tic
                        else :
                            block.times[name] = toc-tic
                        yield value
                    except StopIteration:
                        break
            return generator_wrapper()

        return wrapper
    return decorator
block.times = {}



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)