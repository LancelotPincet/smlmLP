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

import functools
import time
import inspect
from smlmlp import metadatum

def block(timeit=True):
    """
    Decorator for block functions to read values from a config object.

    A decorated function can use a config object with config=config_object to define default
    value of all the keyword only parameters. You can also use a Locs object via
    the locs=mylocsobject attribute for all parameters linked to localizations.

    This decorator works for functions and generators.
    Computation time will be added for each call of the decorated function in block.times dictionary.

    Parameters
    ----------
    timeit : bool, optional
        Whether to record execution time.

    Returns
    -------
    function
        Decorated function.

    Examples
    --------
    >>> from smlmlp import block
    >>> @block()
    ... def myfunc(config=None, *, cuda=False, parallel=False):
    ...     return cuda, parallel, {}
    >>> result = myfunc(config=config_object)
    """

    def decorator(function):
        """Create the decorated callable."""
        name = function.__name__
        @functools.wraps(function)
        def wrapper(*args, config=None, **kwargs):
            """Call the wrapped function with normalized arguments."""
            kwargs = {key: value for key, value in kwargs.items() if value is not None}
            if config is not None:
                signature = inspect.signature(function)
                kw = {}
                for pname, param in signature.parameters.items():
                    attr = kwargs.pop(pname, None)
                    if (param.kind is inspect.Parameter.KEYWORD_ONLY or param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD) and hasattr(config, pname):
                        attr = getattr(config, pname) if attr is None else attr
                        if any(pname == datum for group in metadatum.groups.values() for datum in group):
                            setattr(config, pname, attr)
                        kw[pname] = attr
                kw.update(kwargs)
                kwargs = kw

            # Launch timed function
            tic = time.perf_counter()
            result = function(*args, **kwargs)
            toc = time.perf_counter()

            # Check if generator, if normal function just exit here
            if not inspect.isgenerator(result):
                if timeit:
                    if name in block.times:
                        block.times[name] += toc-tic
                    else:
                        block.times[name] = toc-tic
                return result

            # If is a generator
            def generator_wrapper():
                """Yield values while measuring generator execution time."""
                while True:
                    try:
                        tic = time.perf_counter()
                        value = next(result)
                        toc = time.perf_counter()
                        if timeit:
                            if name in block.times:
                                block.times[name] += toc-tic
                            else:
                                block.times[name] = toc-tic
                        yield value
                    except StopIteration:
                        break
            return generator_wrapper()

        return wrapper
    return decorator

block.times = {}