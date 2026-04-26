#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-25
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : analysis
"""
This function is a decorator to be used on block function, which allow to use config for default values.
"""

import functools
import inspect
import time
from contextlib import nullcontext

from smlmlp import metadatum

def analysis(timeit=True, df_name='detections'):
    """
    Decorator for analysis functions to read values from a Locs object.

    Parameters
    ----------
    timeit : bool, optional
        Whether to record execution time when a Locs object is provided.
    df_name : str, optional
        Name of the Locs dataframe used to fill function arguments.

    Returns
    -------
    function
        Decorator for an analysis function.

    Examples
    --------
    >>> from smlmlp import analysis
    >>> @analysis(df_name="points")
    ... def scaled(x, *, scale=1.0, cuda=False, parallel=False):
    ...     return x * scale, cuda, parallel, {}
    """

    def decorator(function):
        """Create the decorated callable."""
        name = function.__name__
        @functools.wraps(function)
        def wrapper(*args, locs=None, df_name=df_name, **kwargs):
            """Call the wrapped function with normalized arguments."""
            kwargs = {key: value for key, value in kwargs.items() if value is not None}
            if locs is not None:
                config = locs.config
                df = getattr(locs, df_name)
                signature = inspect.signature(function)
                kw = {}
                for pname, param in signature.parameters.items():
                    attr = kwargs.pop(pname, None)
                    if (param.kind is inspect.Parameter.KEYWORD_ONLY or param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD) and hasattr(df, pname):
                        attr = getattr(df, pname) if attr is None else attr
                        kw[pname] = attr
                    elif (param.kind is inspect.Parameter.KEYWORD_ONLY or param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD) and hasattr(config, pname):
                        attr = getattr(config, pname) if attr is None else attr
                        if any(pname == datum for group in metadatum.groups.values() for datum in group):
                            setattr(config, pname, attr)
                        kw[pname] = attr
                kw.update(kwargs)
                kwargs = kw
                printer = locs.printer
                timeit = printer.timeit(f"applying {name} analysis") if printer is not None else nullcontext()
            else:
                timeit = nullcontext()

            with timeit:
                tic = time.perf_counter()
                result = function(*args, **kwargs)
                toc = time.perf_counter()

            if not inspect.isgenerator(result):
                if timeit and locs is not None:
                    if name in locs.times:
                        locs.times[name] += toc-tic
                    else:
                        locs.times[name] = toc-tic
                return result

            def generator_wrapper():
                """Record execution time for generator steps."""
                while True:
                    try:
                        tic = time.perf_counter()
                        value = next(result)
                        toc = time.perf_counter()
                        if timeit and locs is not None:
                            if name in locs.times:
                                locs.times[name] += toc-tic
                            else:
                                locs.times[name] = toc-tic
                        yield value
                    except StopIteration:
                        break
            return generator_wrapper()

        return wrapper
    return decorator