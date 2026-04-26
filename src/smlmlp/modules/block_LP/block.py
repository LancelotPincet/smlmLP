#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-25
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : block

"""
Decorator for block functions to read default values from a config or Locs object.

A decorated function can receive keyword-only and positional-or-keyword parameters
from a config object passed via the ``config`` argument or from a Locs object
passed via the ``locs`` argument. Parameter resolution order:

1. kwargs (explicitly provided)
2. locs.{df_name} (if locs provided)
3. config (if locs provided: locs.config, else: provided config)
4. function default

If ``locs`` is provided, ``config`` must not also be provided.

Execution time is recorded in ``block.times`` (or ``locs.times`` if Locs provided).

This decorator works for both regular functions and generators.
"""

import functools
import inspect
import time

from smlmlp import metadatum


def block(timeit=True, df_name="detections"):
    """
    Decorator for block functions to read values from config or Locs objects.

    Parameter resolution order:
    1. kwargs (explicitly provided)
    2. locs.{df_name} (if locs provided)
    3. config (if locs provided: locs.config, else: provided config)
    4. function default

    If ``locs`` is provided, ``config`` must not also be provided.

    This decorator works for functions and generators. Computation time is
    added for each call in ``block.times`` (or ``locs.times`` if locs provided).

    Parameters
    ----------
    timeit : bool, optional
        Whether to record execution time. Default is ``True``.
    df_name : str, optional
        Name of the Locs dataframe used to fill function arguments. Default is
        ``'detections'``.

    Returns
    -------
    function
        Decorated function.

    Examples
    --------
    >>> from smlmlp import block
    >>> @block()
    ... def scaled_value(value, /, scale=1, *, offset=0, cuda=False, parallel=False):
    ...     return value * scale + offset, {"scale": scale}
    >>> result, info = scaled_value(4, config=type("Config", (), {"scale": 3, "offset": 2})())
    >>> result
    14
    """
    def decorator(function):
        """Create the decorated callable."""
        name = function.__name__

        @functools.wraps(function)
        def wrapper(*args, locs=None, config=None, df_name=df_name, **kwargs):
            """Call the wrapped function with normalized arguments."""
            if locs is not None and config is not None:
                raise ValueError("Cannot specify both locs and config simultaneously.")

            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            if locs is not None:
                df = getattr(locs, df_name)
                config = locs.config
            elif config is not None:
                df = None
            else:
                df = None
                config = None

            if df is not None or config is not None:
                signature = inspect.signature(function)
                kw = {}
                for pname, param in signature.parameters.items():
                    if param.kind not in (
                        inspect.Parameter.KEYWORD_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    ):
                        continue
                    if pname in kwargs:
                        continue
                    if df is not None and hasattr(df, pname):
                        kw[pname] = getattr(df, pname)
                    elif config is not None and hasattr(config, pname):
                        attr = getattr(config, pname)
                        if any(
                            pname == datum
                            for group in metadatum.groups.values()
                            for datum in group
                        ):
                            setattr(config, pname, attr)
                        kw[pname] = attr
                kw.update(kwargs)
                kwargs = kw

            tic = time.perf_counter()
            result = function(*args, **kwargs)
            toc = time.perf_counter()

            if not inspect.isgenerator(result):
                if timeit:
                    if locs is not None:
                        locs.times[name] = locs.times.get(name, 0) + toc - tic
                    else:
                        block.times[name] = block.times.get(name, 0) + toc - tic
                return result

            def generator_wrapper():
                """Yield values while measuring generator execution time."""
                while True:
                    try:
                        tic = time.perf_counter()
                        value = next(result)
                        toc = time.perf_counter()
                        if timeit:
                            if locs is not None:
                                locs.times[name] = locs.times.get(name, 0) + toc - tic
                            else:
                                block.times[name] = block.times.get(name, 0) + toc - tic
                        yield value
                    except StopIteration:
                        break

            return generator_wrapper()

        return wrapper

    return decorator


block.times = {}
