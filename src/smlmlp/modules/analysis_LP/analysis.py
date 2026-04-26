#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-25
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : analysis

"""
Decorator for analysis functions to read default values from a Locs or config object.

A decorated function receives parameters from a Locs object passed via the ``locs``
argument, or a config object passed via ``config``. Parameter resolution order:

1. kwargs (explicitly provided)
2. locs.{df_name} (if locs provided)
3. config (if locs provided: locs.config, else: provided config)
4. function default

If ``locs`` is provided, ``config`` must not also be provided.

Execution time is recorded in ``locs.times`` (or ``block.times`` if no Locs provided).

This decorator works for both regular functions and generators.
"""

import functools
import inspect
import time
from contextlib import nullcontext

from smlmlp import metadatum


def analysis(timeit=True, df_name="detections"):
    """
    Decorator for analysis functions to read values from Locs or config objects.

    Parameter resolution order:
    1. kwargs (explicitly provided)
    2. locs.{df_name} (if locs provided)
    3. config (if locs provided: locs.config, else: provided config)
    4. function default

    If ``locs`` is provided, ``config`` must not also be provided.

    This decorator works for functions and generators. Computation time is
    added for each call.

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
        Decorator for an analysis function.

    Examples
    --------
    >>> from smlmlp import analysis
    >>> @analysis(df_name="points")
    ... def scaled(x, *, scale=1.0, cuda=False, parallel=False):
    ...     return x * scale, cuda, parallel, {}
    >>> import numpy as np
    >>> from types import SimpleNamespace
    >>> locs = SimpleNamespace(
    ...     points=SimpleNamespace(x=np.array([1.0, 2.0])),
    ...     config=SimpleNamespace(scale=3.0),
    ...     printer=None,
    ...     times={},
    ... )
    >>> values, cuda, parallel, info = scaled(locs=locs)
    >>> list(values)
    [3.0, 6.0]
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
                printer = locs.printer
            elif config is not None:
                df = None
                printer = None
            else:
                df = None
                config = None
                printer = None

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

            timeit_printer = (
                printer.timeit(f"applying {name} analysis")
                if printer is not None
                else nullcontext()
            )

            with timeit_printer:
                tic = time.perf_counter()
                result = function(*args, **kwargs)
                toc = time.perf_counter()

            if not inspect.isgenerator(result):
                if timeit:
                    if locs is not None:
                        locs.times[name] = locs.times.get(name, 0) + toc - tic
                    else:
                        from smlmlp.modules.block_LP.block import block
                        block.times[name] = block.times.get(name, 0) + toc - tic
                return result

            def generator_wrapper():
                """Record execution time for generator steps."""
                while True:
                    try:
                        tic = time.perf_counter()
                        value = next(result)
                        toc = time.perf_counter()
                        if timeit:
                            if locs is not None:
                                locs.times[name] = locs.times.get(name, 0) + toc - tic
                            else:
                                from smlmlp.modules.block_LP.block import block
                                block.times[name] = block.times.get(name, 0) + toc - tic
                        yield value
                    except StopIteration:
                        break

            return generator_wrapper()

        return wrapper

    return decorator
