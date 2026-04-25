#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

from smlmlp import block



@block()
def block_template(channels, /, *, cuda=False, parallel=False):
    """Block template placeholder.

    Parameters
    ----------
    channels : sequence
        Input channels.
    cuda : bool, optional
        Whether to use CUDA execution.
    parallel : bool, optional
        Whether to use parallel execution.

    Returns
    -------
    tuple
        A tuple whose last item is an ``info`` dictionary.

    Raises
    ------
    SyntaxError
        Always raised because this template is not implemented.
    """
    raise SyntaxError("Not implemented yet.")
