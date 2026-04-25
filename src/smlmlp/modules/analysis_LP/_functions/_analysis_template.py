#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import analysis


@analysis(df_name="detections")
def analysis_template(x, y, *, cuda=False, parallel=False) :
    """
    Placeholder for analysis template.

    Raises
    ------
    SyntaxError
        Always raised because this analysis is not implemented yet.
    """
    raise SyntaxError("Not implemented yet.")
