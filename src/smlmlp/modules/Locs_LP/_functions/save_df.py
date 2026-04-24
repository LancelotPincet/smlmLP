#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



import csv
from contextlib import nullcontext



def save_df(df, path, head2save=None, printer=None) :
    timeit = nullcontext() if printer is None else printer.timeit(f"saving {df.__class__.__name__} into {path}")

    with timeit :
        #Add quotes
        columns = list(df.columns)
        for pos,col in enumerate(columns) :
            columns[pos] = '"' + col + '"'
        df.columns = columns

        #Save df
        if head2save is not None :
            columns = [col for col in df.columns if col[1:-1] in head2save]
        df.to_csv(path, columns=columns, quoting = csv.QUOTE_NONE, float_format="%.3f")

        #Substract quotes
        columns = list(df.columns)
        for pos,col in enumerate(columns) :
            columns[pos] = col[1:-1]
        df.columns = columns

