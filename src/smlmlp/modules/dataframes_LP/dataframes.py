#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-23
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : dataframes

"""
This is a dict of all the dataframes that are created.
"""



import importlib
from pathlib import Path

path = Path(__file__).parent / '_functions/dataframes'
names = [file.stem[1:] for file in path.iterdir() if file.suffix == '.py' and not file.name.startswith('__')]
dataframes = {name: getattr(importlib.import_module(f'._functions.dataframes._{name}', package=__package__), name) for name in names}


if __name__ == "__main__":
    from corelp import test

    test(__file__)
