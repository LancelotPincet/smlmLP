#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-08-28
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

"""
Adds a dependency to smlmLP
"""



# %% Libraries
from devlp import path
import subprocess



# %% Main function
def main() :
    print('\nRunning add_dependency :')

    todefine = True
    while todefine :
        name = input('     New dependency to add to smlmLP ? >>> ')
        ok = input(f'     "{name}" will be added to smlmLP ? [y]/n >>> ')
        if ok.lower() in ["", "y", "yes", "true"] :
            todefine = False
    subprocess.run(["uv", "add", name], cwd=path.parent / 'libsLP/smlmLP', check=True, stdout=subprocess.PIPE)

    # End
    print('add_dependency finished!\n')



# %% Main function run
if __name__ == "__main__":
    main()