#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-04-27
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : start_server

"""
This function starts a server with a semaphore for managing (V)RAM usage.
"""



# %% Libraries
import time
import psutil
from smlmlp import ResourceServer, computer



# %% Function
def start_server(ram_factor=0.9, vram_factor=0.9) :
    '''
    This function starts a server with a semaphore for managing (V)RAM usage.
    
    Examples
    --------
    >>> if __name__ == "__main__":
    ...     from smlmlp import start_server
    ...     start_server()
    '''

    servers = [
    ResourceServer(
        sock_path="/tmp/ramtokens.sock",
        total_tokens= computer.ram.total() * ram_factor,
        admin_key="ad-key",
        log_prefix="[RAM tokens serv]",
        ),
    ResourceServer(
        sock_path="/tmp/vramtokens.sock",
        total_tokens=computer.vram.total() * vram_factor,
        admin_key="ad-key",
        log_prefix="[VRAM tokens serv]",
        ),
    ]

    for server in servers:
        server.start()

    print("All resource servers started.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for server in servers:
            server.stop()



# %% Test function run
if __name__ == "__main__":
    testing = False
    if testing :
        from corelp import test
        test(__file__)
    else :
        start_server()