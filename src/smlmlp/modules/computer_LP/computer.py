#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-03-03
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : computer

"""
This object gives the various computer current parameters.
"""



# %% Libraries
import psutil
import shutil
import numba.cuda
import subprocess



# %% Class
class Computer() :
    def __init__(self) :
        self.ram = RAM()
        self.vram = VRAM()
        self.cpu = CPU()
        self.gpu = GPU()
        self.disc = Disc()

class RAM() :
    def total(self) :
        return psutil.virtual_memory().total / (1024 ** 3)
    def free(self) :
        return psutil.virtual_memory().available / (1024 ** 3)
    def used(self) :
        return psutil.virtual_memory().used / (1024 ** 3)

class VRAM() :
    def total(self) :
        if not numba.cuda.is_available() : return 0
        info = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip().split("\n")[0].split(", ")
        return float(info[1].split(' ')[0]) / 1000 
    def free(self) :
        if not numba.cuda.is_available() : return 0
        info = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip().split("\n")[0].split(", ")
        return float(info[2].split(' ')[0]) / 1000
    def used(self) :
        if not numba.cuda.is_available() : return 0
        info = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip().split("\n")[0].split(", ")
        return float(info[3].split(' ')[0]) / 1000

class CPU() :
    def cores(self) :
        return psutil.cpu_count(logical=True)

class GPU() :
    def cuda(self) :
        return numba.cuda.is_available()

class Disc() :
    def total(self) :
        return shutil.disk_usage('/').total / (1024 ** 3)
    def free(self) :
        return shutil.disk_usage('/').free / (1024 ** 3)
    def used(self) :
        return shutil.disk_usage('/').used / (1024 ** 3)

computer = Computer()



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)