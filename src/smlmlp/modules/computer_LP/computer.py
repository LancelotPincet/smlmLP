#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-03-03
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : computer

"""
System resource monitoring for RAM, VRAM, CPU, GPU, and disk.

This module provides classes to query various system resources including
physical RAM, CUDA VRAM, CPU cores, GPU availability, and disk usage.
"""

import shutil
import subprocess

import numba.cuda
import psutil

from arrlp import gc


class Computer:
    """
    Container for system resource monitoring.

    Provides access to RAM, VRAM, CPU, GPU, and disk usage information.

    Attributes
    ----------
    ram : RAM
        System RAM probes.
    vram : VRAM
        CUDA VRAM probes.
    cpu : CPU
        CPU information.
    gpu : GPU
        GPU availability.
    disc : Disc
        Disk usage probes.

    Examples
    --------
    >>> from smlmlp import computer
    >>> computer.ram.total()  # doctest: +ELLIPSIS
    8.0...
    """

    def __init__(self):
        """Initialize the computer object with resource monitors."""
        self.ram = RAM()
        self.vram = VRAM()
        self.cpu = CPU()
        self.gpu = GPU()
        self.disc = Disc()


class RAM:
    """
    System RAM probes, returned in GiB.

    Examples
    --------
    >>> ram = RAM()
    >>> ram.total()  # doctest: +ELLIPSIS
    8.0...
    >>> ram.free()  # doctest: +ELLIPSIS
    4.0...
    """

    def total(self):
        """Return total system RAM in GiB."""
        gc()
        return psutil.virtual_memory().total / (1024**3)

    def free(self):
        """Return available system RAM in GiB."""
        gc()
        return psutil.virtual_memory().available / (1024**3)

    def used(self):
        """Return used system RAM in GiB."""
        gc()
        return psutil.virtual_memory().used / (1024**3)


class VRAM:
    """
    CUDA VRAM probes, returned in GB from nvidia-smi values.

    Examples
    --------
    >>> vram = VRAM()
    >>> vram.total()  # Returns 0 if CUDA unavailable
    0
    """

    def total(self):
        """Return total VRAM in GB, or 0 when CUDA is unavailable."""
        if not numba.cuda.is_available():
            return 0
        gc()
        info = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip().split("\n")[0].split(", ")
        return float(info[1].split(" ")[0]) / 1000

    def free(self):
        """Return free VRAM in GB, or 0 when CUDA is unavailable."""
        if not numba.cuda.is_available():
            return 0
        gc()
        info = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip().split("\n")[0].split(", ")
        return float(info[2].split(" ")[0]) / 1000

    def used(self):
        """Return used VRAM in GB, or 0 when CUDA is unavailable."""
        if not numba.cuda.is_available():
            return 0
        gc()
        info = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip().split("\n")[0].split(", ")
        return float(info[3].split(" ")[0]) / 1000


class CPU:
    """
    CPU probes.

    Examples
    --------
    >>> cpu = CPU()
    >>> cpu.cores()  # Returns number of logical cores
    8
    """

    def cores(self):
        """Return the number of logical CPU cores."""
        return psutil.cpu_count(logical=True)


class GPU:
    """
    GPU probes.

    Examples
    --------
    >>> gpu = GPU()
    >>> gpu.cuda()  # Returns whether CUDA is available
    False
    """

    def cuda(self):
        """Return whether CUDA is available through numba."""
        return numba.cuda.is_available()


class Disc:
    """
    Root filesystem usage probes, returned in GiB.

    Examples
    --------
    >>> disc = Disc()
    >>> disc.total()  # doctest: +ELLIPSIS
    100.0...
    """

    def total(self):
        """Return total root filesystem size in GiB."""
        return shutil.disk_usage("/").total / (1024**3)

    def free(self):
        """Return free root filesystem size in GiB."""
        return shutil.disk_usage("/").free / (1024**3)

    def used(self):
        """Return used root filesystem size in GiB."""
        return shutil.disk_usage("/").used / (1024**3)


computer = Computer()


if __name__ == "__main__":
    from corelp import test

    test(__file__)