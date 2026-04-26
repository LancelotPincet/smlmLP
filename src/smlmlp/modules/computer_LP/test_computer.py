#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-03-03
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : computer
"""
This file allows to test computer

computer : This object gives the various computer current parameters.
"""

from smlmlp import computer


# %% test RAM


def test_ram_probes_return_positive_values():
    """System RAM probes return non-negative values."""
    assert computer.ram.total() >= 0
    assert computer.ram.free() >= 0
    assert computer.ram.used() >= 0


def test_ram_total_greater_than_free():
    """Total RAM should be greater than or equal to free RAM."""
    assert computer.ram.total() >= computer.ram.free()


# %% test GPU


def test_gpu_cuda_returns_boolean():
    """GPU CUDA check returns a boolean."""
    assert isinstance(computer.gpu.cuda(), bool)


# %% test CPU


def test_cpu_cores_returns_positive_integer():
    """CPU cores returns a positive integer."""
    assert computer.cpu.cores() > 0


# %% test Disc


def test_disc_probes_return_positive_values():
    """Disk usage probes return non-negative values."""
    assert computer.disc.total() >= 0
    assert computer.disc.free() >= 0
    assert computer.disc.used() >= 0


def test_disc_total_greater_than_free():
    """Total disk space should be greater than or equal to free space."""
    assert computer.disc.total() >= computer.disc.free()


# %% test Computer container


def test_computer_contains_all_resource_probes():
    """Computer object has all expected resource monitors."""
    assert hasattr(computer, "ram")
    assert hasattr(computer, "vram")
    assert hasattr(computer, "cpu")
    assert hasattr(computer, "gpu")
    assert hasattr(computer, "disc")


if __name__ == "__main__":
    from corelp import test

    test(__file__)