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
from types import SimpleNamespace

import smlmlp.modules.computer_LP.computer as computer_module


def test_computer_object_contains_resource_groups() :
    """The global computer object exposes resource groups."""
    assert isinstance(computer_module.computer.ram, computer_module.RAM)
    assert isinstance(computer_module.computer.vram, computer_module.VRAM)
    assert isinstance(computer_module.computer.cpu, computer_module.CPU)
    assert isinstance(computer_module.computer.gpu, computer_module.GPU)
    assert isinstance(computer_module.computer.disc, computer_module.Disc)


def test_ram_cpu_and_disc_use_system_probes(monkeypatch) :
    """RAM, CPU, and disk probes delegate to their system providers."""
    monkeypatch.setattr(computer_module, "gc", lambda: None)
    monkeypatch.setattr(
        computer_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=3 * 1024**3, available=2 * 1024**3, used=1024**3),
    )
    monkeypatch.setattr(computer_module.psutil, "cpu_count", lambda logical=True: 8)
    monkeypatch.setattr(
        computer_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=10 * 1024**3, free=6 * 1024**3, used=4 * 1024**3),
    )

    assert computer_module.RAM().total() == 3
    assert computer_module.RAM().free() == 2
    assert computer_module.RAM().used() == 1
    assert computer_module.CPU().cores() == 8
    assert computer_module.Disc().total() == 10
    assert computer_module.Disc().free() == 6
    assert computer_module.Disc().used() == 4


def test_gpu_and_vram_with_monkeypatched_cuda(monkeypatch) :
    """GPU probes use CUDA availability and nvidia-smi output."""
    output = "NVIDIA Test, 12000 MiB, 7000 MiB, 5000 MiB, 50 %, 40"

    monkeypatch.setattr(computer_module, "gc", lambda: None)
    monkeypatch.setattr(computer_module.numba.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        computer_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=output),
    )

    assert computer_module.GPU().cuda() is True
    assert computer_module.VRAM().total() == 12
    assert computer_module.VRAM().free() == 7
    assert computer_module.VRAM().used() == 5


def test_vram_returns_zero_without_cuda(monkeypatch) :
    """VRAM methods return zero when CUDA is unavailable."""
    monkeypatch.setattr(computer_module.numba.cuda, "is_available", lambda: False)

    assert computer_module.VRAM().total() == 0
    assert computer_module.VRAM().free() == 0
    assert computer_module.VRAM().used() == 0


if __name__ == "__main__":
    from corelp import test

    test(__file__)
