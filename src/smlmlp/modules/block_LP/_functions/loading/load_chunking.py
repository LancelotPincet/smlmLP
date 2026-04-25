#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, computer
import tifffile as tiff
import numpy as np
from corelp import Path



# %% Constants
_MIN_FRAME_SIZE_GB = 1e-12
_MEMORY_COPY_FACTOR = 5



# %% Function
@block(timeit=False)
def load_chunking(*tif_paths, cuda=False, parallel=False):
    """Estimate a safe frame chunk size from RAM/VRAM availability.

    The chunk size is estimated from the size of one frame per TIFF file,
    converted to ``float32``. The result is constrained by free CPU memory and,
    when ``cuda=True``, also by free GPU memory.

    Parameters
    ----------
    *tif_paths : str
        Paths to the TIFF files composing the acquisition.
    cuda : bool, optional
        Whether to include GPU memory constraints in the chunk estimation.
    parallel : bool or int, optional
        CPU parallel loading setting used to reserve per-core input buffers.
        The interpretation is:

        - ``False`` or ``0``: use one core,
        - negative value: use all available worker cores,
        - positive value: use that exact number of worker cores.

    Returns
    -------
    tuple
        A tuple ``(loaded_max, info)`` where:

        - ``loaded_max`` is the estimated maximum number of frames that can be
          loaded in one chunk,
        - ``info`` is a dictionary containing reusable intermediate values.

        The dictionary contains the following keys:

        ``'ram_total_gb'``
            Total system RAM, in gigabytes.
        ``'ram_free_gb'``
            Currently available RAM, in gigabytes.
        ``'vram_total_gb'``
            Total GPU VRAM, in gigabytes.
        ``'vram_free_gb'``
            Currently available GPU VRAM, in gigabytes.
        ``'frame_sizes_gb'``
            Combined size of one frame per TIFF file after conversion to
            ``float32``, in gigabytes.
        ``'loaded_max_cpu'``
            Maximum chunk length constrained by free RAM.
        ``'loaded_max_gpu'``
            Maximum chunk length constrained by free VRAM (or ``np.inf`` when
            ``cuda=False``).

    Raises
    ------
    SyntaxError
        If no TIFF path is provided.
    RuntimeError
        If ``cuda=True`` is requested while CUDA is not available.
    ValueError
        If the requested positive ``parallel`` value exceeds available worker
        cores.

    Notes
    -----
    The memory heuristic keeps the original logic and assumes six float32
    copies per frame (raw, copy, background, signal, buffer and one extra slot).
    GPU memory do not count for raw and CPU memory also takes parallel frames
    into account.

    Examples
    --------
    Estimate a chunk size from a temporary TIFF stack:

    >>> import numpy as np
    >>> import tifffile as tiff
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "ROI.tif"
    ...     tiff.imwrite(path, np.zeros((20, 32, 32), dtype=np.uint16))
    ...     loaded_max, info = load_chunking(str(path), cuda=False, parallel=False)
    ...     loaded_max >= 0 and "loaded_max_cpu" in info
    True

    Use all available worker cores for the CPU estimate:

    >>> with TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "ROI.tif"
    ...     tiff.imwrite(path, np.zeros((20, 32, 32), dtype=np.uint16))
    ...     loaded_max, info = load_chunking(str(path), parallel=-1)
    ...     isinstance(info["loaded_max_cpu"], int)
    True
    """

    if len(tif_paths) < 1:
        raise SyntaxError("Must define at least one tiff file to load")

    cuda_available = computer.gpu.cuda()
    if cuda and not cuda_available:
        raise RuntimeError("cuda=True was requested but CUDA is not available")

    cpu_cores_limit = max(computer.cpu.cores() - 1, 1)
    ncores = _resolve_ncores(parallel, cpu_cores_limit)

    ram_total_gb = computer.ram.total()
    ram_free_gb = computer.ram.free()
    vram_total_gb = computer.vram.total()
    vram_free_gb = computer.vram.free()

    frame_sizes_gb = _estimate_frame_size_gb(tif_paths)
    parallel_buffers_gb = ncores * frame_sizes_gb
    loaded_max_cpu = int(
        (ram_free_gb - parallel_buffers_gb)
        // (frame_sizes_gb * (_MEMORY_COPY_FACTOR + 1))
    )

    loaded_max_gpu = (
        int(vram_free_gb // (frame_sizes_gb * _MEMORY_COPY_FACTOR))
        if cuda
        else np.inf
    )

    loaded_max = min(loaded_max_cpu, loaded_max_gpu)
    info = dict(
        ram_total_gb=ram_total_gb,
        ram_free_gb=ram_free_gb,
        vram_total_gb=vram_total_gb,
        vram_free_gb=vram_free_gb,
        frame_sizes_gb=frame_sizes_gb,
        loaded_max_cpu=loaded_max_cpu,
        loaded_max_gpu=loaded_max_gpu,
    )

    return loaded_max, info



def _resolve_ncores(parallel, cpu_cores_limit):
    """Resolve the number of worker cores from ``parallel``."""
    if not parallel:
        return 1

    parallel_int = int(parallel)
    if parallel_int < 0:
        return cpu_cores_limit

    if parallel_int > cpu_cores_limit:
        raise ValueError("Number of cores asked in parallel are too big")

    return max(parallel_int, 1)



def _estimate_frame_size_gb(tif_paths):
    """Return total first-frame size across files as float32 gigabytes."""
    frame_sizes_gb = 0.0
    for tif_path in tif_paths:
        frame = tiff.imread(Path(tif_path), key=0)
        frame_sizes_gb += np.asarray(frame, dtype=np.float32).nbytes / 1024**3

    return max(float(frame_sizes_gb), _MIN_FRAME_SIZE_GB)
