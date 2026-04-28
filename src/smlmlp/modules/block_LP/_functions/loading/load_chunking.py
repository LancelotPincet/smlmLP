#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import block, computer
import numpy as np



# %% Constants
_MEMORY_COPY_FACTOR = 5
_MEMORY_AUTOCORR_FACTOR = 20



# %% Function
@block(timeit=False)
def load_chunking(channels_frames_sizes_gb, /, chunks=None, pad=0, *,
                  ram_tokens=None, vram_tokens=None, cuda=False, parallel=False):
    """
    Estimate memory tokens and the effective TIFF loading chunk size.

    Parameters
    ----------
    channels_frames_sizes_gb : sequence of float
        Size in GB of one loaded frame for each TIFF input or channel. Values
        are summed for one complete multi-input frame, and the largest value is
        used for temporary single-input buffers.
    chunks : int or None, optional
        Requested number of acquisition frames per chunk. If ``None``, the
        largest safe chunk is inferred from currently free RAM and, when
        ``cuda=True``, currently free VRAM.
    pad : int, optional
        Number of temporal padding frames loaded on each side of every chunk.
    ram_tokens : float or None, optional
        Granted RAM token budget in GB. When provided, ``chunks`` is reduced so
        that the returned RAM estimate fits inside this grant.
    vram_tokens : float or None, optional
        Granted VRAM token budget in GB. This constrains ``chunks`` only when
        ``cuda=True``.
    cuda : bool, optional
        Whether VRAM limits are used to constrain chunk selection.
    parallel : bool or int, optional
        Worker-core setting used for the CPU copy estimate. ``False`` or ``0``
        uses one core, a negative integer uses all available worker cores, and a
        positive integer requests that many cores.

    Returns
    -------
    chunks : int
        Effective number of acquisition frames loaded per chunk.
    ram : float
        Estimated RAM requirement in GB for the effective chunk.
    vram : float
        Estimated VRAM requirement in GB for the effective chunk.
    ram_min : float
        Minimum RAM requirement in GB for one frame plus side padding.
    vram_min : float
        Minimum VRAM requirement in GB for one frame plus side padding.
    info : dict
        Diagnostic values used for the estimate, including memory limits,
        per-scenario chunk maxima, and per-scenario token estimates.

    Raises
    ------
    ValueError
        If frame sizes are missing or non-positive, ``pad`` is negative, or
        ``parallel`` requests more worker cores than are available.
    RuntimeError
        If ``cuda=True`` is requested while CUDA is unavailable.

    Notes
    -----
    1. Frame sizes are converted to floats, summed into one full-frame cost,
       and reduced to their maximum for temporary single-frame buffers.
    2. The worker count is resolved from ``parallel`` and CUDA availability is
       checked when VRAM constraints are requested.
    3. RAM and VRAM limits come from currently free memory, unless granted token
       budgets are supplied by ``ram_tokens`` or ``vram_tokens``.
    4. Four chunk limits are computed by inverting the CPU copy, GPU copy, CPU
       autocorrelation, and GPU autocorrelation memory formulas.
    5. Automatic chunking uses the smallest active limit. Fixed chunking keeps
       the requested value, unless token budgets are provided, in which case the
       requested value is reduced to fit the grants.
    6. The final chunk is clamped to at least one frame, then RAM and VRAM token
       estimates are recomputed for both the final chunk and the one-frame
       minimum reservation.

    Examples
    --------
    Estimate a fixed ten-frame chunk for two channels:

    >>> from smlmlp import load_chunking
    >>> result = load_chunking([0.001, 0.002], chunks=10, pad=1)
    >>> chunks, ram, vram, ram_min, vram_min, info = result
    >>> chunks
    10
    >>> round(ram, 3), round(vram, 3)
    (0.218, 0.18)
    >>> round(info["frame_sizes_gb"], 3), round(info["max_frame_sizes_gb"], 3)
    (0.003, 0.002)

    Recalculate a requested chunk after a smaller RAM grant:

    >>> chunks3, ram3, _, _, _, _ = load_chunking([0.001], chunks=3, pad=1)
    >>> chunks, _, _, _, _, _ = load_chunking([0.001], chunks=6, pad=1, ram_tokens=ram3)
    >>> chunks
    3

    Requesting too many worker cores raises an error:

    >>> load_chunking([0.001], chunks=1, parallel=10**9)
    Traceback (most recent call last):
    ...
    ValueError: Number of cores asked in parallel are too big
    """

    # Normalize inputs
    channels_frames_sizes_gb = [float(size) for size in channels_frames_sizes_gb]
    if len(channels_frames_sizes_gb) < 1: raise ValueError("Must define at least one frame size")
    if min(channels_frames_sizes_gb) <= 0: raise ValueError("Frame sizes must be positive")
    pad = int(pad)
    if pad < 0: raise ValueError("pad must be non-negative")
    if cuda and not computer.gpu.cuda(): raise RuntimeError("cuda=True was requested but CUDA is not available")

    # Read computer limits
    frame_sizes_gb = sum(channels_frames_sizes_gb)
    max_frame_sizes_gb = max(channels_frames_sizes_gb)
    cpu_cores_limit = max(computer.cpu.cores() - 1, 1)
    ncores = _resolve_ncores(parallel, cpu_cores_limit)
    ram_total_gb = computer.ram.total()
    ram_free_gb = computer.ram.free()
    vram_total_gb = computer.vram.total()
    vram_free_gb = computer.vram.free()
    ram_limit_gb = ram_free_gb if ram_tokens is None else float(ram_tokens)
    vram_limit_gb = vram_free_gb if vram_tokens is None else float(vram_tokens)

    # Compute active chunk limits
    ram_copy_fixed_gb = max_frame_sizes_gb * ncores
    ram_copy_frame_gb = frame_sizes_gb * (_MEMORY_COPY_FACTOR + 1)
    vram_copy_frame_gb = frame_sizes_gb * _MEMORY_COPY_FACTOR
    autocorr_fixed_gb = max_frame_sizes_gb * _MEMORY_AUTOCORR_FACTOR
    loaded_max_cpu_copy = np.floor((ram_limit_gb - ram_copy_fixed_gb) / ram_copy_frame_gb) - 2 * pad
    loaded_max_cpu_autocorr = np.floor(ram_limit_gb / (frame_sizes_gb + autocorr_fixed_gb)) - 2 * pad
    loaded_max_gpu_copy = np.floor(vram_limit_gb / vram_copy_frame_gb) - 2 * pad if cuda else np.inf
    loaded_max_gpu_autocorr = np.floor(vram_limit_gb / (frame_sizes_gb + autocorr_fixed_gb)) - 2 * pad if cuda else np.inf
    loaded_max = min(loaded_max_cpu_copy, loaded_max_gpu_copy, loaded_max_cpu_autocorr, loaded_max_gpu_autocorr)

    # Resolve final chunk size
    chunks_requested = None if chunks is None else int(chunks)
    chunks = loaded_max if chunks_requested is None else chunks_requested
    if chunks_requested is not None and (ram_tokens is not None or (cuda and vram_tokens is not None)):
        chunks = min(chunks_requested, loaded_max)
    chunks = max(int(chunks), 1)

    # Estimate tokens for final chunk
    loaded_frames = chunks + 2 * pad
    ram_cpu_copy = ram_copy_fixed_gb + ram_copy_frame_gb * loaded_frames
    vram_cpu_copy = vram_copy_frame_gb * loaded_frames
    ram_cpu_autocorr = (frame_sizes_gb + autocorr_fixed_gb) * loaded_frames
    vram_cpu_autocorr = (frame_sizes_gb + autocorr_fixed_gb) * loaded_frames

    # Estimate minimum tokens for one frame
    loaded_frames_min = 1 + 2 * pad
    ram_cpu_copy_min = ram_copy_fixed_gb + ram_copy_frame_gb * loaded_frames_min
    vram_cpu_copy_min = vram_copy_frame_gb * loaded_frames_min
    ram_cpu_autocorr_min = (frame_sizes_gb + autocorr_fixed_gb) * loaded_frames_min
    vram_cpu_autocorr_min = (frame_sizes_gb + autocorr_fixed_gb) * loaded_frames_min

    # Pack result
    ram = max(ram_cpu_copy, ram_cpu_autocorr)
    vram = max(vram_cpu_copy, vram_cpu_autocorr)
    ram_min = max(ram_cpu_copy_min, ram_cpu_autocorr_min)
    vram_min = max(vram_cpu_copy_min, vram_cpu_autocorr_min)
    info = dict(
        chunks=chunks, chunks_requested=chunks_requested, pad=pad, ncores=ncores,
        ram_total_gb=ram_total_gb, ram_free_gb=ram_free_gb, ram_limit_gb=ram_limit_gb,
        vram_total_gb=vram_total_gb, vram_free_gb=vram_free_gb, vram_limit_gb=vram_limit_gb,
        frame_sizes_gb=frame_sizes_gb, max_frame_sizes_gb=max_frame_sizes_gb,
        loaded_max_cpu_copy=loaded_max_cpu_copy, loaded_max_gpu_copy=loaded_max_gpu_copy,
        loaded_max_cpu_autocorr=loaded_max_cpu_autocorr, loaded_max_gpu_autocorr=loaded_max_gpu_autocorr,
        ram_cpu_copy=ram_cpu_copy, vram_cpu_copy=vram_cpu_copy,
        ram_cpu_autocorr=ram_cpu_autocorr, vram_cpu_autocorr=vram_cpu_autocorr,
        ram_cpu_copy_min=ram_cpu_copy_min, vram_cpu_copy_min=vram_cpu_copy_min,
        ram_cpu_autocorr_min=ram_cpu_autocorr_min, vram_cpu_autocorr_min=vram_cpu_autocorr_min,
    )

    return chunks, ram, vram, ram_min, vram_min, info



def _resolve_ncores(parallel, cpu_cores_limit):
    """Resolve the number of worker cores from ``parallel``."""
    if not parallel: return 1

    parallel_int = int(parallel)
    if parallel_int < 0: return cpu_cores_limit

    if parallel_int > cpu_cores_limit:
        raise ValueError("Number of cores asked in parallel are too big")

    return max(parallel_int, 1)



if __name__ == "__main__":
    from corelp import test
    test(__file__)
