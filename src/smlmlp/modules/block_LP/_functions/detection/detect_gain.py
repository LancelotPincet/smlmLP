#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from arrlp import get_xp, sortloop
import bottleneck as bn
import numpy as np
from scipy.stats import linregress



# %% Function
@block()
def detect_gain(channels, /, nbins=50, *, cuda=False, parallel=False):
    """
    Estimate gain values and gain-related maps from image stacks.

    This function computes, for each channel, robust mean and variance maps
    from the temporal median and median absolute deviation. A gain map is then
    derived as the ratio between the mean and variance. In addition, binned
    mean-variance samples are built and a linear regression is fitted on the
    central intensity range in order to estimate a scalar gain value for each
    channel.

    Parameters
    ----------
    channels : sequence of ndarray
        Sequence of image stacks, one per channel. Each channel is expected to
        have shape ``(n_frames, height, width)``.
    nbins : int, optional
        Number of bins used to build the binned mean-variance relationship.
    cuda : bool, optional
        Whether to use GPU acceleration when supported.
    parallel : bool, optional
        Unused in this function. It is kept for API consistency.

    Returns
    -------
    tuple
        A tuple ``(gains, info)`` where:

        - ``gains`` is the list of scalar gain estimates, one per channel,
        - ``info`` is a dictionary containing reusable intermediate results.

        The dictionary contains the following keys:

        ``'gain'``
            List of gain maps computed as ``mean / var``.
        ``'mean'``
            List of robust mean maps computed from the temporal median.
        ``'var'``
            List of robust variance maps computed from the median absolute
            deviation.
        ``'mean_bins'``
            List of binned mean values used for the linear fit.
        ``'var_bins'``
            List of binned variance values used for the linear fit.
        ``'fit'``
            List of :func:`scipy.stats.linregress` results for each channel.

    Notes
    -----
    The robust variance estimate is computed from the median absolute
    deviation using:

    .. math::

        \\mathrm{var} = (1.4826 \\times \\mathrm{MAD})^2

    The scalar gain is then estimated from the slope of the linear regression
    fitted on the central third of the mean distribution:

    .. math::

        \\mathrm{gain} = \\frac{1}{\\mathrm{slope}}

    Examples
    --------
    Estimate the gain for one channel:

    >>> import numpy as np
    >>> channel = np.random.rand(20, 32, 32).astype(np.float32)
    >>> gains, info = detect_gain([channel], nbins=50)
    >>> len(gains)
    1
    >>> sorted(info.keys())
    ['fit', 'gain', 'mean', 'mean_bins', 'var', 'var_bins']

    Estimate the gain for multiple channels:

    >>> channels = [
    ...     np.random.rand(20, 32, 32).astype(np.float32),
    ...     np.random.rand(20, 32, 32).astype(np.float32),
    ... ]
    >>> gains, info = detect_gain(channels, nbins=40)
    >>> len(gains)
    2
    >>> len(info["gain"])
    2
    """
    # Select the array backend matching the requested execution mode.
    xp = get_xp(cuda)

    gain_maps = []
    mean_maps = []
    var_maps = []
    mean_bins = []
    var_bins = []
    gains = []
    fits = []

    for channel in channels:
        channel = xp.asarray(channel)

        # Compute robust mean and variance estimates from the temporal median
        # and median absolute deviation.
        med = xp.median(channel, axis=0) if cuda else bn.median(channel, axis=0)
        mad = (
            xp.median(xp.abs(channel - med), axis=0)
            if cuda else
            bn.median(xp.abs(channel - med), axis=0)
        )

        mean = med
        var = (1.4826 * mad) ** 2

        if cuda:
            mean = xp.asnumpy(mean)
            var = xp.asnumpy(var)

        gain_maps.append(mean / var)
        mean_maps.append(mean)
        var_maps.append(var)

        # Bin the mean-variance pairs according to the mean intensity.
        bins = np.linspace(mean.min(), mean.max(), nbins + 1)
        digitized = np.digitize(mean, bins)

        mean_binned = []
        var_binned = []

        for _, _, mean_m, var_m in sortloop(digitized, mean, var):
            if len(mean_m) > 10:
                mean_binned.append(mean_m.mean())
                var_binned.append(var_m.mean())

        mean_binned = np.array(mean_binned)
        var_binned = np.array(var_binned)

        mean_bins.append(mean_binned)
        var_bins.append(var_binned)

        # Fit the mean-variance relationship on the central intensity range.
        mask = np.logical_and(
            mean_binned < np.percentile(mean, 66),
            mean_binned > np.percentile(mean, 33),
        )

        fit = linregress(mean_binned[mask], var_binned[mask])
        gain = 1.0 / fit.slope

        gains.append(gain)
        fits.append(fit)

    info = {
        "gain": gain_maps,
        "mean": mean_maps,
        "var": var_maps,
        "mean_bins": mean_bins,
        "var_bins": var_bins,
        "fit": fits,
    }

    return gains, info