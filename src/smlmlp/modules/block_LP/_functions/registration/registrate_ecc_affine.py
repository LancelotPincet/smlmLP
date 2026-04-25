#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

from smlmlp import block
from arrlp import get_xp, transform_parameters
import cv2
import numpy as np



@block()
def registrate_ecc_affine(
    optimized,
    /,
    ref_pix=1.0,
    *,
    cuda=False,
    parallel=False,
):
    """Estimate redundant pairwise affine transforms with ECC registration.

    This function computes an affine transform between every pair of optimized
    channel images with OpenCV's enhanced correlation coefficient registration.
    The OpenCV warp is converted to the row/column convention used by
    ``scipy.ndimage.affine_transform`` and ``arrlp.img_transform`` before being
    decomposed into transform parameters.

    Parameters
    ----------
    optimized : sequence of ndarray
        Sequence of optimized registration images.
    ref_pix : float or tuple of float, optional
        Reference pixel size used to convert fitted parameters.
    cuda : bool, optional
        Whether to use CUDA execution.
    parallel : bool, optional
        Whether to use parallel execution.

    Returns
    -------
    tuple
        A tuple ``(shiftx, shifty, angle, shearx, sheary, scalex, scaley,
        info)`` where each list contains one value per channel pair. Shifts are
        returned in the units defined by ``ref_pix``.

        The dictionary contains the following keys:

        ``'ref_pix'``
            Reference pixel size used for shift conversion.
        ``'shape'``
            Optimized image shape used when extracting affine parameters.
        ``'pairs'``
            List of channel index pairs corresponding to the outputs.
        ``'matrices'``
            Pairwise affine matrices in ``arrlp.img_transform`` convention.
        ``'warp_matrices'``
            Pairwise affine matrices in OpenCV x/y convention.
        ``'ecc'``
            Enhanced correlation coefficient values returned by OpenCV.
    """
    # Select the array backend matching the requested execution mode.
    xp = get_xp(cuda)

    # Normalize the reference pixel size to a (y, x) pair.
    try:
        if len(ref_pix) != 2:
            raise ValueError("ref_pix does not have 2 values (y, x)")
    except TypeError:
        ref_pix = (ref_pix, ref_pix)

    shape = optimized[0].shape if len(optimized) > 0 else None

    shiftx = []
    shifty = []
    angle = []
    shearx = []
    sheary = []
    scalex = []
    scaley = []
    pairs = []
    matrices = []
    warp_matrices = []
    ecc = []

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        1000,
        1e-7,
    )

    for i in range(len(optimized)):
        template = _as_cv2_image(optimized[i], xp=xp, cuda=cuda)

        for j in range(i + 1, len(optimized)):
            image = _as_cv2_image(optimized[j], xp=xp, cuda=cuda)

            if image.shape != template.shape:
                raise ValueError("optimized images must have the same shape")

            warp_matrix = np.eye(2, 3, dtype=np.float32)
            cc, warp_matrix = cv2.findTransformECC(
                template,
                image,
                warp_matrix,
                motionType=cv2.MOTION_AFFINE,
                criteria=criteria,
            )

            matrix = _cv2_to_ndimage_matrix(warp_matrix)
            dx, dy, da, dsx, dsy, scx, scy = _matrix_to_parameters(
                matrix,
                shape,
                ref_pix,
            )

            shiftx.append(dx)
            shifty.append(dy)
            angle.append(da)
            shearx.append(dsx)
            sheary.append(dsy)
            scalex.append(scx)
            scaley.append(scy)
            pairs.append((i, j))
            matrices.append(matrix)
            warp_matrices.append(warp_matrix.copy())
            ecc.append(cc)

    info = {
        "ref_pix": ref_pix,
        "shape": shape,
        "pairs": pairs,
        "matrices": matrices,
        "warp_matrices": warp_matrices,
        "ecc": ecc,
    }

    return shiftx, shifty, angle, shearx, sheary, scalex, scaley, info



def _as_cv2_image(image, *, xp, cuda=False):
    """Convert an optimized image to a contiguous float32 OpenCV image."""
    if cuda:
        image = xp.asnumpy(image)

    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError("optimized images must be two-dimensional")

    return np.ascontiguousarray(image, dtype=np.float32)



def _cv2_to_ndimage_matrix(warp_matrix):
    """Convert an OpenCV x/y affine warp to the ndimage y/x convention."""
    warp_matrix = np.asarray(warp_matrix, dtype=float)

    matrix = np.eye(3, dtype=float)
    matrix[0, 0] = warp_matrix[1, 1]
    matrix[0, 1] = warp_matrix[1, 0]
    matrix[0, 2] = warp_matrix[1, 2]
    matrix[1, 0] = warp_matrix[0, 1]
    matrix[1, 1] = warp_matrix[0, 0]
    matrix[1, 2] = warp_matrix[0, 2]

    return matrix



def _matrix_to_parameters(matrix, shape, ref_pix):
    """Recover transform parameters from an affine matrix."""
    rotation = _polar_rotation_angle(matrix[:2, :2])
    dx, dy, dsx, dsy, da, scx, scy = transform_parameters(
        matrix,
        shape,
        angle=rotation,
    )

    return dx * ref_pix[1], dy * ref_pix[0], da, dsx, dsy, scx, scy



def _polar_rotation_angle(linear):
    """Return the closest proper-rotation angle for a 2x2 affine matrix."""
    u, _, vt = np.linalg.svd(linear)
    rotation = u @ vt

    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    theta = np.arctan2(rotation[1, 0], rotation[0, 0])
    return -np.degrees(theta)
