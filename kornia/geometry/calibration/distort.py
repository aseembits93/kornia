# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Optional

import torch
import torch.nn.functional as F


# Based on https://github.com/opencv/opencv/blob/master/modules/calib3d/src/distortion_model.hpp#L75
def tilt_projection(taux: torch.Tensor, tauy: torch.Tensor, return_inverse: bool = False) -> torch.Tensor:
    r"""Estimate the tilt projection matrix or the inverse tilt projection matrix.

    Args:
        taux: Rotation angle in radians around the :math:`x`-axis with shape :math:`(*, 1)`.
        tauy: Rotation angle in radians around the :math:`y`-axis with shape :math:`(*, 1)`.
        return_inverse: False to obtain the tilt projection matrix. True for the inverse matrix.

    Returns:
        torch.Tensor: Inverse tilt projection matrix with shape :math:`(*, 3, 3)`.

    """
    if taux.shape != tauy.shape:
        raise ValueError(f"Shape of taux {taux.shape} and tauy {tauy.shape} do not match.")

    ndim: int = taux.dim()
    taux = taux.reshape(-1)
    tauy = tauy.reshape(-1)

    cTx = torch.cos(taux)
    sTx = torch.sin(taux)
    cTy = torch.cos(tauy)
    sTy = torch.sin(tauy)
    zero = torch.zeros_like(cTx)
    one = torch.ones_like(cTx)

    Rx = torch.stack([one, zero, zero, zero, cTx, sTx, zero, -sTx, cTx], -1).reshape(-1, 3, 3)
    Ry = torch.stack([cTy, zero, -sTy, zero, one, zero, sTy, zero, cTy], -1).reshape(-1, 3, 3)
    R = Ry @ Rx

    if return_inverse:
        invR22 = 1 / R[..., 2, 2]
        invPz = torch.stack(
            [invR22, zero, R[..., 0, 2] * invR22, zero, invR22, R[..., 1, 2] * invR22, zero, zero, one], -1
        ).reshape(-1, 3, 3)

        inv_tilt = R.transpose(-1, -2) @ invPz
        if ndim == 0:
            inv_tilt = torch.squeeze(inv_tilt)

        return inv_tilt

    Pz = torch.stack(
        [R[..., 2, 2], zero, -R[..., 0, 2], zero, R[..., 2, 2], -R[..., 1, 2], zero, zero, one], -1
    ).reshape(-1, 3, 3)

    tilt = Pz @ R.transpose(-1, -2)
    if ndim == 0:
        tilt = torch.squeeze(tilt)

    return tilt


def distort_points(
    points: torch.Tensor, K: torch.Tensor, dist: torch.Tensor, new_K: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Distortion of a set of 2D points based on the lens distortion model.

    Radial :math:`(k_1, k_2, k_3, k_4, k_4, k_6)`,
    tangential :math:`(p_1, p_2)`, thin prism :math:`(s_1, s_2, s_3, s_4)`, and tilt :math:`(\tau_x, \tau_y)`
    distortion models are considered in this function.

    Args:
        points: Input image points with shape :math:`(*, N, 2)`.
        K: Intrinsic camera matrix with shape :math:`(*, 3, 3)`.
        dist: Distortion coefficients
            :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])`. This is
            a vector with 4, 5, 8, 12 or 14 elements with shape :math:`(*, n)`.
        new_K: Intrinsic camera matrix of the distorted image. By default, it is the same as K but you may additionally
            scale and shift the result by using a different matrix. Shape: :math:`(*, 3, 3)`. Default: None.

    Returns:
        Undistorted 2D points with shape :math:`(*, N, 2)`.

    Example:
        >>> points = torch.rand(1, 1, 2)
        >>> K = torch.eye(3)[None]
        >>> dist_coeff = torch.rand(1, 4)
        >>> points_dist = distort_points(points, K, dist_coeff)

    """
    if points.dim() < 2 and points.shape[-1] != 2:
        raise ValueError(f"points shape is invalid. Got {points.shape}.")

    if K.shape[-2:] != (3, 3):
        raise ValueError(f"K matrix shape is invalid. Got {K.shape}.")

    if new_K is None:
        new_K = K
    elif new_K.shape[-2:] != (3, 3):
        raise ValueError(f"new_K matrix shape is invalid. Got {new_K.shape}.")

    if dist.shape[-1] not in [4, 5, 8, 12, 14]:
        raise ValueError(f"Invalid number of distortion coefficients. Got {dist.shape[-1]}")

    # Adding torch.zeros to obtain vector with 14 coeffs.
    if dist.shape[-1] < 14:
        dist = F.pad(dist, [0, 14 - dist.shape[-1]])

    # Extract all distortion coefficients at once to avoid repeated slicing.
    k1 = dist[..., 0:1]
    k2 = dist[..., 1:2]
    p1 = dist[..., 2:3]
    p2 = dist[..., 3:4]
    k3 = dist[..., 4:5]
    k4 = dist[..., 5:6]
    k5 = dist[..., 6:7]
    k6 = dist[..., 7:8]
    s1 = dist[..., 8:9]
    s2 = dist[..., 9:10]
    s3 = dist[..., 10:11]
    s4 = dist[..., 11:12]

    # Convert 2D points from pixels to normalized camera coordinates
    new_fx = new_K[..., 0:1, 0]
    new_fy = new_K[..., 1:2, 1]
    new_cx = new_K[..., 0:1, 2]
    new_cy = new_K[..., 1:2, 2]

    x: torch.Tensor = (points[..., 0] - new_cx) / new_fx
    y: torch.Tensor = (points[..., 1] - new_cy) / new_fy

    # Distort points — Horner form for the radial polynomial to reduce ops
    r2 = x * x + y * y

    # Numerator: 1 + k1*r2 + k2*r4 + k3*r6 = 1 + r2*(k1 + r2*(k2 + k3*r2))
    num = torch.addcmul(k2, k3, r2).mul_(r2).add_(k1).mul_(r2).add_(1)
    # Denominator: 1 + k4*r2 + k5*r4 + k6*r6 = 1 + r2*(k4 + r2*(k5 + k6*r2))
    den = torch.addcmul(k5, k6, r2).mul_(r2).add_(k4).mul_(r2).add_(1)
    rad_poly = num / den

    # Tangential + thin prism
    xy = x * y
    xx = x * x
    yy = y * y
    xd = x * rad_poly + 2 * p1 * xy + p2 * (r2 + 2 * xx) + s1 * r2 + s2 * r2 * r2
    yd = y * rad_poly + p1 * (r2 + 2 * yy) + 2 * p2 * xy + s3 * r2 + s4 * r2 * r2

    # Compensate for tilt distortion
    if torch.any(dist[..., 12] != 0) or torch.any(dist[..., 13] != 0):
        tilt = tilt_projection(dist[..., 12], dist[..., 13])
        points_untilt = torch.stack([xd, yd, torch.ones_like(xd)], -1) @ tilt.transpose(-2, -1)
        xd = points_untilt[..., 0] / points_untilt[..., 2]
        yd = points_untilt[..., 1] / points_untilt[..., 2]

    # Convert points from normalized camera coordinates to pixel coordinates
    fx = K[..., 0:1, 0]
    fy = K[..., 1:2, 1]
    cx = K[..., 0:1, 2]
    cy = K[..., 1:2, 2]

    return torch.stack([fx * xd + cx, fy * yd + cy], -1)
