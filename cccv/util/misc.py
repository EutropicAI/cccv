import math
import random
from math import exp
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def set_random_seed(seed: int = 0) -> None:
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def resize(img: Tensor, _scale: float) -> Tensor:
    _, _, _h, _w = img.shape
    while _h * _scale % 64 != 0:
        _h += 1
    while _w * _scale % 64 != 0:
        _w += 1
    return F.interpolate(img, size=(int(_h), int(_w)), mode="bilinear", align_corners=False)


def de_resize(img: Any, ori_h: int, ori_w: int) -> Tensor:
    return F.interpolate(img, size=(int(ori_h), int(ori_w)), mode="bilinear", align_corners=False)

    # Flow distance calculator


def distance_calculator(_x: Tensor) -> Tensor:
    dtype = _x.dtype
    u, v = _x[:, 0:1].float(), _x[:, 1:].float()
    return torch.sqrt(u**2 + v**2).to(dtype)


class TMapper:
    def __init__(self, src: float = -1.0, dst: float = 0.0, times: float = -1):
        self.times = dst / src if times == -1 else times
        self.now_step = -1
        self.src = src
        self.dst = dst

    def get_range_timestamps(
        self, _min: float, _max: float, lclose: bool = True, rclose: bool = False, normalize: bool = True
    ) -> list:
        _min_step = math.ceil(_min * self.times)
        _max_step = math.ceil(_max * self.times)
        _start = _min_step if lclose else _min_step + 1
        _end = _max_step if not rclose else _max_step + 1
        if _start >= _end:
            return []
        if normalize:
            return [((_i / self.times) - _min) / (_max - _min) for _i in range(_start, _end)]
        return [_i / self.times for _i in range(_start, _end)]


def gaussian(window_size: int, sigma: float) -> Tensor:
    gauss = Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window_3d(window_size: int, channel: int = 1) -> Tensor:
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
    window = _3D_window.expand(1, channel, window_size, window_size, window_size).contiguous()
    return window


def ssim_matlab(
    img1: Tensor,
    img2: Tensor,
    window_size: int = 11,
    window: Tensor = None,
    size_average: bool = True,
) -> Tensor:
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if torch.max(img1) > 128:
        max_val = 255
    else:
        max_val = 1

    if torch.min(img1) < -0.5:
        min_val = -1
    else:
        min_val = 0
    L = max_val - min_val

    padd = 0
    (_, _, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window_3d(real_size, channel=1).to(img1.device).to(img1.dtype)
        # Channel is set to 1 since we consider color images as volumetric images

    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)

    mu1 = F.conv3d(F.pad(img1, (5, 5, 5, 5, 5, 5), mode="replicate"), window, padding=padd, groups=1)
    mu2 = F.conv3d(F.pad(img2, (5, 5, 5, 5, 5, 5), mode="replicate"), window, padding=padd, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), "replicate"), window, padding=padd, groups=1) - mu1_sq
    sigma2_sq = F.conv3d(F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), "replicate"), window, padding=padd, groups=1) - mu2_sq
    sigma12 = F.conv3d(F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), "replicate"), window, padding=padd, groups=1) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return ret


def check_scene(x1: Tensor, x2: Tensor, enable_scdet: bool, scdet_threshold: float) -> bool:
    """
    Check if the scene is different, based on the SSIM value of the two input tensors.

    Input Tensor can be 3D, 4D, or 5D.

    :param x1: The first input tensor.
    :param x2: The second input tensor.
    :param enable_scdet: Whether to enable the scene change detection.
    :param scdet_threshold: The threshold of the SSIM value.
    """

    if not enable_scdet:
        return False
    if x1.dim() != x2.dim():
        raise ValueError("The dimensions of the two input tensors must be the same.")
    if x1.dim() not in [3, 4, 5]:
        raise ValueError("The input tensor must be 3D, 4D, or 5D.")

    _x1 = x1.clone()
    _x2 = x2.clone()

    if _x1.dim() == 3:
        _x1 = _x1.unsqueeze(0)
        _x2 = _x2.unsqueeze(0)

    if _x1.dim() == 5:
        _x1 = _x1.squeeze(0)
        _x2 = _x2.squeeze(0)

    _x1 = F.interpolate(_x1, (32, 32), mode="bilinear", align_corners=False)
    _x2 = F.interpolate(_x2, (32, 32), mode="bilinear", align_corners=False)

    return ssim_matlab(_x1, _x2).item() < scdet_threshold
