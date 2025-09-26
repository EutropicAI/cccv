# type: ignore
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cccv.arch import ARCH_REGISTRY
from cccv.arch.vfi.vfi_utils.warplayer import warp
from cccv.type import ArchType
from cccv.util.misc import distance_calculator


@ARCH_REGISTRY.register(name=ArchType.DRBA)
class DRBA(nn.Module):
    def __init__(self):
        super(DRBA, self).__init__()
        self.block0 = IFBlock(7 + 32, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 32, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 32, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + 32, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + 32, c=32)
        self.encode = Head()

        support_cupy = True
        try:
            import cupy

            if cupy.cuda.get_cuda_path() is None:
                support_cupy = False
        except Exception:
            support_cupy = False

        if support_cupy:
            from cccv.arch.vfi.vfi_utils.softsplat import softsplat as fwarp
        else:
            from cccv.arch.vfi.vfi_utils.softsplat_torch import softsplat as fwarp

        self.fwarp = fwarp

    def inference(self, x, timestep=0.5, scale_list=None, fastmode=True, ensemble=False, f0=None, f1=None):
        if scale_list is None:
            scale_list = [16, 8, 4, 2, 1]
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]
        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        f0 = self.encode(img0[:, :3]) if f0 is None else f0
        f1 = self.encode(img1[:, :3]) if f1 is None else f1
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        block = [self.block0, self.block1, self.block2, self.block3, self.block4]
        for i in range(5):
            if flow is None:
                flow, mask, feat = block[i](
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1), None, scale=scale_list[i]
                )
                if ensemble:
                    warnings.warn("[CCCV] ensemble is not supported since RIFEv4.21", stacklevel=2)
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, m0, feat = block[i](
                    torch.cat((warped_img0[:, :3], warped_img1[:, :3], wf0, wf1, timestep, mask, feat), 1),
                    flow,
                    scale=scale_list[i],
                )
                if ensemble:
                    warnings.warn("[CCCV] ensemble is not supported since RIFEv4.21", stacklevel=2)
                else:
                    mask = m0
                flow = flow + fd
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))
        mask = torch.sigmoid(mask)
        merged[4] = warped_img0 * mask + warped_img1 * (1 - mask)
        if not fastmode:
            warnings.warn("[CCCV] contextnet is removed", stacklevel=2)
            """
            c0 = self.contextnet(img0, flow[:, :2])
            c1 = self.contextnet(img1, flow[:, 2:4])
            tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
            res = tmp[:, :3] * 2 - 1
            merged[4] = torch.clamp(merged[4] + res, 0, 1)
            """
        return merged[4], flow_list

    def calc_flow(self, a, b, scale, f0=None, f1=None):
        scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        # calc flow at the lowest resolution (significantly faster with almost no quality loss).
        timestep = (a[:, :1].clone() * 0 + 1) * 0.5
        f0 = self.encode(a[:, :3]) if f0 is None else f0
        f1 = self.encode(b[:, :3]) if f1 is None else f1
        flow, _, _ = self.block0(torch.cat((a[:, :3], b[:, :3], f0, f1, timestep), 1), None, scale=scale_list[0])

        # get flow flow0.5 -> 0/1
        flow50, flow51 = flow[:, :2], flow[:, 2:]

        warp_method = "avg"

        # qvi
        # flow05, norm2 = fwarp(flow50, flow50)
        # flow05[norm2]...
        # flow05 = -flow05

        flow05 = -1 * self.fwarp(flow50, flow50, None, warp_method)
        flow15 = -1 * self.fwarp(flow51, flow51, None, warp_method)

        ones_mask = flow05.clone() * 0 + 1
        mask05 = self.fwarp(ones_mask, flow50, None, warp_method)
        mask15 = self.fwarp(ones_mask, flow51, None, warp_method)

        gap05 = mask05 < 0.999
        gap15 = mask15 < 0.999

        flow05[gap05] = (ones_mask * max(flow05.shape[2], flow05.shape[3]))[gap05]
        flow15[gap15] = (ones_mask * max(flow15.shape[2], flow15.shape[3]))[gap15]

        flow01 = flow05 * 2
        flow10 = flow15 * 2

        return flow01, flow10, f0, f1

    def forward(self, x, minus_t, zero_t, plus_t, _left_scene, _right_scene, _scale, _reuse=None):
        _I0, _I1, _I2 = x[:, 0], x[:, 1], x[:, 2]
        flow10, flow01, f1, f0 = self.calc_flow(_I1, _I0, _scale) if not _reuse else _reuse
        if _reuse is None:
            flow12, flow21, f1, f2 = self.calc_flow(_I1, _I2, _scale)
        else:
            flow12, flow21, f1, f2 = self.calc_flow(_I1, _I2, _scale, f0=_reuse[2])

        # Compute the distance using the optical flow and distance calculator
        d10 = distance_calculator(flow10) + 1e-4
        d12 = distance_calculator(flow12) + 1e-4

        # Calculate the distance ratio map
        drm10 = d10 / (d10 + d12)
        drm12 = d12 / (d10 + d12)

        ones_mask = torch.ones_like(drm10, device=drm10.device)

        def calc_drm_rife(_t):
            # The distance ratio map (drm) is initially aligned with I1.
            # To align it with I0 and I2, we need to warp the drm maps.
            # Note: 1. To reverse the direction of the drm map, use 1 - drm and then warp it.
            # 2. For RIFE, drm should be aligned with the time corresponding to the intermediate frame.
            _drm01r = self.fwarp(1 - drm10, flow10 * ((1 - drm10) * 2) * _t, None, strMode="avg")
            _drm21r = self.fwarp(1 - drm12, flow12 * ((1 - drm12) * 2) * _t, None, strMode="avg")

            self.warped_ones_mask01r = self.fwarp(ones_mask, flow10 * ((1 - drm10) * 2) * _t, None, strMode="avg")
            self.warped_ones_mask21r = self.fwarp(ones_mask, flow12 * ((1 - drm12) * 2) * _t, None, strMode="avg")

            holes01r = self.warped_ones_mask01r < 0.999
            holes21r = self.warped_ones_mask21r < 0.999

            _drm01r[holes01r] = _drm01r[holes01r]
            _drm21r[holes21r] = _drm21r[holes21r]

            return _drm01r, _drm21r

        output1, output2 = [], []

        if _left_scene:
            for i in range(len(minus_t)):
                minus_t[i] = -1

        if _right_scene:
            for _ in plus_t:
                zero_t = np.append(zero_t, 0)
            plus_t = []

        disable_drm = False
        if (_left_scene and not _right_scene) or (not _left_scene and _right_scene):
            drm01r, drm21r = (ones_mask.clone() * 0.5 for _ in range(2))
            drm01r = torch.nn.functional.interpolate(drm01r, size=_I0.shape[2:], mode="bilinear", align_corners=False)
            drm21r = torch.nn.functional.interpolate(drm21r, size=_I0.shape[2:], mode="bilinear", align_corners=False)
            disable_drm = True

        for t in minus_t:
            t = -t
            if t == 1:
                output1.append(_I0)
                continue
            if not disable_drm:
                drm01r, _ = calc_drm_rife(t)
            output1.append(
                self.inference(
                    torch.cat((_I1, _I0), 1),
                    timestep=t * (2 * drm01r),
                    scale_list=[16 / _scale, 8 / _scale, 4 / _scale, 2 / _scale, 1 / _scale],
                )[0]
            )
        for _ in zero_t:
            output1.append(_I1)
        for t in plus_t:
            if t == 1:
                output2.append(_I2)
                continue
            if not disable_drm:
                _, drm21r = calc_drm_rife(t)
            output2.append(
                self.inference(
                    torch.cat((_I1, _I2), 1),
                    timestep=t * (2 * drm21r),
                    scale_list=[16 / _scale, 8 / _scale, 4 / _scale, 2 / _scale, 1 / _scale],
                )[0]
            )

        _output = output1 + output2

        # next flow10, flow01 = reverse(current flow12, flow21)
        return _output, (flow21, flow12, f2, f1)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True
        ),
        nn.LeakyReLU(0.2, True),
    )


def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, True),
    )


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 16, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, feat=False):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) * 1.0 / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow, mask, feat
