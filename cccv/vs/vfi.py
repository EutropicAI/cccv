import math
from typing import Callable, Dict

import numpy as np
import torch
import vapoursynth as vs
from vapoursynth import core

from cccv.util.misc import TMapper, check_scene
from cccv.vs.convert import frame_to_tensor, tensor_to_frame


def inference_vfi(
    inference: Callable,
    clip: vs.VideoNode,
    scale: float,
    tar_fps: float,
    device: torch.device,
    num_frame: int = 2,
    scdet: bool = True,
    scdet_threshold: float = 0.3,
) -> vs.VideoNode:
    """
    Inference the video with the model, the clip should be a vapoursynth clip

    :param inference: The inference function
    :param clip: vs.VideoNode
    :param scale: The flow scale factor
    :param tar_fps: The fps of the interpolated video
    :param device: The device
    :param num_frame: The input frame count of vfi method once infer
    :param scdet: Enable SSIM scene change detection
    :param scdet_threshold: SSIM scene change detection threshold (greater is sensitive)
    :return:
    """

    if core.num_threads != 1:
        raise ValueError("[CCCV] The number of threads must be 1 when enable frame interpolation")

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("[CCCV] Only vs.RGBH and vs.RGBS formats are supported")

    if num_frame > clip.num_frames:
        raise ValueError("[CCCV] Input frames should be less than the number of frames in the clip")
    elif num_frame <= 1:
        raise ValueError("[CCCV] Input frames should be greater than 1")

    src_fps = clip.fps.numerator / clip.fps.denominator
    if src_fps > tar_fps:
        raise ValueError("[CCCV] The target fps should be greater than the clip fps")

    if scale < 0 or not math.log2(scale).is_integer():
        raise ValueError("[CCCV] The scale should be greater than 0 and is power of two")

    vfi_methods = {
        2: inference_vfi_two_frame_in,
        3: inference_vfi_three_frame_in,
    }

    if num_frame not in vfi_methods:
        raise ValueError(f"[CCCV] The vfi method with {num_frame} frame input is not supported")

    mapper = TMapper(src_fps, tar_fps)

    return vfi_methods[num_frame](inference, clip, mapper, scale, scdet, scdet_threshold, device)


def inference_vfi_two_frame_in(
    inference: Callable,
    clip: vs.VideoNode,
    mapper: TMapper,
    scale: float,
    scdet: bool,
    scdet_threshold: float,
    device: torch.device,
) -> vs.VideoNode:
    """
    VFI for two frame input models

    f1, f2 -> f1?, f1t?, f2?

    For the two frame input model, the inference function should accept a tensor with shape (b, 2, c, h, w)
    And return a tensor with shape (b, c, h, w)

    :param inference: The inference function
    :param clip: vs.VideoNode
    :param scale: The flow scale factor
    :param mapper: The framerate mapper
    :param scdet: Enable SSIM scene change detection
    :param scdet_threshold: SSIM scene change detection threshold (greater is sensitive)
    :param device: The device
    :return:
    """

    in_idx: int = 0
    out_idx: int = 0
    in_frames: Dict[int, torch.Tensor] = {}
    out_frames: Dict[int, torch.Tensor] = {}
    flag_end: bool = False
    reuse: tuple[torch.Tensor, ...]

    def to_input_tensor(x: vs.VideoFrame) -> torch.Tensor:
        return frame_to_tensor(x, device=device).unsqueeze(0).unsqueeze(0)

    new_clip = clip.std.AssumeFPS(fpsnum=mapper.dst, fpsden=1)
    less_num_frames = math.ceil(clip.num_frames * mapper.dst / mapper.src) - clip.num_frames
    for _ in range(less_num_frames):
        new_clip = new_clip.std.DuplicateFrames(clip.num_frames - 1)

    def _inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        nonlocal in_idx, out_idx, in_frames, out_frames, flag_end, reuse
        if n >= out_idx and not flag_end:
            if in_idx not in in_frames.keys():
                in_frames[in_idx] = to_input_tensor(clip.get_frame(in_idx))
            I0 = in_frames[in_idx]

            if in_idx + 1 >= clip.num_frames - 1:
                flag_end = True
                return tensor_to_frame(out_frames[list(out_frames.keys())[-1]], f[1].copy())

            if in_idx + 1 not in in_frames.keys():
                in_frames[in_idx + 1] = to_input_tensor(clip.get_frame(in_idx + 1))
            I1 = in_frames[in_idx + 1]

            ts = mapper.get_range_timestamps(in_idx, in_idx + 1, lclose=True, rclose=flag_end, normalize=True)

            scene = check_scene(I0, I1, scdet, scdet_threshold)

            for t in ts:
                if scene:
                    out = I0.squeeze(0)
                else:
                    if t == 0:
                        out = I0.squeeze(0)
                    elif t == 1:
                        out = I1.squeeze(0)
                    else:
                        out = inference(torch.cat([I0, I1], dim=1), timestep=t, scale=scale)
                out_frames[out_idx] = out
                out_idx += 1

            # clear input cache
            if in_idx - 1 in in_frames.keys():
                in_frames.pop(in_idx - 1)

            in_idx += 1

        # clear output cache
        if n - 1 in out_frames.keys() and len(out_frames.keys()) > 2:
            out_frames.pop(n - 1)

        if n not in out_frames.keys():
            return tensor_to_frame(out_frames[list(out_frames.keys())[-1]], f[1].copy())

        return tensor_to_frame(out_frames[n], f[1].copy())

    return new_clip.std.ModifyFrame([new_clip, new_clip], _inference)


def inference_vfi_three_frame_in(
    inference: Callable,
    clip: vs.VideoNode,
    mapper: TMapper,
    scale: float,
    scdet: bool,
    scdet_threshold: float,
    device: torch.device,
) -> vs.VideoNode:
    """
    VFI for three frame input models

    f1, f2, f3 -> f1?, f1t?, f2?, f2t?, f3?

    For the three frame input model, the inference function should accept a tensor with shape (b, 3, c, h, w)
    And return a tensor with shape (b, c, h, w)

    :param inference: The inference function
    :param clip: vs.VideoNode
    :param scale: The flow scale factor
    :param mapper: The framerate mapper
    :param scdet: Enable SSIM scene change detection
    :param scdet_threshold: SSIM scene change detection threshold (greater is sensitive)
    :param device: The device
    :return:
    """

    in_idx: int = 0
    out_idx: int = 0
    in_frames: Dict[int, torch.Tensor] = {}
    out_frames: Dict[int, torch.Tensor] = {}
    flag_end: bool = False
    reuse: tuple[torch.Tensor, ...]

    def calc_t(_mapper: TMapper, _idx: float, _flag_end: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ts = _mapper.get_range_timestamps(_idx - 0.5, _idx + 0.5, lclose=True, rclose=_flag_end, normalize=False)
        timestamp = np.asarray(ts, dtype=float) - _idx
        vfi_timestamp = np.round(timestamp, 4)

        minus_t = vfi_timestamp[vfi_timestamp < 0]
        zero_t = vfi_timestamp[vfi_timestamp == 0]
        plus_t = vfi_timestamp[vfi_timestamp > 0]
        return minus_t, zero_t, plus_t

    def to_input_tensor(x: vs.VideoFrame) -> torch.Tensor:
        return frame_to_tensor(x, device=device).unsqueeze(0).unsqueeze(0)

    new_clip = clip.std.AssumeFPS(fpsnum=mapper.dst, fpsden=1)
    less_num_frames = math.ceil(clip.num_frames * mapper.dst / mapper.src) - clip.num_frames
    for _ in range(less_num_frames):
        new_clip = new_clip.std.DuplicateFrames(clip.num_frames - 1)

    def _inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        nonlocal in_idx, out_idx, in_frames, out_frames, flag_end, reuse
        if n >= out_idx and not flag_end:
            if in_idx not in in_frames.keys():
                in_frames[in_idx] = to_input_tensor(clip.get_frame(in_idx))
            I0 = in_frames[in_idx]

            if in_idx + 1 >= clip.num_frames - 1:
                flag_end = True
                return tensor_to_frame(out_frames[list(out_frames.keys())[-1]], f[1].copy())

            if in_idx + 1 not in in_frames.keys():
                in_frames[in_idx + 1] = to_input_tensor(clip.get_frame(in_idx + 1))
            I1 = in_frames[in_idx + 1]

            if in_idx + 2 >= clip.num_frames - 1:
                flag_end = True
            else:
                if in_idx + 2 not in in_frames.keys():
                    in_frames[in_idx + 2] = to_input_tensor(clip.get_frame(in_idx + 2))
                I2 = in_frames[in_idx + 2]

            mt, zt, pt = calc_t(mapper, in_idx, flag_end)
            left_scene = check_scene(I0, I1, scdet, scdet_threshold)
            if in_idx == 0:  # head
                right_scene = left_scene
                output, reuse = inference(torch.cat([I0, I0, I1], dim=1), mt, zt, pt, False, right_scene, scale, None)
            elif flag_end:  # tail
                output, _ = inference(torch.cat([I0, I1, I1], dim=1), mt, zt, pt, left_scene, False, scale, reuse)
            else:
                right_scene = check_scene(I1, I2, scdet, scdet_threshold)
                output, reuse = inference(
                    torch.cat([I0, I1, I2], dim=1), mt, zt, pt, left_scene, right_scene, scale, reuse
                )

            for i in range(output.shape[1]):
                out_frames[out_idx] = output[0, i : i + 1]
                out_idx += 1

            # clear input cache
            if in_idx - 1 in in_frames.keys():
                in_frames.pop(in_idx - 1)

            in_idx += 1

        # clear output cache
        if n - 1 in out_frames.keys() and len(out_frames.keys()) > 2:
            out_frames.pop(n - 1)

        if n not in out_frames.keys():
            return tensor_to_frame(out_frames[list(out_frames.keys())[-1]], f[1].copy())

        return tensor_to_frame(out_frames[n], f[1].copy())

    return new_clip.std.ModifyFrame([new_clip, new_clip], _inference)
