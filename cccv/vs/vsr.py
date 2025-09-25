from threading import Lock
from typing import Any, Callable, Dict, Union

import torch
import vapoursynth as vs

from cccv.vs.convert import frame_to_tensor, tensor_to_frame


def inference_vsr(
    inference: Callable[[torch.Tensor], torch.Tensor],
    clip: vs.VideoNode,
    scale: Union[float, int, Any],
    num_frame: int,
    device: torch.device,
    one_frame_out: bool = False,
) -> vs.VideoNode:
    """
    Inference the video with the model, the clip should be a vapoursynth clip

    :param inference: The inference function
    :param clip: vs.VideoNode
    :param scale: The scale factor
    :param num_frame: Number of input frames
    :param device: The device
    :param one_frame_out: Whether the model is one frame output model
    :return:
    """
    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("[CCCV] Only vs.RGBH and vs.RGBS formats are supported")

    if num_frame > clip.num_frames:
        raise ValueError("[CCCV] Input frames should be less than the number of frames in the clip")
    elif num_frame <= 1:
        raise ValueError("[CCCV] Input frames should be greater than 1")

    if not one_frame_out:
        return inference_vsr_multi_frame_out(inference, clip, scale, num_frame, device)
    else:
        return inference_vsr_one_frame_out(inference, clip, scale, num_frame, device)


def inference_vsr_multi_frame_out(
    inference: Callable[[torch.Tensor], torch.Tensor],
    clip: vs.VideoNode,
    scale: Union[float, int, Any],
    num_frame: int,
    device: torch.device,
) -> vs.VideoNode:
    """
    VSR for multi frame output models

    f1, f2, f3, f4 -> f1', f2', f3', f4'

    For the multi frame output model, the inference function should accept a tensor with shape (1, n, c, h, w)
    And return a tensor with shape (1, n, c, h, w)

    :param inference: The inference function
    :param clip: vs.VideoNode
    :param scale: The scale factor
    :param num_frame: Number of input frames
    :param device: The device
    :return:
    """

    cache: Dict[int, torch.Tensor] = {}

    lock = Lock()

    def _inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        with lock:
            if n not in cache:
                cache.clear()

                img = []
                for i in range(num_frame):
                    index = n + i
                    if index >= clip.num_frames:
                        img.append(frame_to_tensor(clip.get_frame(clip.num_frames - 1), device=device).unsqueeze(0))

                    else:
                        img.append(frame_to_tensor(clip.get_frame(n + i), device=device).unsqueeze(0))

                img = torch.stack(img, dim=1)

                output = inference(img)

                for i in range(output.shape[0]):
                    cache[n + i] = output[0, i, :, :, :]

            res = tensor_to_frame(cache[n], f[1].copy())

        return res

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale, keep=True)
    return new_clip.std.ModifyFrame([clip, new_clip], _inference)


def inference_vsr_one_frame_out(
    inference: Callable[[torch.Tensor], torch.Tensor],
    clip: vs.VideoNode,
    scale: Union[float, int, Any],
    num_frame: int,
    device: torch.device,
) -> vs.VideoNode:
    """
    VSR for one frame output models

    f-2, f-1, f0, f1, f2 -> f0'

    For these models, the inference function should accept a tensor with shape (1, n, c, h, w)
    And return a tensor with shape (1, 1, c, h, w)

    :param inference: The inference function
    :param clip: vs.VideoNode
    :param scale: The scale factor
    :param num_frame: Number of input frames, should be odd
    :param device: The device
    :return:
    """

    if num_frame % 2 == 0:
        raise ValueError("[CCCV] The length of the input frames should be odd")

    lock = Lock()

    def _inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        with lock:
            img = []
            for i in range(num_frame):
                index = i - num_frame // 2 + n
                if index < 0:
                    img.append(frame_to_tensor(clip.get_frame(0), device=device).unsqueeze(0))

                elif index >= clip.num_frames:
                    img.append(frame_to_tensor(clip.get_frame(clip.num_frames - 1), device=device).unsqueeze(0))

                else:
                    img.append(frame_to_tensor(clip.get_frame(index), device=device).unsqueeze(0))

            img = torch.stack(img, dim=1)

            output = inference(img)

            res = tensor_to_frame(output[0, 0, :, :, :], f[1].copy())

        return res

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale, keep=True)
    return new_clip.std.ModifyFrame([clip, new_clip], _inference)
