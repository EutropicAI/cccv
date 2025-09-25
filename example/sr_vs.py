import sys

sys.path.append(".")
sys.path.append("..")

import vapoursynth as vs
from vapoursynth import core

from cccv import AutoModel, CCBaseModel, ConfigType

example = 0

if example == 0:
    # --- sisr, use fp16 to inference (vs.RGBH)

    model: CCBaseModel = AutoModel.from_pretrained(ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x, tile=None)

    clip = core.bs.VideoSource(source="s.mp4")
    clip = core.resize.Bicubic(clip=clip, matrix_in_s="709", format=vs.RGBH)
    clip = model.inference_video(clip)
    clip = core.resize.Bicubic(clip=clip, matrix_s="709", format=vs.YUV420P16)
    clip.set_output()

elif example == 1:
    # ---  use fp32 to inference (vs.RGBS)

    model: CCBaseModel = AutoModel.from_pretrained(
        ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x, fp16=False, tile=None
    )

    clip = core.bs.VideoSource(source="s.mp4")
    clip = core.resize.Bicubic(clip=clip, matrix_in_s="709", format=vs.RGBS)
    clip = model.inference_video(clip)
    clip = core.resize.Bicubic(clip=clip, matrix_s="709", format=vs.YUV420P16)
    clip.set_output()


else:
    raise NotImplementedError
