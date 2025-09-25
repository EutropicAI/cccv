import sys

sys.path.append(".")
sys.path.append("..")

import vapoursynth as vs
from vapoursynth import core

from cccv import AutoModel, CCBaseModel, ConfigType

example = 0

if example == 0:
    # VSR for multi frame output models
    #
    # f1, f2, f3, f4 -> f1', f2', f3', f4'
    model: CCBaseModel = AutoModel.from_pretrained(ConfigType.AnimeSR_v2_4x, tile=None)

    clip = core.bs.VideoSource(source="s.mkv")
    clip = core.resize.Bicubic(clip=clip, matrix_in_s="709", format=vs.RGBH)
    clip = model.inference_video(clip)
    clip = core.resize.Bicubic(clip=clip, matrix_s="709", format=vs.YUV420P16)
    clip.set_output()

elif example == 1:
    # VSR for one frame output models
    #
    # f-2, f-1, f0, f1, f2 -> f0'
    #
    # Should enable self.one_frame_out = True
    # @MODEL_REGISTRY.register(name=ModelType.EDVR)
    # class EDVRModel(VSRBaseModel):
    #     def post_init_hook(self) -> None:
    #         self.one_frame_out = True

    model: CCBaseModel = AutoModel.from_pretrained(ConfigType.EDVR_M_SR_REDS_official_4x, tile=(256, 256))

    clip = core.bs.VideoSource(source="s.mkv")
    clip = core.resize.Bicubic(clip=clip, matrix_in_s="709", format=vs.RGBH)
    clip = model.inference_video(clip)
    clip = core.resize.Bicubic(clip=clip, matrix_s="709", format=vs.YUV420P16)
    clip.set_output()

else:
    raise NotImplementedError
