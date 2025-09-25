import sys

sys.path.append(".")
sys.path.append("..")

import vapoursynth as vs
from vapoursynth import core

from cccv import AutoModel, ConfigType, VFIBaseModel

# --- IFNet, use fp16 to inference (vs.RGBH)

model: VFIBaseModel = AutoModel.from_pretrained(
    ConfigType.RIFE_IFNet_v426_heavy,
    fp16=True,
)

core.num_threads = 1  # should be set to single thread now, TODO: fix it
clip = core.bs.VideoSource(source="s.mp4")
clip = core.resize.Bicubic(clip=clip, matrix_in_s="709", format=vs.RGBH)
clip = model.inference_video(clip, scale=1.0, tar_fps=60, scdet=True, scdet_threshold=0.3)
clip = core.resize.Bicubic(clip=clip, matrix_s="709", format=vs.YUV420P16)
clip.set_output()
