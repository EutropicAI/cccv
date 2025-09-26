# cccv

[![codecov](https://codecov.io/gh/EutropicAI/cccv/graph/badge.svg?token=VK0BHDUXAI)](https://codecov.io/gh/EutropicAI/cccv)
[![CI-test](https://github.com/EutropicAI/cccv/actions/workflows/CI-test.yml/badge.svg)](https://github.com/EutropicAI/cccv/actions/workflows/CI-test.yml)
[![Release](https://github.com/EutropicAI/cccv/actions/workflows/Release.yml/badge.svg)](https://github.com/EutropicAI/cccv/actions/workflows/Release.yml)
[![PyPI version](https://badge.fury.io/py/cccv.svg)](https://badge.fury.io/py/cccv)
![GitHub](https://img.shields.io/github/license/EutropicAI/cccv)

an inference lib for image/video restoration and video frame interpolation with VapourSynth support

### Install

Make sure you have Python >= 3.9 and PyTorch >= 2.0 installed

```bash
pip install cccv
```

- Install VapourSynth (optional)

### Start

#### Load a registered model in cccv

a simple example to use the SISR (Single Image Super-Resolution) model to process an image

```python
import cv2
import numpy as np

from cccv import AutoModel, ConfigType, SRBaseModel

model: SRBaseModel = AutoModel.from_pretrained(ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x)

img = cv2.imdecode(np.fromfile("test.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
img = model.inference_image(img)
cv2.imwrite("test_out.jpg", img)
```

#### Load a custom model from remote repository or local path

a simple example to use [remote repository](https://github.com/EutropicAI/cccv_demo_remote_model), auto register then load

```python
import cv2
import numpy as np

from cccv import AutoModel, SRBaseModel

# remote repo
model: SRBaseModel = AutoModel.from_pretrained("https://github.com/EutropicAI/cccv_demo_remote_model")
# local path
model: SRBaseModel = AutoModel.from_pretrained("/path/to/cccv_demo_model")
```

#### VapourSynth

a simple example to use the VapourSynth to process a video

```python
import vapoursynth as vs
from vapoursynth import core

from cccv import AutoModel, ConfigType, SRBaseModel

model: SRBaseModel = AutoModel.from_pretrained(
    ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x,
    tile=None
)

clip = core.bs.VideoSource(source="s.mp4")
clip = core.resize.Bicubic(clip=clip, matrix_in_s="709", format=vs.RGBH)
clip = model.inference_video(clip)
clip = core.resize.Bicubic(clip=clip, matrix_s="709", format=vs.YUV420P16)
clip.set_output()
```

See more examples in the [example](./example) directory, including:

- SISR (Single Image Super-Resolution)
- VSR (Video Super-Resolution)
- VFI (Video Frame Interpolation)

cccv can register custom configurations and models to extend the functionality

### Current Support

It still in development, the following models are supported:

- [Architecture](./cccv/type/arch.py)
- [Model](./cccv/type/model.py)
- [Weight(Config)](./cccv/type/config.py)

### Notice

- All the architectures have been edited to normalize input and output, and automatic padding is applied. The input and output tensor shapes may differ from the original architectures. For SR models, the input and output are both 4D tensors in the shape of `(b, c, h, w)`. For VSR models, the input and output are both 5D tensors in the shape of `(b, l, c, h, w)`.

- For VSR models with equal l in input and output `(f1, f2, f3, f4 -> f1', f2', f3', f4')`, you can directly extend from `class VSRBaseModel`. For VSR models that output only one frame `(f-2, f-1, f0, f1, f2 -> f0')`, you also need to set `self.one_frame_out = True`.

### Reference

- [PyTorch](https://github.com/pytorch/pytorch)
- [BasicSR](https://github.com/XPixelGroup/BasicSR)
- [mmcv](https://github.com/open-mmlab/mmcv)
- [huggingface transformers](https://github.com/huggingface/transformers)
- [VapourSynth](https://www.vapoursynth.com/)
- [HolyWu's functions](https://github.com/HolyWu)

### License

This project is licensed under the MIT - see
the [LICENSE file](./LICENSE) for details.
