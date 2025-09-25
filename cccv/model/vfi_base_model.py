from typing import Any, List

import numpy as np
import torch

from cccv.config import VFIBaseConfig
from cccv.model import MODEL_REGISTRY
from cccv.model.base_model import CCBaseModel
from cccv.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.VFIBaseModel)
class VFIBaseModel(CCBaseModel):
    def inference(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError

    def inference_image_list(self, img_list: List[np.ndarray], *args: Any, **kwargs: Any) -> List[np.ndarray]:
        raise NotImplementedError

    @torch.inference_mode()  # type: ignore
    def inference_video(
        self,
        clip: Any,
        scale: float = 1.0,
        tar_fps: float = 60,
        scdet: bool = True,
        scdet_threshold: float = 0.3,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Inference the video with the model, the clip should be a vapoursynth clip

        :param clip: vs.VideoNode
        :param scale: The flow scale factor
        :param tar_fps: The fps of the interpolated video
        :param scdet: Enable SSIM scene change detection
        :param scdet_threshold: SSIM scene change detection threshold (greater is sensitive)
        :return:
        """

        from cccv.vs import inference_vfi

        cfg: VFIBaseConfig = self.config

        return inference_vfi(
            inference=self.inference,
            clip=clip,
            scale=scale,
            tar_fps=tar_fps,
            num_frame=cfg.num_frame,
            scdet=scdet,
            scdet_threshold=scdet_threshold,
            device=self.device,
        )
