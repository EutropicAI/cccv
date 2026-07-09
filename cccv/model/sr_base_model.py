from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

from cccv.config import SRBaseConfig
from cccv.model import MODEL_REGISTRY
from cccv.model.base_model import CCBaseModel
from cccv.model.tile import tile_sr
from cccv.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.SRBaseModel)
class SRBaseModel(CCBaseModel):
    def get_bf16_preflight_inputs(self) -> Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]]:
        height, width = self._get_bf16_preflight_image_size()
        img = torch.zeros((1, 3, height, width), device=self.device, dtype=self.half_dtype)
        return (img,), {}

    @torch.inference_mode()  # type: ignore
    def inference(self, img: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        cfg: SRBaseConfig = self.config

        if self.tile is None:
            return self.model(img)

        # tile processing
        return tile_sr(
            model=self.model,
            scale=cfg.scale,
            img=img,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pad_img=self.pad_img,
        )

    @torch.inference_mode()  # type: ignore
    def inference_image(self, img: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Inference the image(BGR) with the model

        :param img: The input image(BGR), can use cv2 to read the image
        :return:
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        if self.fp16 or self.bf16:
            img = img.to(self.half_dtype)

        img = self.inference(img)
        img = self._tensor_to_numpy(img.squeeze(0).permute(1, 2, 0))
        img = (img * 255).clip(0, 255).astype("uint8")

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    @torch.inference_mode()  # type: ignore
    def inference_video(self, clip: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Inference the video with the model, the clip should be a vapoursynth clip

        :param clip: vs.VideoNode
        :return:
        """
        cfg: SRBaseConfig = self.config

        from cccv.vs import inference_sr

        return inference_sr(inference=self.inference, clip=clip, scale=cfg.scale, device=self.device)
