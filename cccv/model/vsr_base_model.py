from typing import Any, List

import cv2
import numpy as np
import torch
from torchvision import transforms

from cccv.config import VSRBaseConfig
from cccv.model import MODEL_REGISTRY
from cccv.model.base_model import CCBaseModel
from cccv.model.tile import tile_vsr
from cccv.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.VSRBaseModel)
class VSRBaseModel(CCBaseModel):
    @torch.inference_mode()  # type: ignore
    def inference(self, img: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        cfg: VSRBaseConfig = self.config

        if self.tile is None:
            return self.model(img)

        # tile processing
        return tile_vsr(
            model=self.model,
            scale=cfg.scale,
            img=img,
            one_frame_out=self.one_frame_out,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pad_img=self.pad_img,
        )

    @torch.inference_mode()  # type: ignore
    def inference_image_list(self, img_list: List[np.ndarray], *args: Any, **kwargs: Any) -> List[np.ndarray]:
        """
        Inference the image list with the VSR model

        :param img_list: List[np.ndarray]
        :return: List[np.ndarray]
        """
        new_img_list = []
        for img in img_list:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
            new_img_list.append(img)

        # b, n, c, h, w
        img_tensor_stack = torch.stack(new_img_list, dim=1)
        if self.fp16:
            img_tensor_stack = img_tensor_stack.half()

        out = self.inference(img_tensor_stack)

        if len(out.shape) == 5:
            res_img_list = []

            for i in range(out.shape[1]):
                img = out[0, i, :, :, :]
                img = img.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).clip(0, 255).astype("uint8")
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                res_img_list.append(img)

            return res_img_list

        elif len(out.shape) == 4:
            img = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img = (img * 255).clip(0, 255).astype("uint8")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            return [img]
        else:
            raise ValueError(f"[CCCV] Unexpected output shape: {out.shape}")

    @torch.inference_mode()  # type: ignore
    def inference_video(self, clip: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Inference the video with the model, the clip should be a vapoursynth clip

        :param clip: vs.VideoNode
        :return:
        """

        from cccv.vs import inference_vsr

        cfg: VSRBaseConfig = self.config

        return inference_vsr(
            inference=self.inference,
            clip=clip,
            scale=cfg.scale,
            num_frame=cfg.num_frame,
            device=self.device,
            one_frame_out=self.one_frame_out,
        )
