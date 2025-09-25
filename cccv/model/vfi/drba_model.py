from typing import Any, List

import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision import transforms

from cccv.model import MODEL_REGISTRY
from cccv.model.vfi_base_model import VFIBaseModel
from cccv.type import ModelType
from cccv.util.misc import de_resize, resize


@MODEL_REGISTRY.register(name=ModelType.DRBA)
class DRBAModel(VFIBaseModel):
    def post_init_hook(self) -> None:
        self.load_state_dict_strict = False

    def transform_state_dict(self, state_dict: Any) -> Any:
        def _convert(param: Any) -> Any:
            return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}

        return _convert(state_dict)

    @torch.inference_mode()  # type: ignore
    def inference(
        self,
        imgs: torch.Tensor,
        minus_t: list[float],
        zero_t: list[float],
        plus_t: list[float],
        left_scene_change: bool,
        right_scene_change: bool,
        scale: float,
        reuse: Any,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Tensor, Any]:
        """
        Inference with the model

        :param imgs: The input frames (B, 3, C, H, W)
        :param minus_t: Timestep between -1 and 0 (I0 and I1)
        :param zero_t: Timestep of 0, if not empty, preserve I1 (I1)
        :param plus_t: Timestep between 0 and 1 (I1 and I2)
        :param left_scene_change: True if there is a scene change between I0 and I1 (I0 and I1)
        :param right_scene_change: True if there is a scene change between I1 and I2 (I1 and I2)
        :param scale: Flow scale.
        :param reuse: Reusable output from model with last frame pair.

        :return: All immediate frames between I0~I2 and reusable contents.
        """

        I0, I1, I2 = imgs[:, 0], imgs[:, 1], imgs[:, 2]
        _, _, h, w = I0.shape
        I0 = resize(I0, scale).unsqueeze(0)
        I1 = resize(I1, scale).unsqueeze(0)
        I2 = resize(I2, scale).unsqueeze(0)

        inp = torch.cat([I0, I1, I2], dim=1)

        results, reuse = self.model(inp, minus_t, zero_t, plus_t, left_scene_change, right_scene_change, scale, reuse)

        results = torch.cat(tuple(de_resize(result, h, w).unsqueeze(0) for result in results), dim=1)

        return results, reuse

    @torch.inference_mode()  # type: ignore
    def inference_image_list(self, img_list: List[np.ndarray], *args: Any, **kwargs: Any) -> List[np.ndarray]:
        """
        Inference numpy image list with the model

        :param img_list: 3 input frames (img0, img1, img2)

        :return: 5 output frames (img0, img0_1, img1, img1_2, img2)
        """
        if len(img_list) != 3:
            raise ValueError("DRBA img_list must contain 3 images")

        new_img_list = []
        for img in img_list:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
            new_img_list.append(img)

        # b, n, c, h, w
        img_tensor_stack = torch.stack(new_img_list, dim=1)
        if self.fp16:
            img_tensor_stack = img_tensor_stack.half()

        results, _ = self.inference(img_tensor_stack, [-1, -0.5], [0], [0.5, 1], False, False, 1.0, None)

        results_list = []
        for i in range(results.shape[1]):
            img = results[0, i, :, :, :]
            img = img.permute(1, 2, 0).cpu().numpy()
            img = (img * 255).clip(0, 255).astype("uint8")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            results_list.append(img)

        return results_list
