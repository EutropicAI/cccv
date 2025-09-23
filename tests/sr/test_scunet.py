import cv2

from cccv import AutoConfig, AutoModel, BaseConfig, ConfigType
from cccv.model import SRBaseModel
from tests.util import (
    ASSETS_PATH,
    CCCV_DEVICE,
    CCCV_FP16,
    CCCV_TILE,
    calculate_image_similarity,
    compare_image_size,
    load_image,
)


class Test_SCUNet:
    def test_official(self) -> None:
        img1 = load_image()

        for k in [
            ConfigType.SCUNet_color_50_1x,
            ConfigType.SCUNet_color_real_psnr_1x,
            ConfigType.SCUNet_color_real_gan_1x,
        ]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: SRBaseModel = AutoModel.from_config(config=cfg, device=CCCV_DEVICE, fp16=CCCV_FP16, tile=CCCV_TILE)
            print(model.device)

            img2 = model.inference_image(img1)
            cv2.imwrite(str(ASSETS_PATH / f"test_{k}_out.jpg"), img2)

            assert calculate_image_similarity(img1, img2)
            assert compare_image_size(img1, img2, cfg.scale)
