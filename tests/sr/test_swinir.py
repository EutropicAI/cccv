import cv2

from cccv import AutoConfig, AutoModel, BaseConfig, ConfigType
from cccv.model import SRBaseModel
from tests.util import (
    ASSETS_PATH,
    CCCV_DEVICE,
    CCCV_TILE,
    calculate_image_similarity,
    compare_image_size,
    load_image,
)


class Test_SwinIR:
    def test_official(self) -> None:
        img1 = load_image()

        for k in [
            ConfigType.SwinIR_classicalSR_DF2K_s64w8_SwinIR_M_2x,
            ConfigType.SwinIR_lightweightSR_DIV2K_s64w8_SwinIR_S_2x,
            ConfigType.SwinIR_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR_L_GAN_4x,
            ConfigType.SwinIR_realSR_BSRGAN_DFO_s64w8_SwinIR_M_GAN_2x,
            ConfigType.SwinIR_realSR_BSRGAN_DFO_s64w8_SwinIR_M_GAN_4x,
        ]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: SRBaseModel = AutoModel.from_config(config=cfg, device=CCCV_DEVICE, fp16=False, tile=CCCV_TILE)
            print(model.device)

            img2 = model.inference_image(img1)
            cv2.imwrite(str(ASSETS_PATH / f"test_{k}_out.jpg"), img2)

            assert calculate_image_similarity(img1, img2)
            assert compare_image_size(img1, img2, cfg.scale)

    def test_custom(self) -> None:
        img1 = load_image()

        for k in [ConfigType.SwinIR_Bubble_AnimeScale_SwinIR_Small_v1_2x]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: SRBaseModel = AutoModel.from_config(config=cfg, device=CCCV_DEVICE, fp16=False, tile=CCCV_TILE)
            print(model.device)

            img2 = model.inference_image(img1)
            cv2.imwrite(str(ASSETS_PATH / f"test_{k}_out.jpg"), img2)

            assert calculate_image_similarity(img1, img2)
            assert compare_image_size(img1, img2, cfg.scale)
