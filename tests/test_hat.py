import cv2
import pytest

from cccv import AutoConfig, AutoModel, BaseConfig, ConfigType
from cccv.model import SRBaseModel

from .util import (
    ASSETS_PATH,
    CCCV_DEVICE,
    CI_ENV,
    calculate_image_similarity,
    compare_image_size,
    load_image,
)


class Test_HAT:
    def test_official_light(self) -> None:
        img1 = load_image()

        for k in [
            ConfigType.HAT_S_2x,
            ConfigType.HAT_S_3x,
            ConfigType.HAT_S_4x,
        ]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: SRBaseModel = AutoModel.from_config(config=cfg, device=CCCV_DEVICE, fp16=False)
            print(model.device)

            img2 = model.inference_image(img1)
            cv2.imwrite(str(ASSETS_PATH / f"test_{k}_out.jpg"), img2)

            assert calculate_image_similarity(img1, img2)
            assert compare_image_size(img1, img2, cfg.scale)

    @pytest.mark.skipif(CI_ENV, reason="Skip on CI test")
    def test_official(self) -> None:
        img1 = load_image()

        for k in [
            ConfigType.HAT_S_2x,
            ConfigType.HAT_S_3x,
            ConfigType.HAT_S_4x,
            ConfigType.HAT_2x,
            ConfigType.HAT_3x,
            ConfigType.HAT_4x,
            ConfigType.HAT_ImageNet_pretrain_2x,
            ConfigType.HAT_ImageNet_pretrain_3x,
            ConfigType.HAT_ImageNet_pretrain_4x,
            ConfigType.HAT_L_ImageNet_pretrain_2x,
            ConfigType.HAT_L_ImageNet_pretrain_3x,
            ConfigType.HAT_L_ImageNet_pretrain_4x,
            ConfigType.HAT_Real_GAN_sharper_4x,
            ConfigType.HAT_Real_GAN_4x,
        ]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: SRBaseModel = AutoModel.from_config(config=cfg, device=CCCV_DEVICE, fp16=False)
            print(model.device)

            img2 = model.inference_image(img1)
            cv2.imwrite(str(ASSETS_PATH / f"test_{k}_out.jpg"), img2)

            assert calculate_image_similarity(img1, img2)
            assert compare_image_size(img1, img2, cfg.scale)

    @pytest.mark.skipif(CI_ENV, reason="Skip on CI test")
    def test_custom(self) -> None:
        img1 = load_image()

        for k in [ConfigType.HAT_Real_GAN_sharper_4x]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: SRBaseModel = AutoModel.from_config(config=cfg, device=CCCV_DEVICE, fp16=False)
            print(model.device)

            img2 = model.inference_image(img1)
            cv2.imwrite(str(ASSETS_PATH / f"test_{k}_out.jpg"), img2)

            assert calculate_image_similarity(img1, img2)
            assert compare_image_size(img1, img2, cfg.scale)
