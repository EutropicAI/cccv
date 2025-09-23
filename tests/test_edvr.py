import cv2

from cccv import AutoConfig, AutoModel, BaseConfig, ConfigType
from cccv.model import VSRBaseModel

from .util import (
    ASSETS_PATH,
    CCCV_DEVICE,
    calculate_image_similarity,
    compare_image_size,
    load_image,
)


class Test_EDVR:
    def test_official_M(self) -> None:
        img = load_image()
        imgList = [img, img, img, img, img]

        for k in [ConfigType.EDVR_M_SR_REDS_official_4x, ConfigType.EDVR_M_woTSA_SR_REDS_official_4x]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: VSRBaseModel = AutoModel.from_config(config=cfg, device=CCCV_DEVICE, fp16=False)
            print(model.device)

            img2 = model.inference_image_list(imgList)
            cv2.imwrite(str(ASSETS_PATH / f"test_{k}_out.jpg"), img2[0])

            assert len(img2) == 1
            assert calculate_image_similarity(img, img2[0])
            assert compare_image_size(img, img2[0], cfg.scale)


class Test_EDVRFeatureExtractor:
    def test_load(self) -> None:
        for k in [
            ConfigType.EDVRFeatureExtractor_REDS_pretrained_for_IconVSR,
            ConfigType.EDVRFeatureExtractor_Vimeo90K_pretrained_for_IconVSR,
        ]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: VSRBaseModel = AutoModel.from_config(config=cfg, device=CCCV_DEVICE, fp16=False)
            assert model is not None
