from typing import Any

import cv2

from cccv import CONFIG_REGISTRY, MODEL_REGISTRY, ArchType, AutoModel
from cccv.config import RealESRGANConfig
from cccv.model import SRBaseModel
from tests.util import (
    ASSETS_PATH,
    CCCV_DEVICE,
    CCCV_FP16,
    CCCV_TILE,
    calculate_image_similarity,
    load_image,
)


def test_auto_class_register() -> None:
    cfg_name = "TESTCONFIG.pth"
    model_name = "TESTMODEL"

    cfg = RealESRGANConfig(
        name=cfg_name,
        url="https://github.com/EutropicAI/cccv/releases/download/model_zoo/RealESRGAN_AnimeJaNai_HD_V3_Compact_2x.pth",
        arch=ArchType.SRVGG,
        model=model_name,
        scale=2,
    )

    CONFIG_REGISTRY.register(cfg)

    @MODEL_REGISTRY.register(name=model_name)
    class TESTMODEL(SRBaseModel):
        def get_cfg(self) -> Any:
            return self.config

    model: TESTMODEL = AutoModel.from_pretrained(cfg_name)
    assert model.get_cfg() == cfg


class Test_AutoModel:
    def test_model_from_remote_repo(self) -> None:
        model: SRBaseModel = AutoModel.from_pretrained(
            "https://github.com/EutropicAI/cccv_demo_remote_model", device=CCCV_DEVICE, fp16=CCCV_FP16, tile=CCCV_TILE
        )

        img1 = load_image()
        img2 = model.inference_image(img1)

        cv2.imwrite(str(ASSETS_PATH / "test_remote_repo_test_out.jpg"), img2)

        assert calculate_image_similarity(img1, img2)
