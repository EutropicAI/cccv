from typing import Any

import cv2

from cccv import CONFIG_REGISTRY, MODEL_REGISTRY, ArchType, AutoConfig, AutoModel, BaseConfig, ConfigType
from cccv.config import RealESRGANConfig
from cccv.model import SRBaseModel
from cccv.util.remote import git_clone
from tests.util import (
    ASSETS_PATH,
    CCCV_DEVICE,
    CCCV_FP16,
    CCCV_TILE,
    calculate_image_similarity,
    compare_image_size,
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


class Test_AutoConfig:
    def test_registered_config(self) -> None:
        cfg = AutoConfig.from_pretrained(ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x)
        assert isinstance(cfg, BaseConfig)
        assert cfg.name == ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x

    def test_config_from_path(self) -> None:
        clone_dir = git_clone("https://github.com/EutropicAI/cccv_demo_remote_model")

        cfg: BaseConfig = AutoConfig.from_pretrained(clone_dir)
        print(cfg)
        img1 = load_image()
        model: SRBaseModel = AutoModel.from_config(config=cfg, device=CCCV_DEVICE, fp16=CCCV_FP16, tile=CCCV_TILE)

        img2 = model.inference_image(img1)

        cv2.imwrite(str(ASSETS_PATH / f"test_{cfg.name}_out.jpg"), img2)

        assert calculate_image_similarity(img1, img2)
        assert compare_image_size(img1, img2, cfg.scale)
