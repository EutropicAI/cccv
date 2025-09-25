from typing import Any

from cccv import CONFIG_REGISTRY, MODEL_REGISTRY, ArchType, AutoModel
from cccv.config import RealESRGANConfig
from cccv.model import SRBaseModel


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
