from typing import Any

from cccv import AutoConfig, AutoModel
from cccv.config import RealESRGANConfig
from cccv.model import SRBaseModel


def test_auto_class_register() -> None:
    cfg_name = "TESTCONFIG.pth"
    model_name = "TESTMODEL"

    cfg = RealESRGANConfig(
        name=cfg_name,
        url="https://github.com/EutropicAI/cccv/releases/download/model_zoo/RealESRGAN_RealESRGAN_x4plus_anime_6B_4x.pth",
        arch="RRDB",
        model=model_name,
        scale=4,
        num_block=6,
    )

    AutoConfig.register(cfg)

    @AutoModel.register(name=model_name)
    class TESTMODEL(SRBaseModel):
        def get_cfg(self) -> Any:
            return self.config

    model: TESTMODEL = AutoModel.from_pretrained(cfg_name)
    assert model.get_cfg() == cfg
