from cccv import AutoConfig, AutoModel, BaseConfig, ConfigType
from cccv.model import VSRBaseModel

from .util import CCCV_DEVICE, CCCV_FP16, CCCV_TILE


class Test_SpyNet:
    def test_load(self) -> None:
        for k in [ConfigType.SpyNet_spynet_sintel_final]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: VSRBaseModel = AutoModel.from_config(config=cfg, device=CCCV_DEVICE, fp16=CCCV_FP16, tile=CCCV_TILE)
            assert model is not None
