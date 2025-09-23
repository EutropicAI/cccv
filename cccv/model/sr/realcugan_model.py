from typing import Any

from cccv.config import RealCUGANConfig
from cccv.model import MODEL_REGISTRY
from cccv.model.sr_base_model import SRBaseModel
from cccv.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.RealCUGAN)
class RealCUGANModel(SRBaseModel):
    def transform_state_dict(self, state_dict: Any) -> Any:
        cfg: RealCUGANConfig = self.config

        if cfg.pro:
            del state_dict["pro"]

        new_state_dict = {}
        for key, value in state_dict.items():
            # 修改键，添加"unet."前缀
            new_key = "unet." + key
            new_state_dict[new_key] = value

        return new_state_dict
