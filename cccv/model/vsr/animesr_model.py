from typing import Any

from cccv.arch import MSRSWVSR
from cccv.config import AnimeSRConfig
from cccv.model import MODEL_REGISTRY
from cccv.model.vsr_base_model import VSRBaseModel
from cccv.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.AnimeSR)
class AnimeSRModel(VSRBaseModel):
    def load_model(self) -> Any:
        cfg: AnimeSRConfig = self.config
        state_dict = self.get_state_dict()

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        model = MSRSWVSR(
            num_feat=cfg.num_feat,
            num_block=cfg.num_block,
            netscale=cfg.scale,
        )

        model.load_state_dict(state_dict)
        model.eval().to(self.device)
        return model
