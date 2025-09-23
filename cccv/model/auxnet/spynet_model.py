from typing import Any

import torch

from cccv.arch import SpyNet
from cccv.model import MODEL_REGISTRY
from cccv.model.auxiliary_base_model import AuxiliaryBaseModel
from cccv.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.SpyNet)
class SpyNetModel(AuxiliaryBaseModel):
    def load_model(self) -> Any:
        state_dict = self.get_state_dict()

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        model = SpyNet()

        model.load_state_dict(state_dict)

        model.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        model.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        model.eval().to(self.device)
        return model
