from typing import Any

from cccv.model import MODEL_REGISTRY
from cccv.model.base_model import CCBaseModel
from cccv.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.AuxiliaryBaseModel)
class AuxiliaryBaseModel(CCBaseModel):
    def inference(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("[CCCV] Auxiliary model should use self.model to load in the main model")
