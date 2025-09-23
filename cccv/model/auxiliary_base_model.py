from typing import Any

from cccv.model.base_model import CCBaseModel


class AuxiliaryBaseModel(CCBaseModel):
    def inference(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Auxiliary model should use self.model to load in the main model")
