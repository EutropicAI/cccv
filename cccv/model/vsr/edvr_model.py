from cccv.model import MODEL_REGISTRY
from cccv.model.vsr_base_model import VSRBaseModel
from cccv.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.EDVR)
class EDVRModel(VSRBaseModel):
    def post_init_hook(self) -> None:
        self.one_frame_out = True
