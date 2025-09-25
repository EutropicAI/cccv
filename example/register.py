from typing import Any

from cccv import CONFIG_REGISTRY, MODEL_REGISTRY, ArchType, AutoModel, SRBaseModel
from cccv.config import RealESRGANConfig

# define your own config name and model name
cfg_name = "TESTCONFIG.pth"
model_name = "TESTMODEL"

# this should be your own config, not RealESRGANConfig
# extend from cccv.BaseConfig then implement your own config parameters
cfg = RealESRGANConfig(
    name=cfg_name,
    url="https://github.com/EutropicAI/cccv/releases/download/model_zoo/RealESRGAN_AnimeJaNai_HD_V3_Compact_2x.pth",
    arch=ArchType.SRVGG,
    model=model_name,
    scale=2,
)

CONFIG_REGISTRY.register(cfg)


# extend from cccv.SRBaseModel then implement your own model
@MODEL_REGISTRY.register(name=model_name)
class TESTMODEL(SRBaseModel):
    def load_model(self) -> Any:
        print("Override load_model function here")
        print("We use default load_model function to load the model")
        return super().load_model()


model: TESTMODEL = AutoModel.from_pretrained(cfg_name)
