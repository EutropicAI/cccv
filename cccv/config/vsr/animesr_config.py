from typing import Tuple, Union

from cccv.config import CONFIG_REGISTRY
from cccv.config.base_config import VSRBaseConfig
from cccv.type import ArchType, ConfigType, ModelType


class AnimeSRConfig(VSRBaseConfig):
    arch: Union[ArchType, str] = ArchType.MSRSWVSR
    model: Union[ModelType, str] = ModelType.VSRBaseModel
    scale: int = 4
    num_frame: int = 3
    num_feat: int = 64
    num_block: Tuple[int, int, int] = (5, 3, 2)


AnimeSRConfigs = [
    AnimeSRConfig(
        name=ConfigType.AnimeSR_v1_PaperModel_4x,
        hash="915ef7f0f7067f04219516b50e88c362581300e48902e3b7f540650e32a20c10",
        scale=4,
    ),
    AnimeSRConfig(
        name=ConfigType.AnimeSR_v2_4x,
        hash="d0f29c8966b53718828bd424bbdc306e7ff0cbf6350beadaf8b5b2500b108548",
        scale=4,
    ),
]

for cfg in AnimeSRConfigs:
    CONFIG_REGISTRY.register(cfg)
