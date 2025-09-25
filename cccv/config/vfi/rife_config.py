from typing import Union

from cccv.config import CONFIG_REGISTRY
from cccv.config.base_config import VFIBaseConfig
from cccv.type import ArchType, ConfigType, ModelType


class RIFEConfig(VFIBaseConfig):
    arch: Union[ArchType, str] = ArchType.IFNET
    model: Union[ModelType, str] = ModelType.RIFE
    num_frame: int = 2


RIFEConfigs = [
    RIFEConfig(
        name=ConfigType.RIFE_IFNet_v426_heavy,
        hash="4cc518e172156ad6207b9c7a43364f518832d83a4325d484240493a9e2980537",
    )
]

for cfg in RIFEConfigs:
    CONFIG_REGISTRY.register(cfg)
