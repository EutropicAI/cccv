from typing import Union

from cccv.config import CONFIG_REGISTRY
from cccv.config.base_config import VFIBaseConfig
from cccv.type import ArchType, ConfigType, ModelType


class DRBAConfig(VFIBaseConfig):
    arch: Union[ArchType, str] = ArchType.DRBA
    model: Union[ModelType, str] = ModelType.DRBA
    num_frame: int = 3


DRBAConfigs = [
    DRBAConfig(
        name=ConfigType.DRBA_IFNet,
        hash="4cc518e172156ad6207b9c7a43364f518832d83a4325d484240493a9e2980537",
    )
]

for cfg in DRBAConfigs:
    CONFIG_REGISTRY.register(cfg)
