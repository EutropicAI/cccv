from typing import Union

from cccv.config import CONFIG_REGISTRY
from cccv.config.base_config import AuxiliaryBaseConfig
from cccv.type import ArchType, ConfigType, ModelType


class SpyNetConfig(AuxiliaryBaseConfig):
    arch: Union[ArchType, str] = ArchType.SPYNET
    model: Union[ModelType, str] = ModelType.SpyNet


SpyNetConfigs = [
    # BasicSR SpyNet
    SpyNetConfig(
        name=ConfigType.SpyNet_spynet_sintel_final,
        hash="3d2a1287666aa71752ebaedc06999212886ef476f77d691a1b0006107088e714",
    ),
]

for cfg in SpyNetConfigs:
    CONFIG_REGISTRY.register(cfg)
