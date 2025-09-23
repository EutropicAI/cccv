from typing import Union

from cccv.config import CONFIG_REGISTRY
from cccv.config.base_config import SRBaseConfig
from cccv.type import ArchType, ConfigType, ModelType


class EDSRConfig(SRBaseConfig):
    arch: Union[ArchType, str] = ArchType.EDSR
    model: Union[ModelType, str] = ModelType.SRBaseModel
    scale: int = 2
    num_in_ch: int = 3
    num_out_ch: int = 3
    num_feat: int = 64
    num_block: int = 16
    res_scale: int = 1
    img_range: float = 255.0
    rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040)


EDSRConfigs = [
    # Official Medium size models
    EDSRConfig(
        name=ConfigType.EDSR_Mx2_f64b16_DIV2K_official_2x,
        hash="3ba7b0861913de93740110738fb621410651897e391e8057b7b6104c4f999254",
        scale=2,
    ),
    EDSRConfig(
        name=ConfigType.EDSR_Mx3_f64b16_DIV2K_official_3x,
        hash="6908f88a1be95e7112f480b7b1d9608ad83b4ffa0c227416a6376f6b036a77f3",
        scale=3,
    ),
    EDSRConfig(
        name=ConfigType.EDSR_Mx4_f64b16_DIV2K_official_4x,
        hash="0c287733e70d1b8b8fc5885613ecbe451e5f3010bcae0307612ef5e4aa08dd5f",
        scale=4,
    ),
]

for cfg in EDSRConfigs:
    CONFIG_REGISTRY.register(cfg)
