from cccv.util.registry import RegistryConfigInstance

CONFIG_REGISTRY: RegistryConfigInstance = RegistryConfigInstance("CONFIG")

from cccv.config.base_config import BaseConfig, SRBaseConfig, VSRBaseConfig  # noqa

# Auxiliary Network

from cccv.config.auxnet.spynet_config import SpyNetConfig  # noqa

# Single Image Super-Resolution

from cccv.config.sr.realesrgan_config import RealESRGANConfig  # noqa
from cccv.config.sr.realcugan_config import RealCUGANConfig  # noqa
from cccv.config.sr.edsr_config import EDSRConfig  # noqa
from cccv.config.sr.swinir_config import SwinIRConfig  # noqa
from cccv.config.sr.scunet_config import SCUNetConfig  # noqa
from cccv.config.sr.dat_config import DATConfig  # noqa
from cccv.config.sr.srcnn_config import SRCNNConfig  # noqa
from cccv.config.sr.hat_config import HATConfig  # noqa

# Video Super-Resolution

from cccv.config.vsr.edvr_config import EDVRConfig  # noqa
from cccv.config.vsr.animesr_config import AnimeSRConfig  # noqa
