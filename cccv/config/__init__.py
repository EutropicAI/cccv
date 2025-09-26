# isort: skip_file
from cccv.util.registry import RegistryConfigInstance

CONFIG_REGISTRY: RegistryConfigInstance = RegistryConfigInstance("CONFIG")

from cccv.config.base_config import BaseConfig, SRBaseConfig, VSRBaseConfig, VFIBaseConfig, AutoBaseConfig

# Auxiliary Network

from cccv.config.auxnet.spynet_config import SpyNetConfig

# Single Image Super-Resolution

from cccv.config.sr.realesrgan_config import RealESRGANConfig
from cccv.config.sr.realcugan_config import RealCUGANConfig
from cccv.config.sr.edsr_config import EDSRConfig
from cccv.config.sr.swinir_config import SwinIRConfig
from cccv.config.sr.scunet_config import SCUNetConfig
from cccv.config.sr.dat_config import DATConfig
from cccv.config.sr.srcnn_config import SRCNNConfig
from cccv.config.sr.hat_config import HATConfig

# Video Super-Resolution

from cccv.config.vsr.edvr_config import EDVRConfig
from cccv.config.vsr.animesr_config import AnimeSRConfig

# Video Frame Interpolation
from cccv.config.vfi.rife_config import RIFEConfig
from cccv.config.vfi.drba_config import DRBAConfig
