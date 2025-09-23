from cccv.util.registry import RegistryConfigInstance

CONFIG_REGISTRY: RegistryConfigInstance = RegistryConfigInstance("CONFIG")

from cccv.config.realesrgan_config import RealESRGANConfig  # noqa
from cccv.config.realcugan_config import RealCUGANConfig  # noqa
from cccv.config.edsr_config import EDSRConfig  # noqa
from cccv.config.swinir_config import SwinIRConfig  # noqa
from cccv.config.edvr_config import EDVRConfig, EDVRFeatureExtractorConfig  # noqa
from cccv.config.spynet_config import SpyNetConfig  # noqa
from cccv.config.basicvsr_config import BasicVSRConfig  # noqa
from cccv.config.iconvsr_config import IconVSRConfig  # noqa
from cccv.config.animesr_config import AnimeSRConfig  # noqa
from cccv.config.scunet_config import SCUNetConfig  # noqa
from cccv.config.dat_config import DATConfig  # noqa
from cccv.config.srcnn_config import SRCNNConfig  # noqa
from cccv.config.hat_config import HATConfig  # noqa
