from cccv.util.registry import Registry

MODEL_REGISTRY: Registry = Registry("MODEL")

from cccv.model.sr_base_model import SRBaseModel  # noqa
from cccv.model.vsr_base_model import VSRBaseModel  # noqa
from cccv.model.auxiliary_base_model import AuxiliaryBaseModel  # noqa
from cccv.model.realesrgan_model import RealESRGANModel  # noqa
from cccv.model.realcugan_model import RealCUGANModel  # noqa
from cccv.model.edsr_model import EDSRModel  # noqa
from cccv.model.swinir_model import SwinIRModel  # noqa
from cccv.model.edvr_model import EDVRModel, EDVRFeatureExtractorModel  # noqa
from cccv.model.tile import tile_sr, tile_vsr  # noqa
from cccv.model.spynet_model import SpyNetModel  # noqa
from cccv.model.animesr_model import AnimeSRModel  # noqa
from cccv.model.scunet_model import SCUNetModel  # noqa
from cccv.model.dat_model import DATModel  # noqa
from cccv.model.srcnn_model import SRCNNModel  # noqa
from cccv.model.hat_model import HATModel  # noqa
