from cccv.util.registry import Registry

MODEL_REGISTRY: Registry = Registry("MODEL")

from cccv.model.tile import tile_sr, tile_vsr  # noqa
from cccv.model.base_model import CCBaseModel  # noqa
from cccv.model.auxiliary_base_model import AuxiliaryBaseModel  # noqa
from cccv.model.sr_base_model import SRBaseModel  # noqa
from cccv.model.vsr_base_model import VSRBaseModel  # noqa

# Auxiliary Network

from cccv.model.auxnet.spynet_model import SpyNetModel  # noqa

# Single Image Super-Resolution

from cccv.model.sr.realcugan_model import RealCUGANModel  # noqa

# Video Super-Resolution

from cccv.model.vsr.edvr_model import EDVRModel  # noqa
from cccv.model.vsr.animesr_model import AnimeSRModel  # noqa
