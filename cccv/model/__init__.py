# isort: skip_file
from cccv.util.registry import Registry

MODEL_REGISTRY: Registry = Registry("MODEL")

from cccv.model.tile import tile_sr, tile_vsr
from cccv.model.base_model import CCBaseModel
from cccv.model.auxiliary_base_model import AuxiliaryBaseModel
from cccv.model.sr_base_model import SRBaseModel
from cccv.model.vsr_base_model import VSRBaseModel

# Auxiliary Network

from cccv.model.auxnet.spynet_model import SpyNetModel

# Single Image Super-Resolution

from cccv.model.sr.realcugan_model import RealCUGANModel

# Video Super-Resolution

from cccv.model.vsr.edvr_model import EDVRModel
