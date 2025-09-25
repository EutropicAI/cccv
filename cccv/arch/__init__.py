# isort: skip_file
from cccv.util.registry import Registry

ARCH_REGISTRY: Registry = Registry("ARCH")

# Auxiliary Network

from cccv.arch.auxnet.spynet_arch import SpyNet

# Single Image Super-Resolution

from cccv.arch.sr.rrdb_arch import RRDBNet
from cccv.arch.sr.srvgg_arch import SRVGGNetCompact
from cccv.arch.sr.upcunet_arch import UpCunet
from cccv.arch.sr.edsr_arch import EDSR
from cccv.arch.sr.swinir_arch import SwinIR
from cccv.arch.sr.scunet_arch import SCUNet
from cccv.arch.sr.dat_arch import DAT
from cccv.arch.sr.srcnn_arch import SRCNN
from cccv.arch.sr.hat_arch import HAT

# Video Super-Resolution

from cccv.arch.vsr.edvr_arch import EDVR
from cccv.arch.vsr.msrswvsr_arch import MSRSWVSR

# Video Frame Interpolation
from cccv.arch.vfi.ifnet_arch import IFNet
from cccv.arch.vfi.drba_arch import DRBA
