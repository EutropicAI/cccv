from cccv.util.registry import Registry

ARCH_REGISTRY: Registry = Registry("ARCH")

# Auxiliary Network

from cccv.arch.auxnet.spynet_arch import SpyNet  # noqa

# Single Image Super-Resolution

from cccv.arch.sr.rrdb_arch import RRDBNet  # noqa
from cccv.arch.sr.srvgg_arch import SRVGGNetCompact  # noqa
from cccv.arch.sr.upcunet_arch import UpCunet  # noqa
from cccv.arch.sr.edsr_arch import EDSR  # noqa
from cccv.arch.sr.swinir_arch import SwinIR  # noqa
from cccv.arch.sr.scunet_arch import SCUNet  # noqa
from cccv.arch.sr.dat_arch import DAT  # noqa
from cccv.arch.sr.srcnn_arch import SRCNN  # noqa
from cccv.arch.sr.hat_arch import HAT  # noqa

# Video Super-Resolution

from cccv.arch.vsr.edvr_arch import EDVR  # noqa
from cccv.arch.vsr.msrswvsr_arch import MSRSWVSR  # noqa
