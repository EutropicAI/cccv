from cccv.util.registry import Registry

ARCH_REGISTRY: Registry = Registry("ARCH")

from cccv.arch.rrdb_arch import RRDBNet  # noqa
from cccv.arch.srvgg_arch import SRVGGNetCompact  # noqa
from cccv.arch.upcunet_arch import UpCunet  # noqa
from cccv.arch.edsr_arch import EDSR  # noqa
from cccv.arch.swinir_arch import SwinIR  # noqa
from cccv.arch.edvr_arch import EDVR  # noqa
from cccv.arch.spynet_arch import SpyNet  # noqa
from cccv.arch.msrswvsr_arch import MSRSWVSR  # noqa
from cccv.arch.scunet_arch import SCUNet  # noqa
from cccv.arch.dat_arch import DAT  # noqa
from cccv.arch.srcnn_arch import SRCNN  # noqa
from cccv.arch.hat_arch import HAT  # noqa
