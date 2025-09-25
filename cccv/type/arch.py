from enum import Enum


# Enum for the architecture type, use capital letters
class ArchType(str, Enum):
    # ------------------------------------- Auxiliary Network ----------------------------------------------------------

    SPYNET = "SPYNET"

    # ------------------------------------- Single Image Super-Resolution ----------------------------------------------

    RRDB = "RRDB"
    SRVGG = "SRVGG"
    UPCUNET = "UPCUNET"
    EDSR = "EDSR"
    SWINIR = "SWINIR"
    SCUNET = "SCUNET"
    DAT = "DAT"
    SRCNN = "SRCNN"
    HAT = "HAT"

    # ------------------------------------- Video Super-Resolution -----------------------------------------------------

    EDVR = "EDVR"
    MSRSWVSR = "MSRSWVSR"

    # ------------------------------------- Video Frame Interpolation --------------------------------------------------

    IFNET = "IFNET"
    DRBA = "DRBA"
