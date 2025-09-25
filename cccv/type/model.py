from enum import Enum


# Enum for model type, use original name
class ModelType(str, Enum):
    # ------------------------------------- Auxiliary Network ----------------------------------------------------------
    AuxiliaryBaseModel = "AuxiliaryBaseModel"

    SpyNet = "SpyNet"

    # ------------------------------------- Single Image Super-Resolution ----------------------------------------------
    SRBaseModel = "SRBaseModel"

    RealESRGAN = "RealESRGAN"
    RealCUGAN = "RealCUGAN"
    EDSR = "EDSR"
    SwinIR = "SwinIR"
    SCUNet = "SCUNet"
    DAT = "DAT"
    SRCNN = "SRCNN"
    HAT = "HAT"

    # ------------------------------------- Video Super-Resolution -----------------------------------------------------
    VSRBaseModel = "VSRBaseModel"

    EDVR = "EDVR"
    AnimeSR = "AnimeSR"

    # ------------------------------------- Video Frame Interpolation --------------------------------------------------
    VFIBaseModel = "VFIBaseModel"

    RIFE = "RIFE"
    DRBA = "DRBA"
