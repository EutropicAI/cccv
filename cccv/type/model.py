from enum import Enum


# Enum for model type, use original name
class ModelType(str, Enum):
    # ------------------------------------- Auxiliary Network ----------------------------------------------------------

    SpyNet = "SpyNet"
    EDVRFeatureExtractor = "EDVRFeatureExtractor"

    # ------------------------------------- Single Image Super-Resolution ----------------------------------------------

    RealESRGAN = "RealESRGAN"
    RealCUGAN = "RealCUGAN"
    EDSR = "EDSR"
    SwinIR = "SwinIR"
    SCUNet = "SCUNet"
    DAT = "DAT"
    SRCNN = "SRCNN"
    HAT = "HAT"

    # ------------------------------------- Video Super-Resolution -----------------------------------------------------

    EDVR = "EDVR"
    AnimeSR = "AnimeSR"
