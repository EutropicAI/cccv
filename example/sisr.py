import cv2
import numpy as np

from cccv import ArchType, AutoConfig, AutoModel, BaseConfig, ConfigType, SRBaseModel
from cccv.config import RealESRGANConfig

example = 3

if example == 0:
    # fast load a pre-trained model
    model: SRBaseModel = AutoModel.from_pretrained(ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x)
elif example == 1:
    # edit the configuration
    config: BaseConfig = AutoConfig.from_pretrained(
        pretrained_model_name=ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x
    )
    print(config)
    config.scale = 2
    model: SRBaseModel = AutoModel.from_config(config=config)
elif example == 3:
    # use your own configuration
    config = RealESRGANConfig(
        name="114514.pth",
        url="https://github.com/EutropicAI/cccv/releases/download/model_zoo/RealESRGAN_AnimeJaNai_HD_V3_Compact_2x.pth",
        hash="af7307eee19e5982a8014dd0e4650d3bde2e25aa78d2105a4bdfd947636e4c8f",
        arch=ArchType.SRVGG,
        scale=2,
    )
    model: SRBaseModel = AutoModel.from_config(config=config)
elif example == 4:
    # use custom model dir and gh proxy
    model: SRBaseModel = AutoModel.from_pretrained(
        pretrained_model_name=ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x,
        model_dir="./",
        gh_proxy="https://github.abskoop.workers.dev/",
    )


else:
    raise ValueError("example not found")


img = cv2.imdecode(np.fromfile("../assets/test.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
img = model.inference_image(img)
cv2.imwrite("../assets/test_sisr_out.jpg", img)
