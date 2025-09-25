from cccv import AutoModel, BaseModelInterface, ConfigType

# --- sisr, use fp16 to inference

model: BaseModelInterface = AutoModel.from_pretrained(ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x)


# ---  use fp32 to inference

model = AutoModel.from_pretrained(
    ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x,
    fp16=False,
)

# --- vsr

model = AutoModel.from_pretrained(ConfigType.AnimeSR_v2_4x)

# --- torch.compile

model = AutoModel.from_pretrained(
    ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x,
    compile=True,
    # compile_backend="inductor",
)

# --- disable tile processing

model = AutoModel.from_pretrained(
    ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x,
    tile=None,
)
