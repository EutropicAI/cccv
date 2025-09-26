import sys
import warnings
from inspect import signature
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import torch

from cccv.arch import ARCH_REGISTRY
from cccv.config import BaseConfig
from cccv.type import BaseModelInterface
from cccv.util.device import DEFAULT_DEVICE
from cccv.util.remote import load_file_from_url


class CCBaseModel(BaseModelInterface):
    """
    CCCV Base model

    :param config: config of the model
    :param device: inference device
    :param fp16: use fp16 precision or not
    :param compile: use torch.compile or not
    :param compile_backend: backend of torch.compile
    :param tile: tile size for tile inference, tile[0] is width, tile[1] is height, None for disable
    :param tile_pad: The padding size for each tile
    :param pad_img: The size for the padded image, pad[0] is width, pad[1] is height, None for auto calculate
    :param model_dir: The path to cache the downloaded model. Should be a full path. If None, use default cache path.
    :param gh_proxy: The proxy for downloading from github release. Example: https://github.abskoop.workers.dev/
    """

    def __init__(
        self,
        config: Any,
        device: Optional[torch.device] = None,
        fp16: bool = True,
        compile: bool = False,
        compile_backend: Optional[str] = None,
        tile: Optional[Tuple[int, int]] = (128, 128),
        tile_pad: int = 8,
        pad_img: Optional[Tuple[int, int]] = None,
        model_dir: Optional[Union[Path, str]] = None,
        gh_proxy: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # extra config
        self.one_frame_out: bool = False  # for vsr model type

        # load_state_dict
        self.load_state_dict_strict: bool = True

        # ---
        self.config = config
        self.device: Optional[torch.device] = device
        self.fp16: bool = fp16
        self.compile: bool = compile
        self.compile_backend: Optional[str] = compile_backend
        self.tile: Optional[Tuple[int, int]] = tile
        self.tile_pad: int = tile_pad
        self.pad_img: Optional[Tuple[int, int]] = pad_img
        self.model_dir: Optional[Union[Path, str]] = model_dir
        self.gh_proxy: Optional[str] = gh_proxy

        # post-hook: edit parameters here if needed
        self.post_init_hook()

        if device is None:
            self.device = DEFAULT_DEVICE

        self.model: torch.nn.Module = self.load_model()

        # fp16
        if self.fp16:
            try:
                self.model = self.model.half()
            except Exception as e:
                warnings.warn(f"[CCCV] {e}. fp16 is not supported on this model, fallback to fp32.", stacklevel=2)
                self.fp16 = False
                self.model = self.load_model()

        # compile
        if self.compile:
            try:
                if self.compile_backend is None:
                    if sys.platform == "darwin":
                        self.compile_backend = "aot_eager"
                    else:
                        self.compile_backend = "inductor"
                self.model = torch.compile(self.model, backend=self.compile_backend)
            except Exception as e:
                warnings.warn(f"[CCCV] {e}, compile is not supported on this model.", stacklevel=2)

    def post_init_hook(self) -> None:
        """
        Hook: Subclasses can override this method to perform any post-initialization processing.
        e.g. edit config parameters like `one_frame_out` for vsr model.
        By default, it does nothing.
        """
        pass

    def get_state_dict(self) -> Any:
        """
        Load the state dict of the model from config

        :return: The state dict of the model
        """
        cfg: BaseConfig = self.config

        if cfg.path is not None:
            state_dict_path = cfg.path
        else:
            try:
                state_dict_path = load_file_from_url(
                    config=cfg, force_download=False, model_dir=self.model_dir, gh_proxy=self.gh_proxy
                )
            except Exception as e:
                warnings.warn(f"[CCCV] Error: {e}, try force download the model...", stacklevel=2)
                state_dict_path = load_file_from_url(
                    config=cfg, force_download=True, model_dir=self.model_dir, gh_proxy=self.gh_proxy
                )

        state_dict = torch.load(state_dict_path, map_location=self.device, weights_only=True)

        return self.transform_state_dict(state_dict)

    def transform_state_dict(self, state_dict: Any) -> Any:
        """
        Hook: Subclasses can override this method to perform any key/value processing on the state_dict.

        :param state_dict: The original state dict
        :return: The transformed state dict
        """
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]
        elif "model_state_dict" in state_dict:
            # For APISR's model
            state_dict = state_dict["model_state_dict"]

        return state_dict

    def load_model(self) -> Any:
        """
        Auto load the model from config

        These params in nn.Module.load_state_dict can be overridden in post_init_hook if needed:

        - self.load_state_dict_strict -> strict

        - self.load_state_dict_assign -> assign

        :return: The initialized model with weights loaded
        """
        cfg: BaseConfig = self.config
        state_dict = self.get_state_dict()

        net = ARCH_REGISTRY.get(cfg.arch)
        cfg_dict = cfg.model_dump()

        try:
            net_kw = {k: cfg_dict[k] for k in signature(net).parameters}
        except (KeyError, TypeError) as e:
            raise RuntimeError(f"[CCCV] Config missing or mismatch required param for {net.__name__}: {e}") from e

        # print(f"[CCCV] net_kw: {net_kw}")
        model = net(**net_kw)

        model.load_state_dict(state_dict, strict=self.load_state_dict_strict)
        model.eval().to(self.device)
        return model

    def inference(self, *args: Any, **kwargs: Any) -> Any:
        """
        Inference the model with the inputs
        """
        raise NotImplementedError

    def inference_video(self, clip: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Inference the video with the model, the clip should be a vapoursynth clip

        :param clip: vs.VideoNode
        :return:
        """
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call the model for inference
        """
        return self.inference(*args, **kwargs)
