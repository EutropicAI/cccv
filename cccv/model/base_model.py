import sys
import warnings
from inspect import signature
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

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
    :param fp16: use fp16 (half) precision or not
    :param bf16: use bf16 (bfloat16) precision or not, takes precedence over fp16; wider dynamic range than fp16 to avoid overflow/NaN on some models (e.g. transformer-based SR)
    :param compile: use torch.compile or not
    :param compile_backend: backend of torch.compile
    :param tile: tile size for tile inference, tile[0] is width, tile[1] is height, None for disable
    :param tile_pad: The padding size for each tile
    :param pad_img: The size for the padded image, pad[0] is width, pad[1] is height, None for auto calculate
    :param bf16_preflight: run a small bf16 inference before actual user inference, fallback if it fails
    :param bf16_preflight_size: The bf16 preflight input size as (height, width), aligned per model if needed
    :param model_dir: The path to cache the downloaded model. Should be a full path. If None, use default cache path.
    :param gh_proxy: The proxy for downloading from github release. Example: https://github.abskoop.workers.dev/
    """

    def __init__(
        self,
        config: Any,
        device: Optional[torch.device] = None,
        fp16: bool = True,
        bf16: bool = False,
        compile: bool = False,
        compile_backend: Optional[str] = None,
        tile: Optional[Tuple[int, int]] = (128, 128),
        tile_pad: int = 8,
        pad_img: Optional[Tuple[int, int]] = None,
        bf16_preflight: bool = True,
        bf16_preflight_size: Tuple[int, int] = (64, 64),
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
        self.bf16: bool = bf16
        # half precision dtype, bf16 takes precedence over fp16
        self.half_dtype: torch.dtype = torch.bfloat16 if bf16 else torch.float16
        self.compile: bool = compile
        self.compile_backend: Optional[str] = compile_backend
        self.tile: Optional[Tuple[int, int]] = tile
        self.tile_pad: int = tile_pad
        self.pad_img: Optional[Tuple[int, int]] = pad_img
        self.bf16_preflight: bool = bf16_preflight
        self.bf16_preflight_size: Tuple[int, int] = bf16_preflight_size
        self.model_dir: Optional[Union[Path, str]] = model_dir
        self.gh_proxy: Optional[str] = gh_proxy

        # post-hook: edit parameters here if needed
        self.post_init_hook()

        if device is None:
            self.device = DEFAULT_DEVICE

        self.model: torch.nn.Module = self.load_model()

        # half precision (fp16 or bf16, bf16 takes precedence)
        if self.fp16 or self.bf16:
            self._try_enable_half_precision()

        # compile
        if self.compile:
            self._try_compile_model()

        if self.bf16 and self.bf16_preflight:
            self._run_bf16_preflight_or_fallback()

    def _try_enable_half_precision(self) -> None:
        self.half_dtype = torch.bfloat16 if self.bf16 else torch.float16

        try:
            self.model = self.model.to(self.half_dtype)
        except Exception as e:
            if self.bf16 and self.fp16:
                warnings.warn(f"[CCCV] {e}. bf16 is not supported on this model, fallback to fp16.", stacklevel=2)
                self.bf16 = False
                self.half_dtype = torch.float16
                self.model = self.load_model()
                try:
                    self.model = self.model.to(self.half_dtype)
                    return
                except Exception as fp16_error:
                    warnings.warn(f"[CCCV] {fp16_error}. fp16 fallback failed, fallback to fp32.", stacklevel=2)
                    self.fp16 = False
                    self.half_dtype = torch.float32
                    self.model = self.load_model()
                    return

            warnings.warn(f"[CCCV] {e}. half precision is not supported on this model, fallback to fp32.", stacklevel=2)
            self.bf16 = False
            self.fp16 = False
            self.half_dtype = torch.float32
            self.model = self.load_model()

    def _try_compile_model(self) -> None:
        try:
            if self.compile_backend is None:
                if sys.platform == "darwin":
                    self.compile_backend = "aot_eager"
                else:
                    self.compile_backend = "inductor"
            self.model = torch.compile(self.model, backend=self.compile_backend)
        except Exception as e:
            warnings.warn(f"[CCCV] {e}, compile is not supported on this model.", stacklevel=2)

    def _run_bf16_preflight_or_fallback(self) -> None:
        preflight = None
        try:
            preflight = self.get_bf16_preflight_inputs()
            if preflight is None:
                return

            preflight_args, preflight_kwargs = preflight
            with torch.inference_mode():
                out = self.inference(*preflight_args, **preflight_kwargs)
            self._synchronize_device()
            del out
        except Exception as e:
            warnings.warn(
                f"[CCCV] {e}. bf16 inference is not supported on this device, disabling bf16 as fallback.",
                stacklevel=2,
            )
            self._fallback_from_bf16()
        finally:
            preflight = None
            self._empty_device_cache()

    def _fallback_from_bf16(self) -> None:
        self.bf16 = False
        self.half_dtype = torch.float16 if self.fp16 else torch.float32
        self.model = self.load_model()

        if self.fp16 or self.bf16:
            try:
                self.model = self.model.to(self.half_dtype)
            except Exception as e:
                warnings.warn(f"[CCCV] {e}. fp16 fallback failed, fallback to fp32.", stacklevel=2)
                self.fp16 = False
                self.half_dtype = torch.float32
                self.model = self.load_model()

        if self.compile:
            self._try_compile_model()

    def _synchronize_device(self) -> None:
        if self.device is not None and torch.device(self.device).type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(self.device))

    def _empty_device_cache(self) -> None:
        if self.device is not None and torch.device(self.device).type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _tensor_to_numpy(tensor: torch.Tensor) -> Any:
        tensor = tensor.cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy()

    def get_bf16_preflight_inputs(self) -> Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]]:
        """
        Hook: Subclasses can return representative inference args/kwargs for bf16 device validation.

        By default, no preflight runs because the base model does not know the input contract.
        """
        return None

    def _get_bf16_preflight_image_size(self, multiple: int = 1) -> Tuple[int, int]:
        height, width = self.bf16_preflight_size
        input_multiple = max(1, multiple, self._infer_bf16_preflight_multiple())
        return self._ceil_to_multiple(height, input_multiple), self._ceil_to_multiple(width, input_multiple)

    def _infer_bf16_preflight_multiple(self) -> int:
        multiples = []

        window_size = getattr(self.config, "window_size", None)
        if isinstance(window_size, int):
            multiples.append(window_size)

        split_size = getattr(self.config, "split_size", None)
        if isinstance(split_size, int):
            multiples.append(split_size)
        elif isinstance(split_size, (list, tuple)):
            multiples.extend(value for value in split_size if isinstance(value, int))

        return max(multiples) if multiples else 1

    @staticmethod
    def _ceil_to_multiple(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple

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
