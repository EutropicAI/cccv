from pathlib import Path
from typing import Any, Optional, Tuple, Union

import torch

from cccv.auto.config import AutoConfig
from cccv.config import BaseConfig
from cccv.model import MODEL_REGISTRY
from cccv.type import ConfigType


class AutoModel:
    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path: Union[ConfigType, str, Path],
        *,
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
    ) -> Any:
        """
        Get a model instance from a registered config name or a local path or a git url.

        :param pretrained_model_name_or_path:
        :param device: inference device
        :param fp16: use fp16 precision or not
        :param compile: use torch.compile or not
        :param compile_backend: backend of torch.compile
        :param tile: tile size for tile inference, tile[0] is width, tile[1] is height, None for disable
        :param tile_pad: The padding size for each tile
        :param pad_img: The size for the padded image, pad[0] is width, pad[1] is height, None for auto calculate
        :param model_dir: The path to cache the downloaded model. Should be a full path. If None, use default cache path.
        :param gh_proxy: The proxy for downloading from github release. Example: https://github.abskoop.workers.dev/
        :return:
        """
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, model_dir=model_dir, **kwargs)

        return AutoModel.from_config(
            config=config,
            device=device,
            fp16=fp16,
            compile=compile,
            compile_backend=compile_backend,
            tile=tile,
            tile_pad=tile_pad,
            pad_img=pad_img,
            model_dir=model_dir,
            gh_proxy=gh_proxy,
            **kwargs,
        )

    @staticmethod
    def from_config(
        config: Union[BaseConfig, Any],
        *,
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
    ) -> Any:
        """
        Get a model instance from a config.

        :param config: The config object. We suggest use cccv.BaseConfig or its subclass.
        :param device: inference device
        :param fp16: use fp16 precision or not
        :param compile: use torch.compile or not
        :param compile_backend: backend of torch.compile
        :param tile: tile size for tile inference, tile[0] is width, tile[1] is height, None for disable
        :param tile_pad: The padding size for each tile
        :param pad_img: The size for the padded image, pad[0] is width, pad[1] is height, None for auto calculate
        :param model_dir: The path to cache the downloaded model. Should be a full path. If None, use default cache path.
        :param gh_proxy: The proxy for downloading from github release. Example: https://github.abskoop.workers.dev/
        :return:
        """
        model = MODEL_REGISTRY.get(config.model)
        model = model(
            config=config,
            device=device,
            fp16=fp16,
            compile=compile,
            compile_backend=compile_backend,
            tile=tile,
            tile_pad=tile_pad,
            pad_img=pad_img,
            model_dir=model_dir,
            gh_proxy=gh_proxy,
            **kwargs,
        )

        return model
