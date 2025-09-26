import importlib.util
import json
from pathlib import Path
from typing import Any, Union

from cccv.config import CONFIG_REGISTRY, AutoBaseConfig
from cccv.type import ConfigType


class AutoConfig:
    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path: Union[ConfigType, str, Path],
        **kwargs: Any,
    ) -> Any:
        """
        Get a config instance of a pretrained model configuration.

        :param pretrained_model_name_or_path: The name or path of the pretrained model configuration
        :return:
        """
        if "pretrained_model_name" in kwargs:
            print(
                "[CCCV] warning: 'pretrained_model_name' is deprecated, please use 'pretrained_model_name_or_path' instead."
            )
            pretrained_model_name_or_path = kwargs.pop("pretrained_model_name")

        # 1. check if it's a registered config name
        if isinstance(pretrained_model_name_or_path, ConfigType):
            pretrained_model_name_or_path = pretrained_model_name_or_path.value
        if str(pretrained_model_name_or_path) in CONFIG_REGISTRY:
            return CONFIG_REGISTRY.get(str(pretrained_model_name_or_path))

        # 2. check if it's a real path
        dir_path = Path(str(pretrained_model_name_or_path))

        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"[CCCV] model configuration '{dir_path}' is not a valid config name or path")

        # load config,json from the directory
        config_path = dir_path / "config.json"
        # check if config.json exists
        if not config_path.exists():
            raise FileNotFoundError(f"[CCCV] no valid config.json not found in {dir_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        for k in ["arch", "model", "name"]:
            if k not in config_dict:
                raise KeyError(
                    f"[CCCV] no key '{k}' in config.json in {dir_path}, you should provide a valid config.json contain a key '{k}'"
                )

        # auto import all .py files in the directory to register the arch, model and config
        for py_file in dir_path.glob("*.py"):
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        config_dict["path"] = str(dir_path / config_dict["name"])

        # convert config_dict to pydantic model
        cfg = AutoBaseConfig.model_validate(config_dict)
        return cfg
