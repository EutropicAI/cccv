import importlib.util
import json
from pathlib import Path
from typing import Any, Optional, Union

from cccv.config import CONFIG_REGISTRY, AutoBaseConfig
from cccv.type import ConfigType
from cccv.util.remote import git_clone


class AutoConfig:
    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path: Union[ConfigType, str, Path],
        *,
        model_dir: Optional[Union[Path, str]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Get a config instance of a pretrained model configuration, can be a registered config name or a local path or a git url.

        :param pretrained_model_name_or_path:
        :param model_dir: The path to cache the downloaded model configuration. Should be a full path. If None, use default cache path.
        :return:
        """
        # 1. check if it's a registered config name, early return if found
        if isinstance(pretrained_model_name_or_path, ConfigType):
            pretrained_model_name_or_path = pretrained_model_name_or_path.value
        if str(pretrained_model_name_or_path) in CONFIG_REGISTRY:
            return CONFIG_REGISTRY.get(str(pretrained_model_name_or_path))

        # 2. check is a url or not, if it's a url, git clone it to model_dir then replace pretrained_model_name_or_path with the local path (Path)
        if str(pretrained_model_name_or_path).startswith("http"):
            pretrained_model_name_or_path = git_clone(
                git_url=str(pretrained_model_name_or_path),
                model_dir=model_dir,
                **kwargs,
            )

        # 3. check if it's a real path
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
        try:
            for py_file in dir_path.glob("*.py"):
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
        except Exception as e:
            raise ImportError(f"[CCCV] failed register model from {dir_path}, error: {e}, please check your .py files")

        if "path" not in config_dict or config_dict["path"] is None or config_dict["path"] == "":
            # add the path to the config_dict
            config_dict["path"] = str(dir_path / config_dict["name"])

        # convert config_dict to pydantic model
        cfg = AutoBaseConfig.model_validate(config_dict)
        return cfg
